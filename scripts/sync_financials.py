#!/usr/bin/env python3
"""
One-shot financial data sync: pull from Tushare → local parquet cache.

Usage:
    python scripts/sync_financials.py --ts-codes data/stock_basic/stock_basic.parquet --quarters 12
    python scripts/sync_financials.py --ts-codes data/stock_basic/stock_basic.parquet --upsert

Idempotent: --upsert merges new quarters into existing data, skipping duplicates.
Goes full speed; only backs off on Tushare rate-limit errors with exponential backoff.
"""

import argparse
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "financial" / "financial_quarters.parquet"
# Rate limiting: go full speed with minimal gap, only back off hard on rate-limit errors.
# 0.1s gap is enough to avoid instant-throttle, 5x faster than the old 0.5s fixed delay.
FIXED_SLEEP_SEC = 0.1
INITIAL_BACKOFF_SEC = 10.0
MAX_BACKOFF_SEC = 120.0
BACKOFF_MULTIPLIER = 2.0
MAX_QUARTERS_DEFAULT = 12

ENDPOINT_SPECS = [
    (
        "income",
        {
            "fields": "ts_code,end_date,report_type,total_revenue,operate_profit,total_cogs,"
            "ebit,n_income_attr_p,n_income",
        },
    ),
    (
        "cashflow",
        {
            "fields": "ts_code,end_date,n_cashflow_act,net_cash_flows_oper_act",
        },
    ),
    (
        "fina_indicator",
        {
            "fields": "ts_code,end_date,roe,roa,debt_to_assets,current_ratio,quick_ratio,"
            "grossprofit_margin,netprofit_margin,eps,bps,or_yoy,op_yoy,profit_dedt",
        },
    ),
    (
        "balancesheet",
        {
            "fields": "ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,"
            "money_cap,accounts_receiv",
        },
    ),
]


def _read_tushare_token() -> Optional[str]:
    for env_name in ("TUSHARE_API_TOKEN", "TUSHARE_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    token_path = PROJECT_ROOT / "tokens.txt"
    try:
        return token_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _get_tushare_client():
    token = _read_tushare_token()
    if not token:
        logger.error("No Tushare token found. Set TUSHARE_TOKEN env var or populate tokens.txt")
        return None
    try:
        import tushare as ts
    except ImportError:
        logger.error("tushare not installed. pip install tushare")
        return None
    ts.set_token(token)
    return ts.pro_api()


def load_ts_codes(ts_codes_path: str) -> list:
    """Load ts_code list from a parquet or CSV file."""
    path = Path(ts_codes_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        logger.error("Unsupported format: %s (use .parquet or .csv)", path.suffix)
        sys.exit(1)

    for col in ("ts_code", "code", "symbol"):
        if col in df.columns:
            return sorted(df[col].dropna().unique().tolist())
    logger.error("No ts_code/code/symbol column found in %s", path)
    sys.exit(1)


def _rate_limited_call(fetcher, backoff_state: dict, ts_code: str, **kwargs):
    """Call a Tushare endpoint with adaptive backoff on rate-limit errors."""
    while True:
        if backoff_state["delay"] > 0:
            logger.info("Backing off %.1fs (rate limited)...", backoff_state["delay"])
            time.sleep(backoff_state["delay"])
        try:
            return fetcher(ts_code=ts_code, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            rate_hit = any(kw in msg for kw in (
                "频率", "rate", "limit", "throttle", "too many", "too frequent",
                "访问受限", "超过", "每分钟",
            ))
            if rate_hit:
                backoff_state["delay"] = max(
                    backoff_state["delay"] * BACKOFF_MULTIPLIER,
                    INITIAL_BACKOFF_SEC,
                )
                backoff_state["delay"] = min(backoff_state["delay"], MAX_BACKOFF_SEC)
                logger.warning("Rate limited — next backoff %.1fs", backoff_state["delay"])
                continue
            raise


def _decay_backoff(backoff_state: dict):
    """Gradually reduce backoff delay after successful calls."""
    if backoff_state["delay"] > 0:
        backoff_state["consecutive_ok"] = backoff_state.get("consecutive_ok", 0) + 1
        if backoff_state["consecutive_ok"] >= 50:
            backoff_state["delay"] = max(0, backoff_state["delay"] - INITIAL_BACKOFF_SEC)
            backoff_state["consecutive_ok"] = 0


def _last_completed_quarter() -> date:
    """Most recent quarter with published financial data (with 7-day grace period)."""
    import calendar
    today = date.today()
    # Quarter start month: 1, 4, 7, 10
    q_start = ((today.month - 1) // 3) * 3 + 1
    # Last completed quarter end month: 3, 6, 9, 12
    q_end = q_start - 1
    year = today.year
    if q_end == 0:
        q_end = 12
        year -= 1
    # 7-day grace period after quarter end — reports not yet published
    if today.month == q_start and today.day <= 7:
        q_end -= 3
        if q_end <= 0:
            q_end += 12
            year -= 1
    last_day = calendar.monthrange(year, q_end)[1]
    return date(year, q_end, last_day)


def fetch_stock_financials(pro, ts_code: str, max_quarters: int,
                           backoff_state: dict | None = None,
                           start_date: str | None = None) -> Optional[pd.DataFrame]:
    """Fetch financial data for a single stock across all endpoints."""
    if backoff_state is None:
        backoff_state = {"delay": 0.0}
    merged = None
    for idx, (endpoint_name, spec) in enumerate(ENDPOINT_SPECS):
        if idx > 0:
            time.sleep(FIXED_SLEEP_SEC)
        fetcher = getattr(pro, endpoint_name, None)
        if fetcher is None:
            continue
        kwargs = {"limit": max_quarters, "fields": spec["fields"]}
        if start_date:
            kwargs["start_date"] = start_date
        try:
            frame = _rate_limited_call(
                fetcher, backoff_state, ts_code=ts_code, **kwargs,
            )
            _decay_backoff(backoff_state)
        except Exception as exc:
            logger.debug("Tushare %s fetch failed for %s: %s", endpoint_name, ts_code, exc)
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.copy()

        if "end_date" not in frame.columns:
            continue
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
        frame = frame[frame["end_date"].notna()]
        if frame.empty:
            continue
        frame = frame.sort_values("end_date").drop_duplicates(subset=["end_date"], keep="last")

        keep_cols = ["ts_code", "end_date"] + [
            col for col in frame.columns if col not in {"ts_code", "end_date"}
        ]
        frame = frame[keep_cols]

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["ts_code", "end_date"], how="outer")

    if merged is None or merged.empty:
        return None
    merged = merged.sort_values("end_date").drop_duplicates(subset=["end_date"], keep="last")
    return merged.tail(max_quarters).reset_index(drop=True)


def sync(
    ts_codes: list,
    max_quarters: int = MAX_QUARTERS_DEFAULT,
    full: bool = False,
) -> pd.DataFrame:
    """Sync financial data for a list of ts_codes. Idempotent and incremental.

    On first run: fetches max_quarters for all stocks (full backfill).
    On subsequent runs: only fetches new quarters for stocks that are behind.
    Use --full to force a complete refresh.
    """
    pro = _get_tushare_client()
    if pro is None:
        sys.exit(1)

    # Load existing data and compute per-stock freshness
    existing = None
    existing_latest: dict[str, pd.Timestamp] = {}
    if not full and OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["end_date"] = pd.to_datetime(existing["end_date"], errors="coerce")
        existing = existing[existing["end_date"].notna()]
        if not existing.empty:
            existing_latest = (
                existing.groupby("ts_code")["end_date"].max().to_dict()
            )
            logger.info("Loaded %d existing rows (%d unique ts_codes)",
                        len(existing), len(existing_latest))

    current_q = _last_completed_quarter()
    logger.info("Last completed quarter: %s", current_q.strftime("%Y-%m-%d"))

    # Classify stocks: skip up-to-date, compute start_date for stale/new
    stale: dict[str, str | None] = {}  # ts_code → start_date (None = no filter)
    skipped = 0
    for ts_code in ts_codes:
        cur = existing_latest.get(ts_code)
        if cur is not None and cur >= pd.Timestamp(current_q):
            skipped += 1
            continue
        if cur is not None:
            # Overlap 1 quarter to catch restatements
            overlap = cur - pd.DateOffset(months=3)
            stale[ts_code] = overlap.strftime("%Y%m%d")
        else:
            stale[ts_code] = None  # New stock → no filter

    if skipped:
        logger.info("Skipping %d up-to-date stocks (already have data through %s)",
                    skipped, current_q.strftime("%Y-%m-%d"))
    logger.info("Fetching %d stocks that need updating", len(stale))
    if not stale:
        logger.info("Nothing to fetch.")
        return existing if existing is not None else pd.DataFrame()

    CHECKPOINT_INTERVAL = 500
    backoff_state = {"delay": 0.0}
    all_frames = [existing] if existing is not None else []
    total = len(stale)
    success = 0
    failed = 0

    def _save_checkpoint(frames, label=""):
        if not frames:
            return
        clean_frames = [f.reset_index(drop=True).dropna(axis=1, how="all") for f in frames]
        clean_frames = [f for f in clean_frames if not f.empty]
        if not clean_frames:
            return
        combined = pd.concat(clean_frames, ignore_index=True)
        combined["end_date"] = pd.to_datetime(combined["end_date"], errors="coerce")
        combined = combined[combined["end_date"].notna()]
        combined = combined.sort_values(["ts_code", "end_date"]).drop_duplicates(
            subset=["ts_code", "end_date"], keep="last"
        )
        combined = combined.reset_index(drop=True)
        if combined.columns.duplicated().any():
            combined = combined.loc[:, ~combined.columns.duplicated()]
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        logger.info(
            "Checkpoint %s: %d rows (%d unique ts_codes) → %s",
            label, len(combined), combined["ts_code"].nunique(), OUTPUT_PATH,
        )

    for i, (ts_code, start_date) in enumerate(stale.items()):
        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (success=%d, failed=%d)", i + 1, total, success, failed)

        frame = fetch_stock_financials(pro, ts_code, max_quarters,
                                       backoff_state, start_date=start_date)
        if frame is not None and not frame.empty:
            all_frames.append(frame.reset_index(drop=True))
            success += 1
        else:
            failed += 1

        if i < total - 1:
            time.sleep(FIXED_SLEEP_SEC)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(all_frames, label=f"{i + 1}/{total}")

    logger.info("Done: %d/%d stocks returned data (%d failed)", success, total, failed)

    if not all_frames:
        logger.warning("No data fetched.")
        return pd.DataFrame()

    _save_checkpoint(all_frames, label="final")
    combined = pd.read_parquet(OUTPUT_PATH)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Sync financial data from Tushare to local parquet")
    parser.add_argument(
        "--ts-codes", required=True,
        help="Path to parquet/CSV with ts_code column (e.g. data/stock_basic/stock_basic.parquet)",
    )
    parser.add_argument(
        "--quarters", type=int, default=MAX_QUARTERS_DEFAULT,
        help=f"Number of quarters to fetch (default: {MAX_QUARTERS_DEFAULT})",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full refresh: discard existing data and re-fetch everything",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N stocks (for testing, 0 = all)",
    )
    args = parser.parse_args()

    ts_codes = load_ts_codes(args.ts_codes)
    logger.info("Loaded %d ts_codes from %s", len(ts_codes), args.ts_codes)

    if args.limit > 0:
        ts_codes = ts_codes[: args.limit]
        logger.info("Limited to %d stocks", len(ts_codes))

    sync(ts_codes, max_quarters=args.quarters, full=args.full)


if __name__ == "__main__":
    main()
