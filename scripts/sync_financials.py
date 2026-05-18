#!/usr/bin/env python3
"""
One-shot financial data sync: pull from Tushare → local parquet cache.

Usage:
    python scripts/sync_financials.py --ts-codes data/stock_basic/stock_basic.parquet --quarters 12
    python scripts/sync_financials.py --ts-codes data/stock_basic/stock_basic.parquet --upsert

Idempotent: --upsert merges new quarters into existing data, skipping duplicates.
Respects Tushare rate limits (0.5s between API calls).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "financial" / "financial_quarters.parquet"
TUSHARE_CALL_INTERVAL_SEC = 0.5
MAX_QUARTERS_DEFAULT = 12

ENDPOINT_SPECS = [
    (
        "income",
        {
            "fields": "ts_code,end_date,report_type,total_revenue,revenue,n_income_attr_p,"
            "n_income,netprofit,gross_margin,grossprofit_margin",
            "rename": {
                "total_revenue": "revenue",
                "n_income_attr_p": "net_income",
                "grossprofit_margin": "gross_margin",
            },
        },
    ),
    (
        "cashflow",
        {
            "fields": "ts_code,end_date,n_cashflow_act,net_cash_flows_oper_act",
            "rename": {"net_cash_flows_oper_act": "op_cashflow"},
        },
    ),
    (
        "fina_indicator",
        {
            "fields": "ts_code,end_date,roe,roa,debt_to_assets,current_ratio,quick_ratio",
            "rename": {},
        },
    ),
    (
        "balancesheet",
        {
            "fields": "ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int",
            "rename": {"total_hldr_eqy_exc_min_int": "total_hldr_eqy"},
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


def fetch_stock_financials(pro, ts_code: str, max_quarters: int) -> Optional[pd.DataFrame]:
    """Fetch financial data for a single stock across all endpoints."""
    merged = None
    for idx, (endpoint_name, spec) in enumerate(ENDPOINT_SPECS):
        if idx > 0:
            time.sleep(TUSHARE_CALL_INTERVAL_SEC)
        fetcher = getattr(pro, endpoint_name, None)
        if fetcher is None:
            continue
        try:
            frame = fetcher(ts_code=ts_code, limit=max_quarters, fields=spec["fields"])
        except Exception as exc:
            logger.debug("Tushare %s fetch failed for %s: %s", endpoint_name, ts_code, exc)
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.copy()

        # Rename columns for unified schema
        frame.rename(columns=spec.get("rename", {}), inplace=True)

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
    upsert: bool = False,
) -> pd.DataFrame:
    """Sync financial data for a list of ts_codes. Returns combined DataFrame."""
    pro = _get_tushare_client()
    if pro is None:
        sys.exit(1)

    existing = None
    if upsert and OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["end_date"] = pd.to_datetime(existing["end_date"], errors="coerce")
        logger.info("Loaded %d existing rows for upsert", len(existing))

    all_frames = [existing] if existing is not None else []
    total = len(ts_codes)
    success = 0
    failed = 0

    for i, ts_code in enumerate(ts_codes):
        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (success=%d, failed=%d)", i + 1, total, success, failed)

        frame = fetch_stock_financials(pro, ts_code, max_quarters)
        if frame is not None and not frame.empty:
            all_frames.append(frame)
            success += 1
        else:
            failed += 1

        # Extra rate-limit between stocks (in addition to inter-endpoint delay)
        if i < total - 1:
            time.sleep(TUSHARE_CALL_INTERVAL_SEC)

    logger.info("Done: %d/%d stocks returned data (%d failed)", success, total, failed)

    if not all_frames:
        logger.warning("No data fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined["end_date"] = pd.to_datetime(combined["end_date"], errors="coerce")
    combined = combined[combined["end_date"].notna()]

    # Deduplicate: keep last per (ts_code, end_date)
    combined = combined.sort_values(["ts_code", "end_date"]).drop_duplicates(
        subset=["ts_code", "end_date"], keep="last"
    )
    combined = combined.reset_index(drop=True)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.info(
        "Wrote %d rows (%d unique ts_codes) to %s",
        len(combined),
        combined["ts_code"].nunique(),
        OUTPUT_PATH,
    )
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
        "--upsert", action="store_true",
        help="Merge into existing parquet instead of overwriting",
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

    sync(ts_codes, max_quarters=args.quarters, upsert=args.upsert)


if __name__ == "__main__":
    main()
