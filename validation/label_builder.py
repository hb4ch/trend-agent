from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validation.signal_store import DEFAULT_SIGNAL_PATH, read_table, write_parquet


DEFAULT_LABEL_PATH = Path("data/signals/signal_labels.parquet")


@dataclass(frozen=True)
class LabelConfig:
    snapshots_path: Path = DEFAULT_SIGNAL_PATH
    labels_path: Path = DEFAULT_LABEL_PATH
    price_root: Path = Path("data/stock_ticks")
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 40)
    entry_price_policy: str = "next_open"


def build_forward_labels(config: LabelConfig, snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    snapshots = read_table(config.snapshots_path) if snapshots is None else snapshots.copy()
    if snapshots.empty:
        raise ValueError("No signal snapshots available for label building")
    required = {"signal_date", "ts_code"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshots missing required columns: {sorted(missing)}")
    snapshots["signal_date"] = pd.to_datetime(snapshots["signal_date"]).dt.normalize()
    snapshots["ts_code"] = snapshots["ts_code"].astype(str)
    if "run_id" in snapshots.columns:
        snapshots["run_id"] = snapshots["run_id"].astype(str)
    else:
        snapshots["run_id"] = "unknown"

    rows: list[dict[str, object]] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for _, signal in snapshots.iterrows():
        ts_code = str(signal["ts_code"])
        if ts_code not in price_cache:
            price_cache[ts_code] = load_price_history(ts_code, config.price_root)
        rows.append(_label_signal_row(signal, price_cache[ts_code], config.horizons))

    labels = pd.DataFrame(rows)
    passthrough_cols = [
        col
        for col in snapshots.columns
        if col not in labels.columns and col != "snapshot_created_at"
    ]
    labels = labels.merge(
        snapshots[["run_id", "signal_date", "ts_code"] + passthrough_cols],
        on=["run_id", "signal_date", "ts_code"],
        how="left",
    )
    labels = labels.sort_values(["signal_date", "run_id", "ts_code"]).reset_index(drop=True)
    write_parquet(labels, config.labels_path)
    return labels


def load_price_history(ts_code: str, price_root: Path) -> pd.DataFrame:
    path = Path(price_root) / f"{ts_code}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path).copy()
    if "trade_date" not in df.columns:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    if "ts_code" not in df.columns:
        df["ts_code"] = ts_code
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "turnover_rate", "vol", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def infer_limit_pct(ts_code: str, row: pd.Series | None = None) -> float:
    exchange = str(row.get("exchange", "") if row is not None else "").upper()
    code = str(ts_code)
    if exchange in {"BSE", "BJSE"} or code.startswith(("8", "4", "920")):
        return 0.30
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def is_limit_up_open(ts_code: str, row: pd.Series, prev_close: float | None = None) -> bool:
    base = _limit_base(row, prev_close)
    if not np.isfinite(base) or base <= 0:
        return False
    open_price = float(row.get("open", np.nan))
    return bool(np.isfinite(open_price) and open_price >= base * (1.0 + infer_limit_pct(ts_code, row)) - 1e-6)


def is_limit_down_open(ts_code: str, row: pd.Series, prev_close: float | None = None) -> bool:
    base = _limit_base(row, prev_close)
    if not np.isfinite(base) or base <= 0:
        return False
    open_price = float(row.get("open", np.nan))
    return bool(np.isfinite(open_price) and open_price <= base * (1.0 - infer_limit_pct(ts_code, row)) + 1e-6)


def _limit_base(row: pd.Series, prev_close: float | None) -> float:
    if "pre_close" in row and pd.notna(row.get("pre_close")):
        return float(row["pre_close"])
    return float(prev_close) if prev_close is not None else np.nan


def _label_signal_row(signal: pd.Series, prices: pd.DataFrame, horizons: Iterable[int]) -> dict[str, object]:
    signal_date = pd.to_datetime(signal["signal_date"]).normalize()
    ts_code = str(signal["ts_code"])
    rec: dict[str, object] = {
        "run_id": str(signal["run_id"]),
        "signal_date": signal_date,
        "ts_code": ts_code,
        "entry_date": pd.NaT,
        "entry_open": np.nan,
        "label_status": "missing_price_history",
        "limit_up_at_entry": False,
        "limit_down_at_entry": False,
        "tradable_buy": False,
    }
    for h in horizons:
        rec[f"exit_date_{h}d"] = pd.NaT
        rec[f"exit_close_{h}d"] = np.nan
        rec[f"ret_{h}d"] = np.nan

    if prices.empty:
        return rec

    candidates = prices[prices["trade_date"] > signal_date]
    if candidates.empty:
        rec["label_status"] = "missing_entry"
        return rec

    entry_idx = int(candidates.index[0])
    entry = prices.loc[entry_idx]
    entry_open = float(entry.get("open", np.nan))
    prev_close = float(prices.loc[entry_idx - 1, "close"]) if entry_idx > 0 and "close" in prices.columns else np.nan
    rec.update(
        {
            "entry_date": entry["trade_date"],
            "entry_open": entry_open,
            "limit_up_at_entry": is_limit_up_open(ts_code, entry, prev_close),
            "limit_down_at_entry": is_limit_down_open(ts_code, entry, prev_close),
            "tradable_buy": bool(np.isfinite(entry_open) and entry_open > 0 and not is_limit_up_open(ts_code, entry, prev_close)),
            "label_status": "ok",
        }
    )

    if not np.isfinite(entry_open) or entry_open <= 0:
        rec["label_status"] = "bad_entry_price"
        return rec

    for h in horizons:
        exit_idx = entry_idx + int(h)
        if exit_idx >= len(prices):
            continue
        exit_row = prices.loc[exit_idx]
        exit_close = float(exit_row.get("close", np.nan))
        rec[f"exit_date_{h}d"] = exit_row["trade_date"]
        rec[f"exit_close_{h}d"] = exit_close
        rec[f"ret_{h}d"] = (exit_close / entry_open) - 1.0 if np.isfinite(exit_close) else np.nan
    return rec
