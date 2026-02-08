#!/usr/bin/env python3
"""
Offline evaluator for Trend Agent candidate quality.

Usage:
  python strategy_evaluator.py --candidates reports/candidates.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_ROOT = Path("data")


def load_price_series(ts_code: str) -> pd.DataFrame:
    path = DATA_ROOT / "stock_ticks" / f"{ts_code}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "trade_date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def forward_return(df: pd.DataFrame, horizon: int) -> float:
    if len(df) <= horizon:
        return np.nan
    entry = float(df["close"].iloc[-(horizon + 1)])
    exit_price = float(df["close"].iloc[-1])
    return (exit_price - entry) / max(entry, 1e-9)


def evaluate_candidates(candidates: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        ts_code = str(row["ts_code"])
        prices = load_price_series(ts_code)
        if prices.empty:
            continue
        rec = {"ts_code": ts_code}
        for h in horizons:
            rec[f"ret_{h}d"] = forward_return(prices, h)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_metrics(joined: pd.DataFrame, score_col: str, ret_col: str) -> Dict[str, float]:
    if joined.empty or score_col not in joined.columns or ret_col not in joined.columns:
        return {}
    valid = joined[[score_col, ret_col]].dropna()
    if valid.empty:
        return {}
    cutoff = valid[score_col].quantile(0.9)
    top = valid[valid[score_col] >= cutoff]
    bottom = valid[valid[score_col] < cutoff]
    hit_rate = float((top[ret_col] > 0).mean()) if not top.empty else np.nan
    top_median = float(top[ret_col].median()) if not top.empty else np.nan
    spread = float(top[ret_col].mean() - bottom[ret_col].mean()) if not bottom.empty else np.nan
    drawdown_proxy = float(top[ret_col].quantile(0.1)) if not top.empty else np.nan
    return {
        "top_decile_hit_rate": hit_rate,
        "top_decile_median_return": top_median,
        "top_decile_spread": spread,
        "max_drawdown_proxy_p10": drawdown_proxy,
    }


def run_ablation(joined: pd.DataFrame, base_score_col: str = "alpha_rank_score") -> Dict[str, Dict[str, float]]:
    scenarios = {"baseline": joined}
    component_cols = [
        "toplist_recency_score",
        "theme_strength_score",
        "audit_risk_score",
        "positive_finding_count",
        "source_quality_score",
        "catalyst_diversity",
    ]
    for col in component_cols:
        if col in joined.columns and base_score_col in joined.columns:
            tmp = joined.copy()
            if col in {"toplist_recency_score", "audit_risk_score"}:
                tmp[base_score_col] = tmp[base_score_col] + tmp[col].fillna(0) * 5.0
            else:
                tmp[base_score_col] = tmp[base_score_col] - tmp[col].fillna(0) * 5.0
            scenarios[f"drop_{col}"] = tmp

    out = {}
    for name, df in scenarios.items():
        out[name] = summarize_metrics(df, score_col=base_score_col, ret_col="ret_10d")
    return out


def promotion_gates(metrics: Dict[str, float]) -> Dict[str, bool]:
    if not metrics:
        return {"pass": False}
    gate = {
        "improve_top_decile_10d": metrics.get("top_decile_median_return", -1) > 0.0,
        "hit_rate_minimum": metrics.get("top_decile_hit_rate", 0.0) >= 0.52,
        "spread_positive": metrics.get("top_decile_spread", -1) > 0.0,
    }
    gate["pass"] = all(gate.values())
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend Agent offline evaluator")
    parser.add_argument("--candidates", required=True, help="CSV/Parquet file containing candidates")
    parser.add_argument("--score-col", default="alpha_rank_score", help="Score column for ranking metrics")
    args = parser.parse_args()

    path = Path(args.candidates)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    if path.suffix.lower() == ".parquet":
        candidates = pd.read_parquet(path)
    else:
        candidates = pd.read_csv(path)

    horizons = [5, 10, 20]
    perf = evaluate_candidates(candidates, horizons=horizons)
    joined = candidates.merge(perf, on="ts_code", how="left")

    metrics = {f"{h}d": summarize_metrics(joined, score_col=args.score_col, ret_col=f"ret_{h}d") for h in horizons}
    ablation = run_ablation(joined, base_score_col=args.score_col)
    gates = promotion_gates(metrics.get("10d", {}))

    report = {
        "metrics": metrics,
        "ablation_10d": ablation,
        "promotion_gates": gates,
        "rows": int(len(joined)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
