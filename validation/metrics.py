from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional at runtime
    stats = None


def calculate_ic(values: pd.Series, returns: pd.Series) -> float:
    paired = pd.concat([values, returns], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="pearson"))


def calculate_rank_ic(values: pd.Series, returns: pd.Series) -> float:
    paired = pd.concat([values, returns], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    if stats is not None:
        return float(stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1], nan_policy="omit").statistic)
    return float(paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank(), method="pearson"))


def ic_by_date(df: pd.DataFrame, factor_col: str, return_col: str) -> pd.DataFrame:
    rows = []
    for signal_date, group in df.groupby("signal_date", dropna=False):
        valid = group[[factor_col, return_col]].dropna()
        rows.append(
            {
                "signal_date": signal_date,
                "factor": factor_col,
                "return_col": return_col,
                "n": int(len(valid)),
                "ic": calculate_ic(valid[factor_col], valid[return_col]),
                "rank_ic": calculate_rank_ic(valid[factor_col], valid[return_col]),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in ic.groupby(["factor", "return_col"]):
        factor, return_col = keys
        rows.append(
            {
                "factor": factor,
                "return_col": return_col,
                "periods": int(group["ic"].notna().sum()),
                "mean_ic": float(group["ic"].mean()),
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "ic_ir": _information_ratio(group["ic"]),
                "rank_ic_ir": _information_ratio(group["rank_ic"]),
            }
        )
    return pd.DataFrame(rows)


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    quantiles: int = 5,
) -> pd.DataFrame:
    frames = []
    for signal_date, group in df.groupby("signal_date", dropna=False):
        valid = group[["ts_code", factor_col, return_col]].dropna().copy()
        if len(valid) < 2:
            continue
        q_count = min(quantiles, valid[factor_col].nunique(), len(valid))
        if q_count < 2:
            continue
        valid["quantile"] = pd.qcut(
            valid[factor_col].rank(method="first"),
            q=q_count,
            labels=False,
            duplicates="drop",
        ) + 1
        valid["signal_date"] = signal_date
        frames.append(valid)
    if not frames:
        return pd.DataFrame()
    assigned = pd.concat(frames, ignore_index=True)
    return (
        assigned.groupby(["signal_date", "quantile"], as_index=False)
        .agg(n=("ts_code", "count"), mean_return=(return_col, "mean"), median_return=(return_col, "median"), hit_rate=(return_col, lambda s: float((s > 0).mean())))
    )


def select_top_n(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.dropna(subset=[score_col])
        .sort_values([score_col, "ts_code"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def top_n_returns(df: pd.DataFrame, factor_col: str, return_col: str, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for signal_date, group in df.groupby("signal_date", dropna=False):
        selected = select_top_n(group, factor_col, top_n)
        valid_returns = selected[return_col].dropna()
        rows.append(
            {
                "signal_date": signal_date,
                "factor": factor_col,
                "return_col": return_col,
                "top_n": top_n,
                "selected": int(len(selected)),
                "mean_return": float(valid_returns.mean()) if not valid_returns.empty else np.nan,
                "median_return": float(valid_returns.median()) if not valid_returns.empty else np.nan,
                "hit_rate": float((valid_returns > 0).mean()) if not valid_returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def robustness_breakdowns(
    df: pd.DataFrame,
    factor_col: str,
    return_cols: Iterable[str],
    group_cols: Iterable[str] = ("industry", "list_type", "market", "exchange"),
) -> pd.DataFrame:
    rows = []
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        for return_col in return_cols:
            for value, group in df.groupby(group_col, dropna=False):
                valid = group[[factor_col, return_col]].dropna()
                rows.append(
                    {
                        "group_col": group_col,
                        "group_value": str(value),
                        "factor": factor_col,
                        "return_col": return_col,
                        "n": int(len(valid)),
                        "mean_return": float(valid[return_col].mean()) if not valid.empty else np.nan,
                        "ic": calculate_ic(valid[factor_col], valid[return_col]),
                        "rank_ic": calculate_rank_ic(valid[factor_col], valid[return_col]),
                    }
                )
    return pd.DataFrame(rows)


def _information_ratio(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) < 2:
        return np.nan
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else np.nan
