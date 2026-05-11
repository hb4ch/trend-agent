from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from validation.label_builder import DEFAULT_LABEL_PATH
from validation.metrics import (
    ic_by_date,
    quantile_returns,
    robustness_breakdowns,
    summarize_ic,
    top_n_returns,
)
from validation.signal_store import read_table


@dataclass(frozen=True)
class FactorEvalResult:
    ic_by_date: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_returns: pd.DataFrame
    top_n_returns: pd.DataFrame
    robustness: pd.DataFrame


def evaluate_factors(
    labels: pd.DataFrame,
    factors: Iterable[str],
    horizons: Iterable[int] = (1, 3, 5, 10, 20, 40),
    quantiles: int = 5,
    top_n: int = 10,
) -> FactorEvalResult:
    if labels is None or labels.empty:
        raise ValueError("Labels are required for factor evaluation")
    frames_ic = []
    frames_q = []
    frames_top = []
    frames_robust = []
    return_cols = [f"ret_{int(h)}d" for h in horizons if f"ret_{int(h)}d" in labels.columns]
    for factor in factors:
        if factor not in labels.columns:
            continue
        for return_col in return_cols:
            frames_ic.append(ic_by_date(labels, factor, return_col))
            q = quantile_returns(labels, factor, return_col, quantiles=quantiles)
            if not q.empty:
                q["factor"] = factor
                q["return_col"] = return_col
                frames_q.append(q)
            frames_top.append(top_n_returns(labels, factor, return_col, top_n=top_n))
        frames_robust.append(robustness_breakdowns(labels, factor, return_cols))

    ic = pd.concat(frames_ic, ignore_index=True) if frames_ic else pd.DataFrame()
    return FactorEvalResult(
        ic_by_date=ic,
        ic_summary=summarize_ic(ic),
        quantile_returns=pd.concat(frames_q, ignore_index=True) if frames_q else pd.DataFrame(),
        top_n_returns=pd.concat(frames_top, ignore_index=True) if frames_top else pd.DataFrame(),
        robustness=pd.concat(frames_robust, ignore_index=True) if frames_robust else pd.DataFrame(),
    )


def evaluate_from_labels_path(
    labels_path: Path = DEFAULT_LABEL_PATH,
    factors: Iterable[str] = ("alpha_rank_score",),
    horizons: Iterable[int] = (1, 3, 5, 10, 20, 40),
    quantiles: int = 5,
    top_n: int = 10,
) -> FactorEvalResult:
    return evaluate_factors(read_table(labels_path), factors=factors, horizons=horizons, quantiles=quantiles, top_n=top_n)
