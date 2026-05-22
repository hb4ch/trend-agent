"""Validation loop for Trend Agent candidate signals."""

from validation.factor_eval import FactorEvalResult, evaluate_factors
from validation.label_builder import LabelConfig, build_forward_labels
from validation.metrics import calculate_ic, calculate_rank_ic, select_top_n
from playground.common.portfolio_backtest import BacktestConfig, BacktestResult, run_backtest
from validation.signal_store import SignalConfig, snapshot_candidates

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FactorEvalResult",
    "LabelConfig",
    "SignalConfig",
    "build_forward_labels",
    "calculate_ic",
    "calculate_rank_ic",
    "evaluate_factors",
    "run_backtest",
    "select_top_n",
    "snapshot_candidates",
]
