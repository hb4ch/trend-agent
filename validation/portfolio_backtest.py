from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from validation.label_builder import (
    DEFAULT_LABEL_PATH,
    infer_limit_pct,
    is_limit_down_open,
    is_limit_up_open,
    load_price_history,
)
from validation.metrics import select_top_n
from validation.signal_store import read_table


@dataclass(frozen=True)
class BacktestConfig:
    labels_path: Path = DEFAULT_LABEL_PATH
    price_root: Path = Path("data/stock_ticks")
    score_col: str = "alpha_rank_score"
    top_n: int = 10
    frequency: str = "W-FRI"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    require_tradable_buy: bool = True
    exclude_limit_up_buys: bool = True
    delay_limit_down_sells: bool = True
    min_turnover_rate: float | None = None
    week_ending: str | pd.Timestamp | None = None


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    stats: dict[str, float]


def run_backtest(config: BacktestConfig, labels: pd.DataFrame | None = None) -> BacktestResult:
    labels = read_table(config.labels_path) if labels is None else labels.copy()
    if labels.empty:
        raise ValueError("Labels are required for backtest")
    if config.score_col not in labels.columns:
        raise ValueError(f"Missing score column: {config.score_col}")

    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    if "entry_date" in labels.columns:
        labels["entry_date"] = pd.to_datetime(labels["entry_date"]).dt.normalize()

    if config.week_ending is not None:
        target = pd.to_datetime(config.week_ending).normalize()
        anchor = labels["signal_date"].apply(_anchor_friday)
        labels = labels[anchor == target].copy()
        if labels.empty:
            raise ValueError(f"No signals found for week ending {target.date()}")

    price_cache: dict[str, pd.DataFrame] = {}
    rebalance_dates = _weekly_signal_dates(labels["signal_date"], config.frequency)
    current: dict[str, float] = {}
    nav = 1.0
    prior_trade_date: pd.Timestamp | None = None
    nav_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for anchor_friday in rebalance_dates:
        if config.frequency == "signal":
            universe = labels[labels["signal_date"] == anchor_friday].copy()
        else:
            anchor = labels["signal_date"].apply(_anchor_friday)
            week_labels = labels[anchor == anchor_friday]
            if week_labels.empty:
                continue
            best_date = _best_signal_date_in_week(week_labels)
            if best_date is None:
                nav_rows.append(
                    {
                        "signal_date": anchor_friday,
                        "trade_date": pd.NaT,
                        "nav": nav,
                        "period_return": np.nan,
                        "holding_count": len(current),
                        "turnover": np.nan,
                    }
                )
                continue
            universe = labels[labels["signal_date"] == best_date].copy()

        if config.min_turnover_rate is not None and "turnover_rate" in universe.columns:
            universe = universe[pd.to_numeric(universe["turnover_rate"], errors="coerce") >= config.min_turnover_rate]
        if config.require_tradable_buy and "tradable_buy" in universe.columns:
            universe = universe[universe["tradable_buy"].fillna(False).astype(bool)]
        if config.exclude_limit_up_buys and "limit_up_at_entry" in universe.columns:
            universe = universe[~universe["limit_up_at_entry"].fillna(False).astype(bool)]

        if universe.empty:
            nav_rows.append(
                {
                    "signal_date": anchor_friday,
                    "trade_date": pd.NaT,
                    "nav": nav,
                    "period_return": np.nan,
                    "holding_count": len(current),
                    "turnover": np.nan,
                }
            )
            continue

        desired = select_top_n(universe, config.score_col, config.top_n)
        trade_date = _trade_date_for_rebalance(desired, labels, anchor_friday)
        if trade_date is None:
            continue

        if prior_trade_date is not None and current:
            nav *= 1.0 + _portfolio_open_to_open_return(current, prior_trade_date, trade_date, config.price_root, price_cache)

        desired_codes = set(desired["ts_code"].astype(str))
        next_codes: set[str] = set()
        delayed_sells: list[str] = []

        for ts_code in sorted(current):
            if ts_code in desired_codes:
                next_codes.add(ts_code)
                continue
            row = _price_row_at_or_after(ts_code, trade_date, config.price_root, price_cache)
            prev_close = _previous_close(ts_code, row, config.price_root, price_cache)
            if config.delay_limit_down_sells and row is not None and is_limit_down_open(ts_code, row, prev_close):
                next_codes.add(ts_code)
                delayed_sells.append(ts_code)
                trade_rows.append(_trade_record(anchor_friday, trade_date, ts_code, "sell_delayed", row, "limit_down"))
            else:
                trade_rows.append(_trade_record(anchor_friday, trade_date, ts_code, "sell", row, "rebalance"))

        skipped_buys: list[str] = []
        for _, row in desired.iterrows():
            ts_code = str(row["ts_code"])
            if ts_code in next_codes:
                continue
            price_row = _price_row_at_or_after(ts_code, pd.to_datetime(row.get("entry_date", trade_date)), config.price_root, price_cache)
            prev_close = _previous_close(ts_code, price_row, config.price_root, price_cache)
            if config.exclude_limit_up_buys and price_row is not None and is_limit_up_open(ts_code, price_row, prev_close):
                skipped_buys.append(ts_code)
                trade_rows.append(_trade_record(anchor_friday, trade_date, ts_code, "buy_skipped", price_row, "limit_up"))
                continue
            next_codes.add(ts_code)
            trade_rows.append(_trade_record(anchor_friday, trade_date, ts_code, "buy", price_row, "rebalance"))

        old_weights = current
        new_weight = 1.0 / len(next_codes) if next_codes else 0.0
        current = {code: new_weight for code in sorted(next_codes)}
        turnover = _turnover(old_weights, current)
        nav *= 1.0 - turnover * (config.transaction_cost_bps + config.slippage_bps) / 10000.0

        holding_rows.append(
            {
                "signal_date": anchor_friday,
                "trade_date": trade_date,
                "holdings": json.dumps(sorted(current), ensure_ascii=False),
                "holding_count": len(current),
                "delayed_sells": json.dumps(delayed_sells, ensure_ascii=False),
                "skipped_buys": json.dumps(skipped_buys, ensure_ascii=False),
                "turnover": turnover,
            }
        )
        nav_rows.append(
            {
                "signal_date": anchor_friday,
                "trade_date": trade_date,
                "nav": nav,
                "period_return": np.nan if len(nav_rows) == 0 else nav / nav_rows[-1]["nav"] - 1.0,
                "holding_count": len(current),
                "turnover": turnover,
            }
        )
        prior_trade_date = trade_date

    nav_df = pd.DataFrame(nav_rows)
    holdings_df = pd.DataFrame(holding_rows)
    trades_df = pd.DataFrame(trade_rows)
    return BacktestResult(nav=nav_df, holdings=holdings_df, trades=trades_df, stats=_backtest_stats(nav_df))


def _anchor_friday(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.Timedelta(days=(4 - d.day_of_week))


def _best_signal_date_in_week(week_labels: pd.DataFrame) -> pd.Timestamp | None:
    """Return the most recent signal_date in the week that has at least one tradable label."""
    if week_labels.empty:
        return None
    wl = week_labels.copy()
    if "tradable_buy" in wl.columns:
        wl["_valid"] = wl["tradable_buy"].fillna(False).astype(bool)
    else:
        wl["_valid"] = wl.get("label_status", pd.Series("ok", index=wl.index)) == "ok"
    valid = wl[wl["_valid"]]
    if valid.empty:
        return None
    return pd.to_datetime(valid["signal_date"].max()).normalize()


def _weekly_signal_dates(dates: pd.Series, frequency: str) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).dt.normalize().dropna().unique()).sort_values()
    if unique.empty:
        return []
    if frequency == "signal":
        return [pd.Timestamp(x) for x in unique]
    anchors = unique.apply(_anchor_friday).sort_values().unique()
    return [pd.Timestamp(x) for x in anchors]


def _trade_date_for_rebalance(desired: pd.DataFrame, labels: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    if not desired.empty and "entry_date" in desired.columns:
        valid = pd.to_datetime(desired["entry_date"]).dropna()
        if not valid.empty:
            return valid.min().normalize()
    same_day = labels[labels["signal_date"] == signal_date]
    if "entry_date" in same_day.columns:
        valid = pd.to_datetime(same_day["entry_date"]).dropna()
        if not valid.empty:
            return valid.min().normalize()
    return None


def _price_row_at_or_after(
    ts_code: str,
    date: pd.Timestamp,
    price_root: Path,
    cache: dict[str, pd.DataFrame],
) -> pd.Series | None:
    if ts_code not in cache:
        cache[ts_code] = load_price_history(ts_code, price_root)
    prices = cache[ts_code]
    if prices.empty:
        return None
    candidates = prices[prices["trade_date"] >= pd.to_datetime(date).normalize()]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _previous_close(ts_code: str, row: pd.Series | None, price_root: Path, cache: dict[str, pd.DataFrame]) -> float | None:
    if row is None:
        return None
    if "pre_close" in row and pd.notna(row["pre_close"]):
        return float(row["pre_close"])
    if ts_code not in cache:
        cache[ts_code] = load_price_history(ts_code, price_root)
    prices = cache[ts_code]
    idx = prices.index[prices["trade_date"] == row["trade_date"]]
    if len(idx) == 0 or int(idx[0]) == 0:
        return None
    return float(prices.loc[int(idx[0]) - 1, "close"])


def _portfolio_open_to_open_return(
    weights: dict[str, float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_root: Path,
    cache: dict[str, pd.DataFrame],
) -> float:
    returns = []
    for ts_code, weight in weights.items():
        start_row = _price_row_at_or_after(ts_code, start, price_root, cache)
        end_row = _price_row_at_or_after(ts_code, end, price_root, cache)
        if start_row is None or end_row is None:
            continue
        start_open = float(start_row.get("open", np.nan))
        end_open = float(end_row.get("open", np.nan))
        if np.isfinite(start_open) and start_open > 0 and np.isfinite(end_open):
            returns.append(weight * (end_open / start_open - 1.0))
    return float(np.nansum(returns)) if returns else 0.0


def _turnover(old: dict[str, float], new: dict[str, float]) -> float:
    codes = set(old) | set(new)
    return float(sum(abs(new.get(code, 0.0) - old.get(code, 0.0)) for code in codes))


def _trade_record(
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    ts_code: str,
    action: str,
    row: pd.Series | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "trade_date": trade_date,
        "ts_code": ts_code,
        "action": action,
        "reason": reason,
        "open": float(row.get("open", np.nan)) if row is not None else np.nan,
        "limit_pct": infer_limit_pct(ts_code, row) if row is not None else np.nan,
    }


def _backtest_stats(nav: pd.DataFrame, periods_per_year: int = 52) -> dict[str, float]:
    if nav.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan, "periods": 0}
    values = pd.to_numeric(nav["nav"], errors="coerce")
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    periods = int(values.notna().sum())
    total_return = float(values.iloc[-1] - 1.0)
    annualized = float(values.iloc[-1] ** (periods_per_year / max(periods, 1)) - 1.0) if values.iloc[-1] > 0 else np.nan
    return {
        "total_return": total_return,
        "annualized_return_proxy": annualized,
        "max_drawdown": float(drawdown.min()),
        "periods": periods,
        "mean_period_return": float(pd.to_numeric(nav.get("period_return"), errors="coerce").mean()),
    }
