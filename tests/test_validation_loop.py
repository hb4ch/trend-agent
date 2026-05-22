import json

import pandas as pd
import pytest

from validation.label_builder import LabelConfig, build_forward_labels
from validation.metrics import calculate_ic, calculate_rank_ic, select_top_n
from playground.common.portfolio_backtest import BacktestConfig, run_backtest
from validation.signal_store import SignalConfig, snapshot_candidates


def _write_prices(root, ts_code, rows):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / f"{ts_code}.parquet", index=False)


def test_ic_and_rank_ic_calculation():
    factor = pd.Series([1.0, 2.0, 3.0, 4.0])
    returns = pd.Series([4.0, 3.0, 2.0, 1.0])

    assert calculate_ic(factor, returns) == pytest.approx(-1.0)
    assert calculate_rank_ic(factor, returns) == pytest.approx(-1.0)


def test_forward_return_alignment_uses_next_open(tmp_path):
    price_root = tmp_path / "prices"
    _write_prices(
        price_root,
        "000001.SZ",
        [
            {"trade_date": "2026-01-01", "open": 10.0, "close": 10.5, "pre_close": 9.8},
            {"trade_date": "2026-01-02", "open": 11.0, "close": 12.0, "pre_close": 10.5},
            {"trade_date": "2026-01-05", "open": 12.5, "close": 13.2, "pre_close": 12.0},
            {"trade_date": "2026-01-06", "open": 13.0, "close": 14.0, "pre_close": 13.2},
        ],
    )
    snapshots = pd.DataFrame(
        [{"run_id": "r1", "signal_date": "2026-01-01", "ts_code": "000001.SZ", "alpha_rank_score": 90.0}]
    )

    labels = build_forward_labels(
        LabelConfig(
            snapshots_path=tmp_path / "unused.parquet",
            labels_path=tmp_path / "labels.parquet",
            price_root=price_root,
            horizons=(1, 3),
        ),
        snapshots=snapshots,
    )

    row = labels.iloc[0]
    assert row["entry_date"] == pd.Timestamp("2026-01-02")
    assert row["entry_open"] == pytest.approx(11.0)
    assert row["exit_date_1d"] == pd.Timestamp("2026-01-05")
    assert row["ret_1d"] == pytest.approx(13.2 / 11.0 - 1.0)
    assert pd.isna(row["ret_3d"])


def test_top_n_selection_is_score_descending_then_code_ascending():
    df = pd.DataFrame(
        [
            {"ts_code": "000003.SZ", "score": 80.0},
            {"ts_code": "000002.SZ", "score": 90.0},
            {"ts_code": "000001.SZ", "score": 90.0},
        ]
    )

    selected = select_top_n(df, "score", 2)

    assert selected["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]


def test_limit_up_buy_exclusion(tmp_path):
    price_root = tmp_path / "prices"
    _write_prices(
        price_root,
        "000001.SZ",
        [
            {"trade_date": "2026-01-01", "open": 10.0, "close": 10.0, "pre_close": 10.0},
            {"trade_date": "2026-01-02", "open": 11.0, "close": 11.0, "pre_close": 10.0},
            {"trade_date": "2026-01-05", "open": 11.2, "close": 11.1, "pre_close": 11.0},
        ],
    )
    _write_prices(
        price_root,
        "000002.SZ",
        [
            {"trade_date": "2026-01-01", "open": 10.0, "close": 10.0, "pre_close": 10.0},
            {"trade_date": "2026-01-02", "open": 10.2, "close": 10.3, "pre_close": 10.0},
            {"trade_date": "2026-01-05", "open": 10.4, "close": 10.5, "pre_close": 10.3},
        ],
    )
    snapshots = pd.DataFrame(
        [
            {"run_id": "r1", "signal_date": "2026-01-01", "ts_code": "000001.SZ", "alpha_rank_score": 100.0},
            {"run_id": "r1", "signal_date": "2026-01-01", "ts_code": "000002.SZ", "alpha_rank_score": 90.0},
        ]
    )
    labels = build_forward_labels(
        LabelConfig(labels_path=tmp_path / "labels.parquet", price_root=price_root, horizons=(1,)),
        snapshots=snapshots,
    )

    result = run_backtest(
        BacktestConfig(
            price_root=price_root,
            score_col="alpha_rank_score",
            top_n=1,
            frequency="signal",
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
        labels=labels,
    )

    holdings = json.loads(result.holdings.iloc[-1]["holdings"])
    assert holdings == ["000002.SZ"]
    assert bool(labels.loc[labels["ts_code"] == "000001.SZ", "limit_up_at_entry"].iloc[0]) is True


def test_limit_down_sell_delay(tmp_path):
    price_root = tmp_path / "prices"
    _write_prices(
        price_root,
        "000001.SZ",
        [
            {"trade_date": "2026-01-01", "open": 10.0, "close": 10.0, "pre_close": 10.0},
            {"trade_date": "2026-01-02", "open": 10.1, "close": 10.2, "pre_close": 10.0},
            {"trade_date": "2026-01-08", "open": 10.0, "close": 10.0, "pre_close": 10.2},
            {"trade_date": "2026-01-09", "open": 9.0, "close": 9.0, "pre_close": 10.0},
        ],
    )
    _write_prices(
        price_root,
        "000002.SZ",
        [
            {"trade_date": "2026-01-01", "open": 20.0, "close": 20.0, "pre_close": 20.0},
            {"trade_date": "2026-01-02", "open": 20.1, "close": 20.0, "pre_close": 20.0},
            {"trade_date": "2026-01-08", "open": 20.2, "close": 20.3, "pre_close": 20.0},
            {"trade_date": "2026-01-09", "open": 20.4, "close": 20.5, "pre_close": 20.3},
        ],
    )
    labels = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "ts_code": "000001.SZ",
                "alpha_rank_score": 100.0,
                "tradable_buy": True,
                "limit_up_at_entry": False,
            },
            {
                "run_id": "r2",
                "signal_date": "2026-01-08",
                "entry_date": "2026-01-09",
                "ts_code": "000002.SZ",
                "alpha_rank_score": 100.0,
                "tradable_buy": True,
                "limit_up_at_entry": False,
            },
            {
                "run_id": "r2",
                "signal_date": "2026-01-08",
                "entry_date": "2026-01-09",
                "ts_code": "000001.SZ",
                "alpha_rank_score": 1.0,
                "tradable_buy": True,
                "limit_up_at_entry": False,
            },
        ]
    )

    result = run_backtest(
        BacktestConfig(
            price_root=price_root,
            score_col="alpha_rank_score",
            top_n=1,
            frequency="signal",
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
        labels=labels,
    )

    delayed = result.trades[result.trades["action"] == "sell_delayed"]
    assert delayed["ts_code"].tolist() == ["000001.SZ"]
    assert json.loads(result.holdings.iloc[-1]["holdings"]) == ["000001.SZ", "000002.SZ"]


def test_snapshot_append_idempotency_and_conflict_detection(tmp_path):
    path = tmp_path / "signal_snapshots.parquet"
    config = SignalConfig(snapshot_path=path, signal_date="20260101")
    df = pd.DataFrame([{"ts_code": "000001.SZ", "alpha_rank_score": 80.0, "matched_themes": ["AI"]}])

    first = snapshot_candidates(df, config, run_id="r1", agent_version="test", config_hash="cfg")
    second = snapshot_candidates(df, config, run_id="r1", agent_version="test", config_hash="cfg")

    assert len(first) == 1
    assert len(second) == 1

    changed = pd.DataFrame([{"ts_code": "000001.SZ", "alpha_rank_score": 81.0, "matched_themes": ["AI"]}])
    with pytest.raises(ValueError, match="Conflicting immutable snapshot"):
        snapshot_candidates(changed, config, run_id="r1", agent_version="test", config_hash="cfg")
