from __future__ import annotations

import argparse
import sys
from pathlib import Path

from validation.factor_eval import evaluate_from_labels_path
from validation.label_builder import LabelConfig, build_forward_labels
from playground.common.portfolio_backtest import BacktestConfig, _weekly_signal_dates, run_backtest
from validation.report import write_backtest_report, write_factor_report
from validation.signal_store import SignalConfig, load_candidates, read_table, snapshot_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trend Agent validation loop")
    sub = parser.add_subparsers(dest="command", required=True)

    p_snapshot = sub.add_parser("snapshot", help="Persist candidate signals")
    p_snapshot.add_argument("--input", required=True, type=Path)
    p_snapshot.add_argument("--signal-date", required=True)
    p_snapshot.add_argument("--run-id")
    p_snapshot.add_argument("--agent-version", default="unknown")
    p_snapshot.add_argument("--config-hash")
    p_snapshot.add_argument("--output", type=Path, default=Path("data/signals/signal_snapshots.parquet"))

    p_labels = sub.add_parser("build-labels", help="Build forward return labels")
    p_labels.add_argument("--snapshots", type=Path, default=Path("data/signals/signal_snapshots.parquet"))
    p_labels.add_argument("--prices", type=Path, default=Path("data/stock_ticks"))
    p_labels.add_argument("--output", type=Path, default=Path("data/signals/signal_labels.parquet"))
    p_labels.add_argument("--horizons", default="1,3,5,10,20,40")

    p_eval = sub.add_parser("eval-factors", help="Evaluate factor predictive power")
    p_eval.add_argument("--labels", type=Path, default=Path("data/signals/signal_labels.parquet"))
    p_eval.add_argument("--factor", action="append")
    p_eval.add_argument("--horizons", default="1,3,5,10,20,40")
    p_eval.add_argument("--quantiles", type=int, default=5)
    p_eval.add_argument("--top-n", type=int, default=10)

    p_bt = sub.add_parser("backtest", help="Run weekly top-N equal-weight backtest")
    p_bt.add_argument("--labels", type=Path, default=Path("data/signals/signal_labels.parquet"))
    p_bt.add_argument("--prices", type=Path, default=Path("data/stock_ticks"))
    p_bt.add_argument("--score-col", default="alpha_rank_score")
    p_bt.add_argument("--top-n", type=int, default=10)
    p_bt.add_argument("--cost-bps", type=float, default=10.0)
    p_bt.add_argument("--slippage-bps", type=float, default=5.0)
    p_bt.add_argument("--frequency", default="W-FRI")
    p_bt.add_argument("--week-ending", default=None)

    p_report = sub.add_parser("report", help="Generate validation reports")
    p_report.add_argument("--kind", choices=["factor", "backtest", "all"], default="all")
    p_report.add_argument("--labels", type=Path, default=Path("data/signals/signal_labels.parquet"))
    p_report.add_argument("--prices", type=Path, default=Path("data/stock_ticks"))
    p_report.add_argument("--factor", action="append")
    p_report.add_argument("--score-col", default="alpha_rank_score")
    p_report.add_argument("--top-n", type=int, default=10)
    p_report.add_argument("--cost-bps", type=float, default=10.0)
    p_report.add_argument("--slippage-bps", type=float, default=5.0)
    p_report.add_argument("--output-dir", type=Path, default=Path("data/validation_reports"))
    p_report.add_argument("--week-ending", default=None)

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        candidates = load_candidates(args.input)
        config = SignalConfig(snapshot_path=args.output, signal_date=args.signal_date)
        out = snapshot_candidates(
            candidates,
            config=config,
            run_id=args.run_id or args.input.stem,
            agent_version=args.agent_version,
            config_hash=args.config_hash,
        )
        print(f"Wrote {len(out)} snapshot rows to {args.output}")
        return 0

    if args.command == "build-labels":
        config = LabelConfig(
            snapshots_path=args.snapshots,
            labels_path=args.output,
            price_root=args.prices,
            horizons=_parse_ints(args.horizons),
        )
        labels = build_forward_labels(config)
        print(f"Wrote {len(labels)} label rows to {args.output}")
        return 0

    if args.command == "eval-factors":
        result = evaluate_from_labels_path(
            labels_path=args.labels,
            factors=_factor_args(args.factor),
            horizons=_parse_ints(args.horizons),
            quantiles=args.quantiles,
            top_n=args.top_n,
        )
        print(result.ic_summary.to_string(index=False) if not result.ic_summary.empty else "No factor rows")
        return 0

    if args.command == "backtest":
        result = run_backtest(
            BacktestConfig(
                labels_path=args.labels,
                price_root=args.prices,
                score_col=args.score_col,
                top_n=args.top_n,
                transaction_cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                frequency=args.frequency,
                week_ending=args.week_ending,
            )
        )
        labels_df = read_table(args.labels)
        if not labels_df.empty:
            dates = _weekly_signal_dates(labels_df["signal_date"], args.frequency)
            if dates:
                print(f"Rebalance dates ({len(dates)} weeks): {', '.join(d.strftime('%Y-%m-%d') for d in dates)}")
        print(result.nav.tail(10).to_string(index=False) if not result.nav.empty else "No backtest rows")
        print(result.stats)
        return 0

    if args.command == "report":
        if args.kind in {"factor", "all"}:
            factor_result = evaluate_from_labels_path(labels_path=args.labels, factors=_factor_args(args.factor))
            html, md = write_factor_report(factor_result, output_dir=args.output_dir)
            print(f"Wrote factor report: {html} ({md})")
        if args.kind in {"backtest", "all"}:
            bt_config = BacktestConfig(
                labels_path=args.labels,
                price_root=args.prices,
                score_col=args.score_col,
                top_n=args.top_n,
                transaction_cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                week_ending=args.week_ending,
            )
            html, md = write_backtest_report(run_backtest(bt_config), config=bt_config, output_dir=args.output_dir)
            print(f"Wrote backtest report: {html} ({md})")
        return 0

    return 1


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _factor_args(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else ("alpha_rank_score",)


if __name__ == "__main__":
    sys.exit(main())
