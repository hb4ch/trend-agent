#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create the venv before running weekly_backtest.sh." >&2
  exit 1
fi

score_col="${SCORE_COL:-alpha_rank_score}"
top_n="${TOP_N:-10}"
cost_bps="${COST_BPS:-10}"
slippage_bps="${SLIPPAGE_BPS:-5}"

week_ending_arg=""
if [[ -n "${WEEK_ENDING:-}" ]]; then
  week_ending_arg="--week-ending ${WEEK_ENDING}"
  echo "[$(date -Is)] Filtering for week ending: ${WEEK_ENDING}"
fi

snapshots_file="data/signals/signal_snapshots.parquet"
if [[ ! -e "$snapshots_file" ]]; then
  echo "ERROR: Signal snapshots not found at ${snapshots_file}" >&2
  echo "  Run daily.sh first to generate signal snapshots." >&2
  exit 1
fi

echo "[$(date -Is)] Building validation labels"

".venv/bin/python" -m validation.cli build-labels \
  --snapshots "$snapshots_file" \
  --prices data/stock_ticks \
  --output data/signals/signal_labels.parquet

echo "[$(date -Is)] Signal dates in snapshot (with anchor Fridays):"
".venv/bin/python" -c "
import pandas as pd
labels = pd.read_parquet('data/signals/signal_labels.parquet')
dates = pd.to_datetime(labels['signal_date']).dt.normalize().sort_values().unique()
if len(dates):
    for d in dates:
        anchor = d + pd.Timedelta(days=(4 - d.day_of_week))
        print(f'  signal_date={d.date()}  anchor_friday={anchor.date()}')
else:
    print('  (none)')
"

labels_file="data/signals/signal_labels.parquet"
if [[ ! -e "$labels_file" ]]; then
  echo "ERROR: Labels file not found at ${labels_file}" >&2
  exit 1
fi

echo "[$(date -Is)] Evaluating factors: factor=${score_col}, top_n=${top_n}"

".venv/bin/python" -m validation.cli eval-factors \
  --labels "$labels_file" \
  --factor "$score_col" \
  --top-n "$top_n"

echo "[$(date -Is)] Running weekly backtest: score_col=${score_col}, top_n=${top_n}, cost_bps=${cost_bps}, slippage_bps=${slippage_bps}"

".venv/bin/python" -m validation.cli backtest \
  --labels "$labels_file" \
  --prices data/stock_ticks \
  --score-col "$score_col" \
  --top-n "$top_n" \
  --cost-bps "$cost_bps" \
  --slippage-bps "$slippage_bps" \
  $week_ending_arg

echo "[$(date -Is)] Writing validation reports"

".venv/bin/python" -m validation.cli report \
  --kind all \
  --labels "$labels_file" \
  --prices data/stock_ticks \
  --factor "$score_col" \
  --score-col "$score_col" \
  --top-n "$top_n" \
  --cost-bps "$cost_bps" \
  --slippage-bps "$slippage_bps" \
  $week_ending_arg

echo "[$(date -Is)] Weekly validation complete"
