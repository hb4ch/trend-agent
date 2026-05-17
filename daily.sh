#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create the venv before running daily.sh." >&2
  exit 1
fi

run_started_at="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${SIGNAL_DATE:-}" ]]; then
    signal_date="$SIGNAL_DATE"
else
    day_of_week="$(date +%u)"
    if [[ "$day_of_week" -eq 6 ]]; then
        signal_date="$(date -d 'yesterday' +%Y%m%d)"
    elif [[ "$day_of_week" -eq 7 ]]; then
        signal_date="$(date -d '2 days ago' +%Y%m%d)"
    else
        signal_date="$(date +%Y%m%d)"
    fi
fi

echo "[$(date -Is)] Starting daily Trend Agent run for signal_date=${signal_date}"

".venv/bin/python" trend_agent.py

latest_candidates="$(ls -t reports/candidates_*.csv 2>/dev/null | head -1 || true)"
if [[ -z "$latest_candidates" ]]; then
  echo "No reports/candidates_*.csv file found after trend_agent.py run." >&2
  exit 1
fi

run_id="${RUN_ID:-$(basename "$latest_candidates" .csv)}"
agent_version="${AGENT_VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

echo "[$(date -Is)] Snapshotting ${latest_candidates} as run_id=${run_id}, agent_version=${agent_version}"

".venv/bin/python" -m validation.cli snapshot \
  --input "$latest_candidates" \
  --signal-date "$signal_date" \
  --run-id "$run_id" \
  --agent-version "$agent_version"

echo "[$(date -Is)] Daily run complete: ${run_started_at}"
