from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from validation.factor_eval import FactorEvalResult, evaluate_from_labels_path
from validation.label_builder import DEFAULT_LABEL_PATH
from playground.common.portfolio_backtest import BacktestConfig, BacktestResult, run_backtest


REPORT_ROOT = Path("data/validation_reports")


def write_factor_report(
    result: FactorEvalResult | None = None,
    output_dir: Path = REPORT_ROOT,
    as_of: str | None = None,
) -> tuple[Path, Path]:
    result = result or evaluate_from_labels_path(DEFAULT_LABEL_PATH)
    as_of = as_of or datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"factor_report_{as_of}.html"
    md_path = output_dir / f"factor_report_{as_of}.md"
    sections = [
        ("IC Summary", result.ic_summary),
        ("IC By Date", result.ic_by_date),
        ("Quantile Returns", result.quantile_returns),
        ("Top-N Returns", result.top_n_returns),
        ("Robustness", result.robustness),
    ]
    html_path.write_text(_html_document("Factor Validation Report", sections), encoding="utf-8")
    md_path.write_text(_markdown_document("Factor Validation Report", sections), encoding="utf-8")
    return html_path, md_path


def write_backtest_report(
    result: BacktestResult | None = None,
    config: BacktestConfig | None = None,
    output_dir: Path = REPORT_ROOT,
    as_of: str | None = None,
) -> tuple[Path, Path]:
    config = config or BacktestConfig()
    result = result or run_backtest(config)
    as_of = as_of or datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"backtest_report_{as_of}.html"
    md_path = output_dir / f"backtest_report_{as_of}.md"
    stats = pd.DataFrame([result.stats])
    cfg = pd.DataFrame([asdict(config)])
    sections = [
        ("Configuration", cfg),
        ("Stats", stats),
        ("NAV", result.nav),
        ("Holdings", result.holdings),
        ("Trades", result.trades),
    ]
    html_path.write_text(_html_document("Top-N Backtest Report", sections), encoding="utf-8")
    md_path.write_text(_markdown_document("Top-N Backtest Report", sections), encoding="utf-8")
    return html_path, md_path


def _html_document(title: str, sections: list[tuple[str, pd.DataFrame]]) -> str:
    body = []
    for heading, df in sections:
        body.append(f"<h2>{escape(heading)}</h2>")
        body.append(_html_table(df))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17202a; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #d7dde5; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f6f9; position: sticky; top: 0; }}
    .meta {{ color: #5b6775; margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="meta">Generated {escape(datetime.now().isoformat(timespec="seconds"))}</div>
  {''.join(body)}
</body>
</html>
"""


def _html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No rows.</p>"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d")
    return view.head(500).to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.6g}")


def _markdown_document(title: str, sections: list[tuple[str, pd.DataFrame]]) -> str:
    chunks = [f"# {title}", f"Generated {datetime.now().isoformat(timespec='seconds')}"]
    for heading, df in sections:
        chunks.append(f"\n## {heading}")
        if df is None or df.empty:
            chunks.append("No rows.")
        else:
            chunks.append(_markdown_table(df.head(200)))
    return "\n\n".join(chunks) + "\n"


def _markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d")
    columns = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
