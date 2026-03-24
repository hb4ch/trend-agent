# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Start Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full pipeline (all 5 phases)
python trend_agent.py

# Run stock screening only (Phase 2)
python screen_growth_stocks.py

# Check environment setup
python check_setup.py

# Run tests
python -m unittest tests/test_zhipu_search.py

# Enable debug logging
export DEBUG_DEEPSEEK=1
export DEBUG_ZHIPU_SEARCH=1
export FORCE_LLM_LOGGING=1  # Print all LLM I/O to screen
```

---

## Project Overview

**Trend Agent** is an A-share (Chinese stock market) investment research and stock screening system following a "momentum-focused, filter-based, timing-oriented" approach.

### Core Philosophy: **"重势、通过滤、待时机"**

1. **Phase 1 - Market Intelligence**: Web search + Dragon Tiger List → market themes
2. **Phase 2 - Quantitative Mining**: DuckDB + technical criteria → candidate stocks (relaxed filters)
3. **Phase 3 - Deep Research**: Opportunity Discovery (first) + Adversarial Veto Audit (second)
4. **Phase 4 - Visualization**: Generate K-line charts with Plotly
5. **Phase 5 - Report Generation**: DeepSeek generates self-contained HTML reports plus debug Markdown

---

## Key Modules

| File | Purpose |
|------|---------|
| `trend_agent.py` | Main pipeline orchestrator (all 5 phases) |
| `llm_provider.py` | Unified LLM provider (DeepSeek, Zhipu, Qwen) with thinking mode |
| `screen_growth_stocks.py` | Stock screening logic & technical analysis |
| `deep_researcher.py` | AI research engine (Zhipu search + query planning) |
| `utils.py` | Shared utilities (Dragon Tiger List, JSON parsing, etc.) |

### `trend_agent.py` - Key Functions

- `phase1_market_intel(llm)`: Extract market themes from web search + Dragon Tiger List
- `phase2_quant_filter(themes)`: Technical screening with DuckDB (relaxed parameters)
- `phase3_deep_audit(llm, candidates, themes)`: Two-pass research:
  - **Pass 1 - Opportunity Discovery**: Find positive catalysts (contracts, tech, policy, expansion)
  - **Pass 2 - Adversarial Veto**: Due diligence to check for hard fails
- `phase4_plot_charts(candidates)`: Generate K-line charts
- `phase5_report_with_deepseek(...)`: Generate AI-powered reports with findings and catalysts

### `llm_provider.py` - LLM Abstraction

```python
from llm_provider import get_llm, invoke_llm_messages, invoke_deepseek_thinking

# Get langchain LLM instance
llm = get_llm(model="deepseek")  # or "zhipu", "qwen"

# Simple invoke with message dicts
response = invoke_llm_messages("deepseek", [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
])

# DeepSeek thinking mode (reasoning_content + content)
result = invoke_deepseek_thinking([
    {"role": "user", "content": "Complex question..."}
])
# result["reasoning_content"] = chain of thought
# result["content"] = final answer
```

### `screen_growth_stocks.py` - Screening Parameters (Relaxed)

```python
MARKET_CAP_MIN = 10e8       # 1 billion RMB (relaxed from 2B)
MARKET_CAP_MAX = 500e8      # 50 billion RMB (relaxed from 30B)
MIN_DATA_DAYS = 180         # Minimum trading days (relaxed from 250)
VOLATILITY_THRESHOLD = 0.50 # 50% max amplitude (relaxed from 35%)
```

**Composite Score Weights (Rebalanced):**
- Consolidation: 40% (from 60%)
- Momentum: 20% (new)
- Volume boost: 25%
- Turnover: 15%

### `deep_researcher.py` - Search Functions

- `zhipu_search(query)`: Web search with `site:domain.com` support
- `deepseek_plan_queries(...)`: Dynamic query planning for multi-pass research
- `generate_opportunity_queries(name, theme)`: Generate opportunity discovery queries (without site: restrictions)
- `extract_positive_findings(results, name, category)`: Extract positive findings with confidence scoring
- `deepseek_plan_opportunity_queries(...)`: AI-planned queries for filling evidence gaps

**Opportunity Query Categories:**
- `policy_driver`: Government support, subsidies, projects
- `tech_breakthrough`: R&D, patents, innovation
- `market_expansion`: New products, overseas, capacity
- `competitive_moat`: Market position, advantages
- `contract_evidence`: Orders, customers, agreements

---

## Data Architecture

### Input Data (Parquet in `data/`)

| Directory | Description |
|-----------|-------------|
| `stock_basic/` | Basic stock info (codes, names, industries) |
| `stock_company/` | Company details (introduction, business scope) |
| `stock_ticks/{ts_code}.parquet` | Historical OHLCV + fundamentals per stock |
| `top_list/YYYYMMDD.parquet` | Dragon Tiger List daily aggregates |
| `top_inst/YYYYMMDD.parquet` | Dragon Tiger List institutional details |

### Output Files

| Location | Description |
|----------|-------------|
| `charts/{ts_code}.png` | Optional chart PNG exports |
| `reports/report_*.html` | Self-contained HTML research reports |
| `reports/report_*.md` | Debug markdown reports |
| `reports/audit_trace_*.jsonl` | Audit process traces |

---

## Environment Configuration

### Required `.env` Variables

```env
# Zhipu AI (Web Search)
ZHIPUAI_API_KEY=xxx

# SiliconFlow (DeepSeek/Qwen)
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=sk-xxx
DEEPSEEK_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
QWEN_MODEL=Qwen/Qwen3-8B

# Debug Flags
DEBUG_DEEPSEEK=0
DEBUG_ZHIPU_SEARCH=0
FORCE_LLM_LOGGING=0  # Print all LLM I/O to screen
```

### System Dependencies

No Pandoc / XeLaTeX dependency is required for report generation.

---

## Data Schema Reference

### Key Parquet Schemas

**`stock_ticks/{ts_code}.parquet`** - Daily OHLCV + fundamentals:
- `ts_code`, `trade_date`, `open`, `high`, `low`, `close`, `vol`, `amount`
- `turnover_rate`, `pe`, `pb`, `total_mv`, `circ_mv`

**`top_list/YYYYMMDD.parquet`** - Dragon Tiger List (~84 records/day):
- `ts_code`, `name`, `close`, `pct_change`, `turnover_rate`
- `l_buy`, `l_sell`, `net_amount` (capital flow signal)
- `net_rate`, `amount_rate` (capital intensity signals)

**`stock_company/stock_company.parquet`** - Company info:
- `ts_code`, `name`, `industry`, `main_business`, `business_scope`, `introduction`

---

## Research & Opportunity Discovery

### Phase 3 Two-Pass Strategy

1. **Opportunity Discovery Pass (First)**: Find positive catalysts
   - Uses broad searches (no site: restrictions)
   - Extracts: contracts, customers, policy support, tech breakthroughs, expansion
   - Outputs: `PositiveFinding` and `GrowthCatalyst` objects

2. **Adversarial Veto Pass (Second)**: Due diligence
   - Uses official sources (site:cninfo.com.cn)
   - Checks for hard fails
   - Outputs: verdict (pass/warn/fail)

### New Data Structures

**PositiveFinding:**
- `category`: contract, customer, policy, technology, expansion
- `description`, `evidence`, `confidence` (0.0-1.0)
- `source_url`, `date`

**GrowthCatalyst:**
- `catalyst_type`: policy, tech_breakthrough, market_expansion, competitive_moat
- `timeframe`: near_term, medium_term, long_term
- `confidence`

**Enhanced AuditResult:**
- `positive_findings`: List of positive findings
- `growth_catalysts`: List of catalysts
- `confidence_score`: Overall confidence (0.0-1.0)
- `research_depth`: "standard" or "deep"
- `capital_signal_summary`: From Dragon Tiger List

---

## Audit & Risk Assessment

### Hard Fail Conditions (One-Vote Veto)

- 立案调查 (formal investigation)
- 重大诉讼 (major litigation)
- 减持计划 (shareholder reduction plan)
- 退市风险 (delisting risk)
- 近期行政处罚 (recent penalties, within `REGULATORY_MAX_AGE_DAYS=730`)

### Regulatory Pattern Matching

**Severe** (fail): `行政处罚|纪律处分|公开谴责|市场禁入`
**Minor** (warn): `监管函|问询函|关注函|责令改正`

### Source Domain Tiers (Expanded)

**Tier 1 - Primary (Official Disclosures):**
- `cninfo.com.cn` - 巨潮资讯网 (official announcements)
- `sse.com.cn` - 上交所
- `szse.cn` - 深交所

**Tier 2 - Secondary (Financial News & Analysis):**
- `eastmoney.com` - 东方财富
- `10jqka.com.cn` - 同花顺
- `cls.cn` - 财联社
- `yicai.com` - 第一财经
- `caixin.com` - 财新网
- `sina.com.cn` - 新浪财经
- `gelonghui.com` - 格隆汇
- `xueqiu.com` - 雪球

**Tier 3 - Policy Sources:**
- `gov.cn` - 政府门户
- `ndrc.gov.cn` - 发改委
- `miit.gov.cn` - 工信部
- `most.gov.cn` - 科技部

**Tier 4 - Company Background:**
- `tianyancha.com` - 天眼查
- `qichacha.com` - 企查查

---

## LLM Model Configuration

| Model | Provider | Purpose |
|-------|----------|---------|
| `glm-4-flash` | Zhipu AI | Web search |
| `DeepSeek-V3.2` | SiliconFlow | Report generation, query planning, thinking mode |
| `Qwen3-8B` | SiliconFlow | Theme matching, classification |

### DeepSeek Thinking Mode

DeepSeek V3.2 supports reasoning mode that returns both `reasoning_content` (chain of thought) and `content` (final answer). Configured via `llm_provider.py`:

```python
# Enable thinking mode for complex analysis
result = invoke_deepseek_thinking(messages, thinking_budget=4096)
```

### Tool Calling (Phase 5)

Report generation supports tool calling:
- `web_search`: Additional web searches
- `duckdb`: SQL queries on stock data
- `python`: Python code execution for analysis

---

## Troubleshooting

### HTML Report Looks Blank

Check that the generated `report_*.html` file is opened in a browser with JavaScript enabled.

### Zhipu Search Not Working

```bash
# Check API key and enable debug
export ZHIPUAI_API_KEY=xxx
export DEBUG_ZHIPU_SEARCH=1
```

### Data Files Missing

Ensure `data/` symlink points to valid data directory with Parquet files.

---

## Related Projects

Data source: `tushare-stock-select` - Stock data fetching and ETL system
