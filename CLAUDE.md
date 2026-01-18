# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Trend Agent** is an A-share (Chinese stock market) investment research and stock screening system following a "momentum-focused, filter-based, timing-oriented" approach. The project implements a complete pipeline for stock analysis, research, and reporting.

### Core Philosophy

**"重势、通过滤、待时机"** (Emphasize momentum, apply strict filtering, wait for timing)

1. **定方向** (Define Direction): Web search to extract current market themes (3-5 themes)
2. **选标的** (Select Targets): Local Parquet data mining for technical patterns
3. **做尽调** (Due Diligence): Audit-style research to verify themes and eliminate risks
4. **交付** (Deliver): K-line charts + Markdown/PDF research reports

### Architecture - 5-Phase Pipeline

| Phase | Name | Description |
|-------|------|-------------|
| Phase 1 | Market Intelligence | Extract market themes using AI + web search |
| Phase 2 | Quantitative Mining | Screen stocks using DuckDB + technical criteria |
| Phase 3 | Deep Research | AI-powered risk audit and due diligence |
| Phase 4 | Visualization | Generate candlestick charts with mplfinance |
| Phase 5 | Report Generation | Create comprehensive research reports |

---

## Project Structure

```
trend-agent/
├── trend_agent.py              # Main pipeline orchestrator (Phase 1-5)
├── screen_growth_stocks.py     # Stock screening logic & technical analysis
├── deep_researcher.py          # AI research engine (Zhipu search + DeepSeek/Qwen)
├── search_tool.py             # Simple search wrapper
├── check_setup.py             # Environment validation script
├── CLAUDE.md                   # This file - project documentation
├── README.md                   # User-facing documentation
├── .env                        # Environment configuration (gitignored)
├── .gitignore                  # Git ignore rules
├── data/ -> ../tushare-stock-select/data  # Linked data directory
├── charts/                     # Generated technical analysis charts
├── reports/                    # Research reports and audit traces
├── tests/                      # Unit tests
│   ├── test_zhipu_search.py   # Zhipu search tests
│   └── __init__.py
├── .cache/                     # Cache directory
└── .venv/                      # Python virtual environment
```

---

## Key Modules

### 1. `trend_agent.py` (Main Pipeline)

**Purpose**: Core orchestrator implementing all 5 phases

**Key Classes**:
- `ThemeItem`: Market themes with keywords, summary, sources
- `AuditResult`: Audit findings with verdict, rationale, sources

**Key Functions**:
- `phase1_market_intel(llm)`: Extract market themes from web search
- `phase2_quant_filter(themes)`: Technical screening with DuckDB
- `phase3_deep_audit(llm, candidates)`: Risk audit and due diligence
- `phase4_plot_charts(candidates)`: Generate K-line charts
- `phase5_report_with_deepseek(...)`: Generate AI-powered reports

**Utility Functions**:
- `run_search(query)`: Execute web search via Zhipu
- `run_duckdb_sql(sql, context)`: Execute SQL on stock data
- `run_python(code, context)`: Safe Python code execution
- `qwen_match_themes(themes, candidates)`: Semantic theme matching

---

### 2. `screen_growth_stocks.py` (Stock Screener)

**Purpose**: Technical analysis and stock screening

**Screening Parameters**:
```python
MARKET_CAP_MIN = 20e8   # 2 billion RMB
MARKET_CAP_MAX = 300e8  # 30 billion RMB
MIN_DATA_DAYS = 120     # Minimum trading days
CONSOLIDATION_DAYS = 120  # Consolidation observation period
VOLATILITY_THRESHOLD = 0.35  # 35% volatility threshold
```

**Consolidation Score (0-100)**:
- 40 points: 120-day volatility < 35%
- 25 points: MA20/MA60/MA120 convergence < 15%
- 5 points: MA spread stability (std < 3%)
- 20 points: Price in middle of consolidation range (30%-70%)
- 10 points: Recent volume boost > 1.2x

**Output Files**:
- `screening_results.json` - Full results (JSON)
- `screening_results_YYYYMMDD_HHMMSS.csv` - Full results (CSV)

---

### 3. `deep_researcher.py` (AI Research Engine)

**Purpose**: AI-powered research and information gathering

**Integrations**:
- **Zhipu AI**: Native SDK web search (`ZhipuAiClient`)
- **SiliconFlow**: DeepSeek-V3.2 and Qwen3-8B models
- **LangChain**: Tool integration for search

**Key Functions**:
- `zhipu_search(query)`: Web search with domain filtering
- `deepseek_chat(messages)`: DeepSeek model chat
- `qwen_chat(messages)`: Qwen model chat
- `deepseek_plan_queries(...)`: Query planning for deep research

**Search Tool**:
- Supports `site:domain.com` shortcut for domain filtering
- Configurable result count, recency filter, content size
- Retry logic with exponential backoff

---

## Data Storage

### Parquet Format

Efficient columnar storage in `data/` directory:

| Directory | Contents |
|-----------|----------|
| `stock_basic/` | Basic stock info (codes, names, industries) |
| `stock_company/` | Company details (introduction, business scope) |
| `stock_ticks/` | Historical price data per stock |
| `financial/` | Financial statements and metrics |

### Output Files

| Type | Location | Format |
|------|----------|--------|
| Charts | `charts/{ts_code}.png` | PNG |
| Reports | `reports/report_YYYYMMDD_HHMMSS.md` | Markdown |
| Reports | `reports/report_YYYYMMDD_HHMMSS.pdf` | PDF |
| Audit Traces | `reports/audit_trace_*.jsonl` | JSONL |
| Debug Traces | `reports/deepseek_trace_*.jsonl` | JSONL |

---

## Environment Configuration

### Required Environment Variables

```env
# Zhipu AI (Web Search)
ZHIPUAI_API_KEY=xxx
ZHIPU_SEARCH_COUNT=15
DEBUG_ZHIPU_SEARCH=0

# SiliconFlow (DeepSeek/Qwen)
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=sk-xxx
DEEPSEEK_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
QWEN_MODEL=Qwen/Qwen3-8B

# Debug Flags
DEBUG_DEEPSEEK=0
DEBUG_SILICONFLOW=0
USE_QWEN_THEME_MATCH=1

# Strategy Parameters
REGULATORY_MAX_AGE_DAYS=730
```

### System Dependencies

- **pandoc**: For Markdown to PDF conversion
- **xelatex**: For Chinese PDF rendering
- **Python**: 3.10+ recommended

### Python Dependencies

```bash
pip install pandas numpy duckdb mplfinance matplotlib \
            langchain-core langchain-community python-dotenv zai
```

---

## Data Schema and ETL Logic

### 1. Stock Basic Info (`data/stock_basic/stock_basic.parquet`)

**Source:** `pro.stock_basic()` API

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| ts_code | string | Stock code (exchange.suffix) | `000001.SZ` |
| symbol | string | Stock symbol | `000001` |
| name | string | Company name | `平安银行` |
| area | string | Geographic area | `深圳` |
| industry | string | Industry classification | `银行` |
| list_date | string | Listing date (YYYYMMDD) | `19910403` |
| market | string | Market segment | `主板` |
| exchange | string | Exchange code | `SZSE` |

---

### 2. Stock Company Info (`data/stock_company/stock_company.parquet`)

**Source:** `pro.stock_company()` API

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| ts_code | string | Stock code | `688322.SH` |
| chairman | string | Chairman name | `黄源浩` |
| manager | string | Manager name | `黄源浩` |
| secretary | string | Secretary name | `靳尚` |
| reg_capital | float | Registered capital | `40114.4240` |
| setup_date | string | Setup date (YYYYMMDD) | `20130118` |
| province | string | Province | `广东` |
| city | string | City | `深圳市` |
| introduction | string | Company introduction | Long text description |
| website | string | Company website | `www.orbbec.com.cn` |
| employees | float | Number of employees | `687.0` |
| main_business | string | Main business description | Text description |
| business_scope | string | Business scope | Text description |

---

### 3. Stock Ticks (`data/stock_ticks/{ts_code}.parquet`)

**Source:** Merged from `ts.pro_bar()` (price data) and `pro.daily_basic()` (fundamental data)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| ts_code | string | Stock code | `600036.SH` |
| trade_date | string | Trade date (YYYYMMDD) | `20260109` |
| open | float | Opening price | `41.54` |
| high | float | Highest price | `41.76` |
| low | float | Lowest price | `41.12` |
| close | float | Closing price | `41.30` |
| pre_close | float | Previous close | `41.58` |
| change | float | Price change | `-0.28` |
| pct_chg | float | Percentage change | `-0.67` |
| vol | float | Volume (shares) | `985470.74` |
| amount | float | Trading amount (RMB) | `4069939.125` |
| turnover_rate | float | Turnover rate (%) | `0.4777` |
| volume_ratio | float | Volume ratio | `NaN` |
| pe | float | P/E ratio | `7.0192` |
| pe_ttm | float | P/E ratio (TTM) | `6.9915` |
| pb | float | P/B ratio | `0.9557` |
| ps | float | P/S ratio | `3.0863` |
| ps_ttm | float | P/S ratio (TTM) | `3.0981` |
| dv_ratio | float | Dividend ratio (%) | `4.8426` |
| dv_ttm | float | Dividend ratio TTM (%) | `4.8426` |
| total_share | float | Total shares (万) | `2.52e+06` |
| float_share | float | Float shares (万) | `2.06e+06` |
| free_share | float | Free float shares (万) | `1.17e+06` |
| total_mv | float | Total market cap (元) | `1.04e+08` |
| circ_mv | float | Circulating market cap (元) | `8.52e+07` |

---

### 4. Financial Data (`data/financial/income/{ts_code}.parquet`)

**Source:** `pro.income()` API

| Column | Type | Description |
|--------|------|-------------|
| ts_code | string | Stock code |
| ann_date | string | Announcement date |
| f_ann_date | string | First announcement date |
| end_date | string | Report end date (YYYYMMDD) |
| report_type | string | Report type (1=合并报表) |
| basic_eps | float | Basic EPS per share |
| diluted_eps | float | Diluted EPS per share |
| total_revenue | float | Total revenue |
| revenue | float | Operating revenue |
| oper_cost | float | Operating cost |
| operate_profit | float | Operating profit |
| total_profit | float | Total profit |
| n_income | float | Net income |
| n_income_attr_p | float | Net income attributable to parent |

---

### 5. Enriched Stock Analysis (In-Memory)

**Source:** Computed during selection

| Field | Description |
|-------|-------------|
| ts_code | Stock code |
| current_price | Latest closing price |
| consolidation_score | Technical consolidation score (0-100) |
| ma_spread | MA convergence metric |
| volume_boost | Recent volume multiple |
| ma_trend | Trend (bullish/bearish/neutral) |
| matched_themes | AI-matched market themes |

---

## Audit & Risk Assessment

### Hard Fail Conditions (One-Vote Veto)

- 立案调查
- 重大诉讼
- 减持计划
- 终止上市/退市风险
- 近期行政处罚/纪律处分

### Regulatory Pattern Matching

**Severe** (within `REGULATORY_MAX_AGE_DAYS`):
```
行政处罚|处罚决定书|纪律处分|公开谴责|市场禁入
```

**Minor** (warning only):
```
监管函|问询函|关注函|责令改正|监管措施决定书
```

### Primary Source Domains

- `cninfo.com.cn` - 巨潮资讯网
- `sse.com.cn` - 上交所
- `szse.cn` - 深交所

---

## Testing

### Running Tests

```bash
python -m unittest tests/test_zhipu_search.py
```

### Test Coverage

- `tests/test_zhipu_search.py`: Zhipu search functionality with mocked responses

---

## Common Tasks

### Run the Full Pipeline

```bash
source .venv/bin/activate
python trend_agent.py
```

### Run Stock Screening Only

```bash
python screen_growth_stocks.py
```

### Check Environment Setup

```bash
python check_setup.py
```

### Enable Debug Logging

```bash
export DEBUG_DEEPSEEK=1
export DEBUG_ZHIPU_SEARCH=1
export DEBUG_SILICONFLOW=1
```

---

## AI/LLM Integration

### Model Usage

| Model | Provider | Purpose |
|-------|----------|---------|
| glm-4-flash | Zhipu AI | Theme extraction, audit analysis |
| DeepSeek-V3.2 | SiliconFlow | Report generation, query planning |
| Qwen3-8B | SiliconFlow | Theme matching, classification |

### Tool Calling

The report generation phase supports tool calling:
- `web_search`: Search for additional information
- `duckdb`: Execute SQL queries on data
- `python`: Execute Python code for analysis

---

## Screening Criteria (Four Dimensions)

### 1. Valuation & Market Cap (Safety Margin)
- Market cap: 2-30 billion RMB
- Exclude stocks at historical highs or 2x recent price

### 2. Themes & Sentiment (Market Sentiment)
- Match current hot market themes
- Prefer companies with "stories not yet fully realized"

### 3. Technical Patterns
- Long-term consolidation (3-6 months, volatility < 35%)
- MA convergence (MA20/MA60/MA120 intertwined or bullish)
- Volume surge signals at bottom

### 4. Fundamentals
- Authentic theme targets (technology, products, market share)
- Exclude ST, delisting risk, fraud suspects

---

## Common Market Themes Reference

| Category | Core Concepts |
|----------|---------------|
| AI Applications | AI营销, 端侧AI, 物理AI, AI大模型 |
| Humanoid Robots | 减速器, 电机, 执行器 |
| Brain-Computer Interface | 神经科技, 脑机设备 |
| Low-Altitude Economy | eVTOL, 无人机, 低空飞行 |
| Semiconductors | 存储芯片, 国产算力, 半导体设备 |
| M&A Reorgs | 产业整合, 国企改革 |
| Commercial Aerospace | 航天制造, 卫星应用 |
| Intelligent Driving | L3级自动驾驶, 车路协同 |
| Energy Storage | 固态电池, 储能系统 |

---

## Investment Guidelines (For Reference)

1. **Position Sizing**: Build in 3-4 batches, avoid chasing highs
2. **Portfolio Allocation**: Core (60%) + Satellite (40%)
3. **Profit/Loss Management**:
   - Take profit: 30% gain in batches
   - Stop loss: -15% strict cutoff
4. **Holding Period**: 3-6 months medium-term

### Risk Warnings

- Theme speculation risk: Some stocks are concept-heavy, verify actual revenue
- Industry cyclicality: Sensitive to macroeconomic conditions
- Technical uncertainty: Emerging tech commercialization is uncertain
- Small market cap: Higher volatility in 2-20B RMB range

---

## Font Configuration for Charts

The system uses Chinese fonts for chart rendering. Candidate fonts:
- `Source Han Sans CN` (思源黑体)
- `Noto Sans CJK SC`
- `WenQuanYi Zen Hei`
- `SimHei`

Font files are auto-loaded from `/usr/share/fonts/adobe-source-han-sans/` if available.

---

## Performance Notes

- **Data Format**: Parquet for efficient columnar storage and querying
- **Query Engine**: DuckDB for fast SQL on Parquet files
- **Caching**: Theme matching results cached in `.cache/`
- **Batch Processing**: LLM requests batched for efficiency

---

## Troubleshooting

### PDF Generation Fails

```bash
# Install system dependencies
# Arch Linux
sudo pacman -S pandoc texlive-xetex

# Debian/Ubuntu
sudo apt-get install pandoc texlive-xetex
```

### Zhipu Search Not Working

- Check `ZHIPUAI_API_KEY` is set
- Enable debug: `DEBUG_ZHIPU_SEARCH=1`
- Verify network connectivity to api.zhipuai.cn

### Data Files Missing

- Ensure `data/` symlink points to valid data directory
- Verify Parquet files exist in `data/stock_ticks/`

---

## Related Projects

Data source: `tushare-stock-select` - Stock data fetching and ETL system
