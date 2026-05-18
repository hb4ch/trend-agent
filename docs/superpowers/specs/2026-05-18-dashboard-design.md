# Contrarian Agent Dashboard — Design Spec

## Overview

A web dashboard for the Contrarian Agent (trend-agent) project — an HTTP-based SPA that surfaces technical analysis on multiple time horizons, macro market intelligence, and screening signals. The dashboard combines T+1 historical Parquet data with intraday yfinance quotes to provide near-live market visibility.

**Project directory**: sibling to trend-agent (`/home/robertpeng/contrarian-dashboard`)  
**Launched via**: `uv run` or `npm run dev`  
**Accessed from**: Chrome browser, localhost  

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend framework | FastAPI (Python) | Same Python ecosystem, shares data access patterns with pipeline |
| Frontend framework | React + TypeScript | Rich interactivity, professional polish |
| UI components | shadcn/ui | Clean, modern, dark-theme friendly, strong table/chart primitives |
| Charts | Recharts + lightweight-charts (TradingView) | Recharts for dashboard sparklines; lightweight-charts for K-line |
| Data query | DuckDB | Reads Parquet directly, no ETL needed, shared with pipeline |
| Live data | yfinance | Free, ~15min delay for A-shares, sufficient for non-intraday use |
| Live fallback | Sina / Eastmoney public endpoints | Gaps where yfinance coverage is thin |
| Styling | Tailwind CSS | Comes with shadcn/ui, rapid UI development |
| Dev server | Vite | Standard React tooling |
| LLM | Gemma 4 on DGX Spark (vLLM) | Dedicated local server, already configured in llm_provider.py. Batch prompts for low-latency narrative generation |

### Why not Streamlit/Dash

Streamlit was considered but rejected: it produces recognizable Streamlit-looking apps that are hard to make look professional. shadcn/ui + React gives full control over visual polish. The extra frontend build pipeline is justified by the dashboard's complexity (8-panel grid, K-line charts, filterable tables).

---

## Project Structure

```
contrarian-dashboard/
├── backend/
│   ├── main.py              # FastAPI app, CORS, lifespan
│   ├── routes/
│   │   ├── macro.py         # GET /api/macro/*
│   │   ├── stock.py         # GET /api/stock/{ts_code}
│   │   ├── themes.py        # GET /api/themes
│   │   ├── screener.py      # GET /api/screener, POST /api/screener/nl
│   │   └── llm.py           # POST /api/llm/commentary, POST /api/llm/briefing, etc.
│   ├── services/
│   │   ├── duckdb_service.py    # DuckDB connection pool, Parquet queries
│   │   ├── yfinance_service.py  # yfinance batch fetch + cache
│   │   ├── fallback_service.py  # Sina/Eastmoney scrape for missing tickers
│   │   ├── compute_service.py   # Factor aggregation, sentiment calc, breadth
│   │   ├── pipeline_reader.py   # Read latest pipeline outputs (themes, candidates, audits)
│   │   └── llm_service.py       # vLLM Gemma client, prompt batching, SSE streaming
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── MacroDashboard.tsx    # / — 8-panel grid
│   │   │   ├── ThemeExplorer.tsx     # /themes
│   │   │   ├── StockScreener.tsx     # /screener
│   │   │   └── StockDetail.tsx       # /stock/:code
│   │   ├── components/
│   │   │   ├── panels/               # Dashboard panel components
│   │   │   │   ├── MarketSnapshot.tsx
│   │   │   │   ├── SentimentGauge.tsx
│   │   │   │   ├── HotThemes.tsx
│   │   │   │   ├── CapitalFlow.tsx
│   │   │   │   ├── TopMovers.tsx
│   │   │   │   ├── MarketBreadth.tsx
│   │   │   │   ├── DragonTigerFeed.tsx
│   │   │   │   └── SignalAlerts.tsx
│   │   │   ├── charts/
│   │   │   │   ├── KLineChart.tsx     # lightweight-charts wrapper
│   │   │   │   ├── Sparkline.tsx      # Recharts mini chart
│   │   │   │   └── HorizonFactors.tsx # Factor comparison across timeframes
│   │   │   ├── layout/
│   │   │   │   ├── TopNav.tsx
│   │   │   │   └── DashboardGrid.tsx  # 8-panel grid layout engine
│   │   │   └── ui/                    # shadcn/ui generated components
│   │   ├── hooks/
│   │   │   ├── usePolling.ts          # Polling hook with configurable interval
│   │   │   └── useApiData.ts          # Fetch + loading state + error boundary
│   │   ├── lib/
│   │   │   └── api.ts                 # API client (fetch wrappers)
│   │   └── types/
│   │       └── index.ts               # TypeScript type definitions
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
└── README.md
```

The dashboard lives in a **separate project directory** (`contrarian-dashboard/`) alongside `trend-agent/`. It reads data from `trend-agent/data/` via symlink or configured path. No code shared between projects — they communicate through Parquet files and the API boundary.

---

## Data Architecture

### Data Sources

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   yfinance   │     │  Tushare Parquet  │     │  Sina/Eastmoney │
│  (intraday)  │     │     (T+1)         │     │    (fallback)   │
└──────┬───────┘     └────────┬─────────┘     └────────┬────────┘
       │                      │                        │
       ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │ yfinance pool│  │ DuckDB     │  │ Pipeline Reader │  │
│  │ (5min cache) │  │ (read-only)│  │ (latest outputs)│  │
│  └──────┬───────┘  └─────┬──────┘  └────────┬────────┘  │
│         └────────────────┼──────────────────┘            │
│                          ▼                               │
│                   Compute Layer                           │
│         (merge live+history, factor calc, agg)           │
└──────────────────────────┬───────────────────────────────┘
                           │ REST JSON
                           ▼
                  ┌────────────────┐
                  │  React SPA     │
                  │  (Vite + React)│
                  └────────────────┘
```

### Live + History Merge Strategy

For GET `/api/stock/{ts_code}`:
1. Query yfinance cache for latest intraday OHLCV
2. Query DuckDB for historical OHLCV from `stock_ticks/{ts_code}.parquet`
3. Merge: if yfinance has a bar for today, replace the latest row; otherwise use history as-is
4. Return unified OHLCV array with `source` field (`history` | `live`)

For indices (上证/深证/创业板/科创50):
- yfinance symbols: `000001.SS`, `399001.SZ`, `399006.SZ`, `000688.SS`
- History from pipeline outputs or computed from constituent stock_ticks

### Cache Strategy

- yfinance batch quotes cached in-memory (Python dict, 5-minute TTL)
- DuckDB results cached at API layer (5-minute TTL for aggregate queries, 1-minute for stock detail)
- Parquet files are read-only — no write contention with pipeline

---

## API Endpoints

All endpoints return JSON. Base URL: `http://localhost:8000/api`

### GET /api/macro/indices
Live index quotes with sparkline history.

```json
{
  "shanghai": {"price": 3250.1, "change_pct": -1.2, "sparkline": [3248, 3250, ...]},
  "shenzhen": {"price": 10850.3, "change_pct": -0.8, "sparkline": [...]},
  "chinext": {"price": 2150.0, "change_pct": 0.3, "sparkline": [...]},
  "star50": {"price": 980.5, "change_pct": 0.1, "sparkline": [...]}
}
```

### GET /api/macro/sentiment
Composite sentiment from Dragon Tiger List + market breadth.

```json
{
  "score": 62,
  "label": "NEUTRAL-BULLISH",
  "components": {"capital_flow": 0.7, "breadth": 0.5, "theme_strength": 0.65}
}
```

### GET /api/macro/themes
Hot theme rankings.

```json
[
  {"name": "半导体自主可控", "change_pct": 3.2, "net_flow": 111.09, "status": "加速"},
  ...
]
```

### GET /api/macro/capital-flow
Capital flow breakdown and history.

```json
{
  "northbound": 12.3,
  "institutional": 8.7,
  "retail_active": -3.2,
  "history": [/* last 20 trading days */]
}
```

### GET /api/macro/top-movers
Top 10 screening candidates by composite score.

```json
[
  {"ts_code": "688256.SH", "name": "寒武纪", "composite_score": 0.85, ...}
]
```

### GET /api/macro/breadth
Market breadth indicators from factor_panel.

```json
{
  "advance_decline_ratio": 0.85,
  "pct_above_ma20": 48.2,
  "pct_above_ma60": 52.1,
  "volume_breadth": 0.72
}
```

### GET /api/macro/dragon-tiger
Today's Dragon Tiger List entries.

```json
[
  {"ts_code": "000078.SZ", "name": "海王生物", "reason": "...", "net_amount": 1.2e8, ...}
]
```

### GET /api/macro/signal-alerts
Latest signal labels from the validation pipeline.

```json
[
  {"ts_code": "688256.SH", "label": "entry", "date": "2026-05-16", ...}
]
```

### GET /api/stock/{ts_code}
Single stock full data. Query params: `?horizon=20` (5/20/60/120, default 20).

```json
{
  "ts_code": "000001.SZ",
  "name": "平安银行",
  "live_price": 10.24,
  "live_change_pct": -1.2,
  "ohlcv": [/* merged live+history array for K-line */],
  "factors": {
    "5d": {"ret": -2.1, "rsi_14": 48, "adx": 32, "dist_ma20": -2.8, "atr_pct": 2.1, "bbw": 0.12},
    "20d": {"ret": 3.4, "rsi_14": 55, "adx": 30, "dist_ma20": 1.2, "atr_pct": 2.8, "bbw": 0.15},
    "60d": {"ret": 12.1, "rsi_14": 62, "adx": 28, "dist_ma60": 5.1, "atr_pct": 3.2, "bbw": 0.18},
    "120d": {"ret": 8.7, "rsi_14": 71, "adx": 25, "dist_ma60": 3.8, "atr_pct": 3.5, "bbw": 0.20}
  },
  "audit": {
    "verdict": "pass",
    "confidence": 0.78,
    "positive_findings": [...],
    "growth_catalysts": [...]
  },
  "dragon_tiger_presence": {
    "recent_appearances": 3,
    "net_flow": 1.2,
    "last_reason": "日涨幅偏离值达到7%..."
  }
}
```

### GET /api/themes
Theme rankings with drill-down data.

```json
{
  "themes": [
    {
      "name": "半导体自主可控",
      "change_pct": 3.2,
      "net_flow": 111.09,
      "appearances": 66,
      "capital_structure": {"northbound": 18, "institutional": 7, "retail": 75},
      "constituents": ["688256.SH", "001309.SZ", ...],
      "trend": "加速"
    }
  ]
}
```

### GET /api/screener
Candidate list with filtering. Query params: `?industry=电子&min_score=0.6&horizon=20`.

```json
{
  "candidates": [
    {
      "ts_code": "688256.SH",
      "name": "寒武纪",
      "composite_score": 0.85,
      "consolidation": 0.78,
      "momentum": 0.82,
      "volume_boost": 0.91,
      "turnover": 0.75,
      "audit_verdict": "pass",
      "confidence": 0.82
    }
  ]
}
```

### POST /api/screener/nl
Natural language screening. Translates user query to filter params via LLM.

Request: `{"query": "低估值小市值科技股，近期放量突破"}`
Response:
```json
{
  "filters": {"pe_max": 20, "total_mv_max": 100e8, "industry": ["电子", "计算机"], "vol_ratio_min": 1.5},
  "explanation": "筛选PE<20、总市值<100亿的电子/计算机行业股票，且量比>1.5",
  "candidates": [...]
}
```

---

## LLM Integration (Gemma 4 on DGX Spark)

The dashboard embeds local LLM intelligence at five touchpoints. All calls go through backend → vLLM Gemma endpoint. Prompts are batched where possible (vLLM continuous batching handles concurrent requests natively). Never call LLM from the frontend.

### POST /api/llm/briefing
Daily market briefing. Generated once per data refresh, cached.

Response:
```json
{
  "text": "市场情绪偏中性(62分)，半导体板块持续走强，北上资金净流入12.3亿...",
  "generated_at": "2026-05-18T09:35:00"
}
```

### POST /api/llm/commentary/{ts_code}
Per-stock analysis commentary. Cached per stock+date.

Request: `{"horizon": 20}`
Response:
```json
{
  "text": "该股短期RSI中性(48)，距MA20偏离-2.8%处于盘整格局。但60日动量强劲(+12.1%)，中期均线多头排列，ADX=32显示趋势强度良好。关注MA20支撑确认后的入场时机。",
  "generated_at": "2026-05-18T10:00:00"
}
```

### POST /api/llm/themes
Theme narratives (batched — all themes in one prompt). Cached per day.

Response:
```json
{
  "narratives": [
    {"theme": "半导体自主可控", "text": "资金持续流入，游资主导但趋势加速，关注外部制裁催化与国产替代订单落地。"},
    ...
  ],
  "generated_at": "2026-05-18T09:35:00"
}
```

### POST /api/llm/signals
Signal alert interpretations (batched). Cached per signal run.

Response:
```json
{
  "interpretations": [
    {"ts_code": "688256.SH", "text": "盘整突破+放量确认+半导体板块共振，短期动能充足"},
    ...
  ],
  "generated_at": "2026-05-18T09:35:00"
}
```

### Batching Strategy

vLLM's continuous batching means multiple prompts submitted concurrently are processed together. The backend exploits this:

- **At data refresh time** (e.g., 9:35 after T+1 data lands): fire briefing + themes + signals in one concurrent batch. All three complete in ~3s total, not 9s sequentially.
- **On stock page load**: commentary is a single-prompt call (~2s). If the user navigates between stocks quickly, concurrent requests batch naturally.
- **NL screener**: single-prompt call (~1.5s), not batched — it's user-initiated and needs immediate response.

### Fallback

If Gemma is unreachable, LLM-dependent panels show raw data without commentary, and a dismissible warning banner appears at the top: "Gemma LLM 不可用，AI 注释已暂停 — 检查 DGX Spark 服务状态". Panels continue functioning with raw data only.

---

## Frontend Design

### Navigation

```
┌──────────────────────────────────────────────────┐
│  📊 Contrarian Agent   大盘  主题  选股       Live ●│
└──────────────────────────────────────────────────┘
```

Top navbar, 4 routes. No sidebar (saves horizontal space for the dense grid).

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Macro Dashboard | 8-panel grid, default landing |
| `/themes` | Theme Explorer | Sector drill-down with constituents |
| `/screener` | Stock Screener | Filterable candidate table |
| `/stock/:code` | Stock Detail | K-line + horizon tabs + audit |

### Macro Dashboard Layout (8-Panel Grid)

```
┌──────────────────────────────────────────────┐
│  Market Briefing (LLM, cached)               │
│  市场情绪偏中性，半导体板块持续走强...          │
├──────────────────────┬──────────────────┐
│  Market Snapshot     │  Hot Themes      │
│  (indices + change%) │  (ranked sectors)│
├──────────────────────┼──────────────────┤
│  Sentiment Gauge     │  Capital Flow    │
│  (composite score)   │  (北/机/游)      │
├──────────────────────┼──────────────────┤
│  Top Movers          │  Signal Alerts   │
│  (highest scores)    │  (latest signals)│
├──────────────────────┴──────────────────┤
│  Market Breadth Chart (wide)            │
├──────────────────────┬──────────────────┤
│  Dragon Tiger Feed   │                  │
│  (today's entries)   │                  │
└──────────────────────┴──────────────────┘
```

The briefing banner spans the full width at top. It's generated once per data refresh and cached — not re-generated on every page load. Below it, the original 8-panel grid.

The grid uses CSS Grid with `grid-template-columns: 2fr 1fr` for the main row split, with the Breadth chart spanning full width. On smaller screens, panels stack vertically.

### Loading Strategy

Each panel fetches its own data independently:
- **Initial load**: All 8 panels show skeleton/spinner. No blocking waterfall.
- **Data arrival**: Panels render as their data arrives. Each panel has 3 states: `loading` → `data` → `error`.
- **Polling refresh**: 60-second interval for live-dependent panels (Market Snapshot, Dragon Tiger Feed). 5-minute interval for T+1 panels.
- **Stale indicator**: Each panel shows last-updated timestamp. Grays out after 2x poll interval without refresh.

### Stock Detail Page

```
┌──────────────────────────────────────────┐
│  000001.SZ 平安银行              10.24 -1.2% │
│  市值 1850亿 · 行业 银行                    │
├──────────────────────────────────────────┤
│  [5日] [20日] [60日] [120日]              │
├──────────────────────────────────────────┤
│                                          │
│  K-line Chart + MA overlays + Volume     │
│  (lightweight-charts, 400px height)      │
│                                          │
├──────────┬──────────┬──────────┬─────────┤
│ RSI 48   │ ADX 32   │Dist→MA20 │BBW 0.12 │
│          │          │  -2.8%   │         │
├──────────┴──────────┴──────────┴─────────┤
│  📝 LLM Analysis: 该股短期RSI中性(48)...    │
│  (cached per stock+date, fades if stale)   │
├──────────────────────────────────────────┤
│  Audit: ✅ PASS  Confidence: 0.78        │
│  Catalysts: tech_breakthrough ×1, ...    │
│  Dragon Tiger: 3 recent appearances      │
└──────────────────────────────────────────┘
```

When the user switches horizon tabs, the factor cards update in-place (no page reload). The K-line chart adjusts its MA overlays and visible range.

### Theme Explorer Page

Sector heatmap → click a sector → drill-down view:
- Constituent stocks ranked by composite score
- Capital flow history for that theme
- Theme trend: accelerating / stable / fading

### Stock Screener Page

Filterable table with columns: ts_code, name, industry, composite_score, consolidation, momentum, volume_boost, audit_verdict. Sortable by any column. Click row → navigate to `/stock/:code`.

Natural language search bar at top: "低估值小市值科技股，近期放量突破" → LLM translates to filter params → filters apply automatically. The LLM also returns a human-readable explanation so the user sees what filters were applied and can adjust them manually.

---

## Visual Style: Dark Professional

```
Background:  #0a0e1a (deep navy)
Card bg:     #131b2e
Border:      #1e293b
Text:        #f8fafc
Muted:       #64748b (slate-500)
Accent:      #8b9cc7
Green:       #22c55e (positive, bullish)
Red:         #ef4444 (negative, bearish)
Yellow:      #eab308 (warning, neutral)
Blue:        #3b82f6 (links, info)
```

Font: Inter (UI), JetBrains Mono (data/codes). Financial data colors: green = up, red = down (Chinese market convention). All panels use the same card component from shadcn/ui with a dark variant. Cards have subtle borders, no harsh shadows.

---

## Key Non-Functional Requirements

- **Per-panel loading**: No panel blocks another. Each fetches independently with skeleton states.
- **yfinance is never called from the frontend**: All external data fetches happen server-side.
- **Read-only data access**: DuckDB opens Parquet files read-only. No risk of dashboard corrupting pipeline data.
- **Graceful degradation**: If yfinance is down, panels show T+1 data with a "Live unavailable" badge.
- **Symlink to data**: Dashboard reads `trend-agent/data/` via config path or symlink. No duplication.
- **No authentication**: Local-only, single-user. Auth can be added later if needed.

---

## Out of Scope

- Real-time WebSocket streaming (yfinance 15min delay makes this unnecessary)
- Floating chat assistant (other LLM integrations included, but free-form chat is deferred)
- Portfolio tracking / watchlist management
- Mobile responsive (desktop-first, Chrome-focused)
- User authentication / multi-user
- Dark/light theme toggle (dark only)
- Backtesting integration
```

