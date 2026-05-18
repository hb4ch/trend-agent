# Contrarian Agent Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + React/shadcn/ui web dashboard that surfaces Contrarian Agent data (macro market intelligence, technical analysis, themes, screening signals) with live yfinance quotes and local Gemma 4 LLM commentary.

**Architecture:** Separate project at `/home/robertpeng/contrarian-dashboard`. Backend reads Parquet data via symlink from `trend-agent/data/`. Frontend fetches per-panel REST endpoints with independent loading states. LLM calls go through backend → vLLM Gemma on DGX Spark with SSE streaming (~12 tok/s). yfinance provides intraday quotes cached at 5-min TTL.

**Tech Stack:** FastAPI + DuckDB + yfinance + Gemma 4 (vLLM) backend; React + TypeScript + shadcn/ui + Recharts + lightweight-charts + Tailwind CSS frontend.

**Latency Budgets (DGX Spark @ 12 tok/s):**
- Briefing (~80 tok): ~7s → SSE streaming to show text arriving
- Stock commentary (~60 tok): ~5s → SSE streaming
- Theme narratives batched (~150 tok): ~6s batched on vLLM, SSE for each narrative
- Signal interpretations batched (~100 tok): ~5s batched, SSE for each interpretation
- NL screener (~30 tok): ~2.5s → SSE streaming

---

## File Structure

```
/home/robertpeng/contrarian-dashboard/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, lifespan, router mounts
│   ├── pyproject.toml
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── macro.py               # GET /api/macro/{indices,sentiment,themes,capital-flow,top-movers,breadth,dragon-tiger,signal-alerts}
│   │   ├── stock.py               # GET /api/stock/{ts_code}
│   │   ├── themes.py              # GET /api/themes
│   │   ├── screener.py            # GET /api/screener, POST /api/screener/nl
│   │   └── llm.py                 # POST /api/llm/{briefing,commentary,commentary/{ts_code},themes,signals}
│   └── services/
│       ├── __init__.py
│       ├── duckdb_service.py      # DuckDB connection, query helpers
│       ├── yfinance_service.py    # yfinance batch fetch + TTL cache
│       ├── fallback_service.py    # Sina/Eastmoney public endpoint scrape
│       ├── compute_service.py     # Factor aggregation, sentiment calc, market breadth
│       ├── pipeline_reader.py     # Read latest pipeline outputs (themes, candidates, audit traces)
│       └── llm_service.py         # vLLM Gemma client, prompt building, batched dispatch, SSE streaming
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── components.json            # shadcn/ui config
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css              # Tailwind directives + dark theme CSS vars
│       ├── pages/
│       │   ├── MacroDashboard.tsx
│       │   ├── ThemeExplorer.tsx
│       │   ├── StockScreener.tsx
│       │   └── StockDetail.tsx
│       ├── components/
│       │   ├── panels/
│       │   │   ├── MarketSnapshotPanel.tsx
│       │   │   ├── SentimentGaugePanel.tsx
│       │   │   ├── HotThemesPanel.tsx
│       │   │   ├── CapitalFlowPanel.tsx
│       │   │   ├── TopMoversPanel.tsx
│       │   │   ├── MarketBreadthPanel.tsx
│       │   │   ├── DragonTigerFeedPanel.tsx
│       │   │   ├── SignalAlertsPanel.tsx
│       │   │   └── BriefingBanner.tsx
│       │   ├── charts/
│       │   │   ├── KLineChart.tsx
│       │   │   ├── SparklineChart.tsx
│       │   │   └── HorizonFactorCards.tsx
│       │   ├── layout/
│       │   │   ├── TopNav.tsx
│       │   │   └── DashboardGrid.tsx
│       │   └── ui/                # shadcn/ui generated (button, card, table, badge, skeleton, etc.)
│       ├── hooks/
│       │   ├── usePolling.ts
│       │   ├── useApiData.ts
│       │   └── useSSE.ts
│       ├── lib/
│       │   └── api.ts
│       └── types/
│           └── index.ts
```

---

## Phase 1: Project Scaffold

### Task 1: Create project directory and backend skeleton

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/main.py`
- Create: `backend/routes/__init__.py`
- Create: `backend/services/__init__.py`

- [ ] **Step 1: Create directory structure**

Run: `mkdir -p /home/robertpeng/contrarian-dashboard/backend/{routes,services}`

- [ ] **Step 2: Write backend/pyproject.toml**

```toml
[project]
name = "contrarian-dashboard-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "duckdb>=1.2.0",
    "yfinance>=0.2.55",
    "httpx>=0.28.0",
    "pandas>=2.2.0",
    "pyarrow>=19.0.0",
    "pydantic>=2.10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.3.0", "pytest-asyncio>=0.25.0", "httpx>=0.28.0"]
```

- [ ] **Step 3: Write backend/services/__init__.py**

```python
# Services package
```

- [ ] **Step 4: Write backend/routes/__init__.py**

```python
# Routes package
```

- [ ] **Step 5: Write backend/main.py**

```python
"""Contrarian Agent Dashboard — FastAPI backend."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import macro, stock, themes, screener, llm
from services.duckdb_service import DuckDBService
from services.yfinance_service import YFinanceService
from services.compute_service import ComputeService
from services.pipeline_reader import PipelineReader
from services.llm_service import LLMService
from services.fallback_service import FallbackService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.duckdb = DuckDBService(data_dir="/home/robertpeng/trend-agent/data")
    app.state.yfinance = YFinanceService()
    app.state.fallback = FallbackService()
    app.state.compute = ComputeService(app.state.duckdb)
    app.state.pipeline = PipelineReader(app.state.duckdb)
    app.state.llm = LLMService(base_url="http://192.168.3.46:8000/v1", model="gemma-4-31B-nvfp4")
    try:
        yield
    finally:
        app.state.duckdb.close()


app = FastAPI(title="Contrarian Agent Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(macro.router, prefix="/api/macro", tags=["macro"])
app.include_router(stock.router, prefix="/api/stock", tags=["stock"])
app.include_router(themes.router, prefix="/api/themes", tags=["themes"])
app.include_router(screener.router, prefix="/api/screener", tags=["screener"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Install backend dependencies**

Run: `cd /home/robertpeng/contrarian-dashboard/backend && uv sync`

- [ ] **Step 7: Verify the app starts**

Run: `cd /home/robertpeng/contrarian-dashboard/backend && timeout 5 uv run uvicorn main:app --port 8000 || true`
Expected: App starts, fails on missing route modules (expected — not yet created)

- [ ] **Step 8: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/pyproject.toml ../contrarian-dashboard/backend/main.py ../contrarian-dashboard/backend/routes/__init__.py ../contrarian-dashboard/backend/services/__init__.py
git commit -m "feat: scaffold backend project with FastAPI skeleton"
```

---

### Task 2: Scaffold frontend with Vite + React + shadcn/ui

**Files:**
- Create: `frontend/` via `npm create vite`
- Create: `frontend/src/index.css`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/components.json`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`

- [ ] **Step 1: Scaffold Vite + React + TypeScript**

Run: `cd /home/robertpeng/contrarian-dashboard && npm create vite@latest frontend -- --template react-ts`

- [ ] **Step 2: Install dependencies**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/frontend && npm install && \
npm install tailwindcss @tailwindcss/vite \
  react-router-dom recharts lightweight-charts \
  lucide-react
```

- [ ] **Step 3: Initialize shadcn/ui**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/frontend
npx shadcn@latest init -d --style new-york --base-color zinc
```

- [ ] **Step 4: Add shadcn/ui components needed**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/frontend
npx shadcn@latest add card table badge skeleton scroll-area separator
```

- [ ] **Step 5: Write frontend/src/index.css**

```css
@import "tailwindcss";

@theme {
  --color-bg-primary: #0a0e1a;
  --color-bg-card: #131b2e;
  --color-bg-hover: #1a2540;
  --color-border-subtle: #1e293b;
  --color-text-primary: #f8fafc;
  --color-text-muted: #64748b;
  --color-accent: #8b9cc7;
  --color-bullish: #22c55e;
  --color-bearish: #ef4444;
  --color-warning: #eab308;
  --color-info: #3b82f6;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-family: 'Inter', sans-serif;
}

code, .mono {
  font-family: 'JetBrains Mono', monospace;
}
```

- [ ] **Step 6: Write frontend/src/main.tsx**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
```

- [ ] **Step 7: Write frontend/src/App.tsx**

```tsx
import { Routes, Route } from 'react-router-dom'
import TopNav from './components/layout/TopNav'
import MacroDashboard from './pages/MacroDashboard'
import ThemeExplorer from './pages/ThemeExplorer'
import StockScreener from './pages/StockScreener'
import StockDetail from './pages/StockDetail'

export default function App() {
  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]">
      <TopNav />
      <main className="max-w-[1600px] mx-auto p-4">
        <Routes>
          <Route path="/" element={<MacroDashboard />} />
          <Route path="/themes" element={<ThemeExplorer />} />
          <Route path="/screener" element={<StockScreener />} />
          <Route path="/stock/:code" element={<StockDetail />} />
        </Routes>
      </main>
    </div>
  )
}
```

- [ ] **Step 8: Verify dev server starts**

Run: `cd /home/robertpeng/contrarian-dashboard/frontend && timeout 10 npm run dev || true`
Expected: Vite starts on port 5173, fails on missing component imports (expected)

- [ ] **Step 9: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/
git commit -m "feat: scaffold frontend with Vite + React + shadcn/ui + Tailwind"
```

---

## Phase 2: Backend Data Services

### Task 3: Implement DuckDB service

**Files:**
- Create: `backend/services/duckdb_service.py`

- [ ] **Step 1: Write backend/services/duckdb_service.py**

```python
"""DuckDB read-only service for Parquet data."""
import os
from pathlib import Path
import duckdb
import pandas as pd


class DuckDBService:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._conn = duckdb.connect(":memory:")
        self._conn.execute("SET threads = 4")

    def close(self):
        self._conn.close()

    def query_parquet(self, path: str, query: str | None = None) -> pd.DataFrame:
        """Run a query against a parquet file or glob pattern."""
        full_path = str(self.data_dir / path)
        if query:
            sql = query.replace("{path}", f"'{full_path}'")
        else:
            sql = f"SELECT * FROM '{full_path}'"
        return self._conn.execute(sql).df()

    def get_stock_ticks(self, ts_code: str) -> pd.DataFrame:
        """Get historical OHLCV for a stock."""
        filepath = self.data_dir / "stock_ticks" / f"{ts_code}.parquet"
        if not filepath.exists():
            return pd.DataFrame()
        return self._conn.execute(
            f"SELECT * FROM '{filepath}' ORDER BY trade_date"
        ).df()

    def get_latest_trade_date(self) -> str | None:
        """Get the most recent trade_date across all tick data."""
        result = self._conn.execute("""
            SELECT MAX(trade_date) FROM '{path}'
        """.replace("{path}", str(self.data_dir / "stock_ticks" / "*.parquet"))).fetchone()
        return result[0] if result else None

    def get_top_list(self, date: str) -> pd.DataFrame:
        """Get Dragon Tiger List for a specific date."""
        filepath = self.data_dir / "top_list" / f"{date}.parquet"
        if not filepath.exists():
            return pd.DataFrame()
        return self._conn.execute(f"SELECT * FROM '{filepath}'").df()

    def get_latest_top_list(self) -> pd.DataFrame:
        """Get the most recent Dragon Tiger List data."""
        import glob
        files = sorted(glob.glob(str(self.data_dir / "top_list" / "*.parquet")))
        if not files:
            return pd.DataFrame()
        return self._conn.execute(f"SELECT * FROM '{files[-1]}'").df()

    def get_top_inst(self, date: str) -> pd.DataFrame:
        """Get institutional Dragon Tiger details for a date."""
        filepath = self.data_dir / "top_inst" / f"{date}.parquet"
        if not filepath.exists():
            return pd.DataFrame()
        return self._conn.execute(f"SELECT * FROM '{filepath}'").df()

    def get_latest_top_inst(self) -> pd.DataFrame:
        """Get the most recent institutional Dragon Tiger data."""
        import glob
        files = sorted(glob.glob(str(self.data_dir / "top_inst" / "*.parquet")))
        if not files:
            return pd.DataFrame()
        return self._conn.execute(f"SELECT * FROM '{files[-1]}'").df()

    def get_factor_panel(self) -> pd.DataFrame:
        """Get the full factor panel."""
        return self._conn.execute(
            f"SELECT * FROM '{self.data_dir}/factor_panel.parquet'"
        ).df()

    def get_factor_panel_for_date(self, trade_date: str) -> pd.DataFrame:
        """Get factor data for a specific trade date."""
        return self._conn.execute(f"""
            SELECT * FROM '{self.data_dir}/factor_panel.parquet'
            WHERE trade_date = '{trade_date}'
        """).df()

    def get_latest_factor_date(self) -> str | None:
        """Get most recent trade_date in factor panel."""
        result = self._conn.execute(
            f"SELECT MAX(trade_date) FROM '{self.data_dir}/factor_panel.parquet'"
        ).fetchone()
        return str(result[0])[:10] if result else None

    def get_signal_labels(self, latest_only: bool = True) -> pd.DataFrame:
        """Get signal labels from validation pipeline."""
        filepath = self.data_dir / "signals" / "signal_labels.parquet"
        if not filepath.exists():
            return pd.DataFrame()
        df = self._conn.execute(f"SELECT * FROM '{filepath}'").df()
        if latest_only and not df.empty:
            latest_date = df["signal_date"].max()
            df = df[df["signal_date"] == latest_date]
        return df

    def query_df(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SQL query."""
        return self._conn.execute(sql).df()
```

- [ ] **Step 2: Verify with a quick test**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run python -c "
from services.duckdb_service import DuckDBService
svc = DuckDBService('/home/robertpeng/trend-agent/data')
df = svc.get_latest_top_list()
print(f'Top list rows: {len(df)}')
print(f'Latest trade date: {svc.get_latest_trade_date()}')
print(f'Factor panel latest: {svc.get_latest_factor_date()}')
"
```
Expected: Prints row counts and dates from real data.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/duckdb_service.py
git commit -m "feat: add DuckDB read-only service for Parquet data"
```

---

### Task 4: Implement yfinance service

**Files:**
- Create: `backend/services/yfinance_service.py`

- [ ] **Step 1: Write backend/services/yfinance_service.py**

```python
"""yfinance intraday quote service with TTL cache."""
import time
import asyncio
from typing import Optional
import pandas as pd
import yfinance as yf

A_INDEX_SYMBOLS = {
    "shanghai": "000001.SS",
    "shenzhen": "399001.SZ",
    "chinext": "399006.SZ",
    "star50": "000688.SS",
}


class YFinanceService:
    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, dict]] = {}

    def _is_fresh(self, key: str) -> bool:
        entry = self._cache.get(key)
        return entry is not None and (time.monotonic() - entry[0]) < self._ttl

    def _get_cached(self, key: str) -> dict | None:
        if self._is_fresh(key):
            return self._cache[key][1]
        return None

    def _set_cache(self, key: str, data: dict):
        self._cache[key] = (time.monotonic(), data)

    async def get_indices(self) -> dict:
        """Get live index quotes with 1-day sparkline."""
        cached = self._get_cached("indices")
        if cached:
            return cached
        result = {}
        for name, symbol in A_INDEX_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                info = ticker.info or {}
                prev_close = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else 0)
                price = hist["Close"].iloc[-1] if not hist.empty else 0
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                sparkline = hist["Close"].tail(60).tolist() if not hist.empty else []
                result[name] = {
                    "price": round(float(price), 2),
                    "change_pct": round(float(change_pct), 2),
                    "sparkline": [round(float(x), 2) for x in sparkline],
                }
            except Exception:
                result[name] = {"price": 0, "change_pct": 0, "sparkline": [], "error": "unavailable"}
        self._set_cache("indices", result)
        return result

    async def get_quote(self, ts_code: str) -> dict:
        """Get live quote for a single A-share stock."""
        symbol = self._ts_code_to_yfinance(ts_code)
        cached = self._get_cached(f"quote_{ts_code}")
        if cached:
            return cached
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            info = ticker.info or {}
            prev_close = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else 0)
            price = hist["Close"].iloc[-1] if not hist.empty else 0
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            result = {
                "price": round(float(price), 2),
                "change_pct": round(float(change_pct), 2),
                "volume": int(hist["Volume"].iloc[-1]) if not hist.empty and "Volume" in hist else 0,
                "high": round(float(hist["High"].iloc[-1]), 2) if not hist.empty else 0,
                "low": round(float(hist["Low"].iloc[-1]), 2) if not hist.empty else 0,
                "open": round(float(hist["Open"].iloc[-1]), 2) if not hist.empty else 0,
            }
            self._set_cache(f"quote_{ts_code}", result)
            return result
        except Exception:
            return {"price": 0, "change_pct": 0, "error": "unavailable"}

    async def get_quotes_batch(self, ts_codes: list[str]) -> dict[str, dict]:
        """Get quotes for multiple stocks. Use yfinance batch download."""
        missing = [c for c in ts_codes if not self._is_fresh(f"quote_{c}")]
        if missing:
            symbols = [self._ts_code_to_yfinance(c) for c in missing]
            try:
                tickers = yf.Tickers(" ".join(symbols))
                for code, symbol in zip(missing, symbols):
                    try:
                        t = tickers.tickers.get(symbol)
                        if t:
                            hist = t.history(period="2d")
                            info = t.info or {}
                            prev_close = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else 0)
                            price = hist["Close"].iloc[-1] if not hist.empty else 0
                            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                            self._set_cache(f"quote_{code}", {
                                "price": round(float(price), 2),
                                "change_pct": round(float(change_pct), 2),
                            })
                    except Exception:
                        self._set_cache(f"quote_{code}", {"price": 0, "change_pct": 0, "error": "unavailable"})
            except Exception:
                for code in missing:
                    self._set_cache(f"quote_{code}", {"price": 0, "change_pct": 0, "error": "unavailable"})

        return {c: self._get_cached(f"quote_{c}") or {"price": 0, "change_pct": 0} for c in ts_codes}

    @staticmethod
    def _ts_code_to_yfinance(ts_code: str) -> str:
        """Convert 000001.SZ → 000001.SZ (yfinance uses same format for A-shares)."""
        return ts_code
```

- [ ] **Step 2: Verify yfinance works**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run python -c "
import asyncio
from services.yfinance_service import YFinanceService
svc = YFinanceService()
async def test():
    indices = await svc.get_indices()
    for name, data in indices.items():
        print(f'{name}: {data[\"price\"]} ({data[\"change_pct\"]:+.2f}%)')
asyncio.run(test())
"
```
Expected: Prints index prices (may show 0 if market closed).

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/yfinance_service.py
git commit -m "feat: add yfinance service with 5min TTL cache"
```

---

### Task 5: Implement pipeline reader service

**Files:**
- Create: `backend/services/pipeline_reader.py`

- [ ] **Step 1: Write backend/services/pipeline_reader.py**

```python
"""Read latest outputs from the trend-agent pipeline."""
import os
import glob
import json
import pandas as pd
from pathlib import Path


class PipelineReader:
    def __init__(self, duckdb_service):
        self.duckdb = duckdb_service
        self.reports_dir = Path("/home/robertpeng/trend-agent/reports")

    def get_latest_candidates(self) -> pd.DataFrame:
        """Get the most recent candidates CSV."""
        files = sorted(glob.glob(str(self.reports_dir / "candidates_*.csv")))
        if not files:
            return pd.DataFrame()
        return pd.read_csv(files[-1])

    def get_latest_audit_trace(self) -> list[dict]:
        """Get the most recent audit trace as parsed JSON objects."""
        files = sorted(glob.glob(str(self.reports_dir / "audit_trace_*.jsonl")))
        if not files:
            return []
        results = []
        with open(files[-1]) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return results

    def get_audit_for_stock(self, ts_code: str) -> dict | None:
        """Get the audit result for a specific stock from the latest trace."""
        trace = self.get_latest_audit_trace()
        for record in trace:
            payload = record.get("payload", {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    continue
            stock = payload.get("ts_code", "") or payload.get("code", "")
            if stock == ts_code:
                return payload
        return None

    def get_latest_themes(self) -> list[dict]:
        """Extract themes from latest deepseek trace."""
        files = sorted(glob.glob(str(self.reports_dir / "deepseek_trace_*.jsonl")))
        if not files:
            return []
        themes = []
        with open(files[-1]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                content = payload.get("content", "")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        continue
                for theme in content.get("themes", []):
                    themes.append(theme)
        return themes
```

- [ ] **Step 2: Verify with real data**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run python -c "
from services.duckdb_service import DuckDBService
from services.pipeline_reader import PipelineReader
svc = DuckDBService('/home/robertpeng/trend-agent/data')
reader = PipelineReader(svc)
candidates = reader.get_latest_candidates()
print(f'Latest candidates: {len(candidates)} rows')
print(f'Columns: {list(candidates.columns)[:10]}')
themes = reader.get_latest_themes()
print(f'Themes: {len(themes)} found')
"
```
Expected: Prints candidate count and theme count from real data.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/pipeline_reader.py
git commit -m "feat: add pipeline reader for latest candidates, audits, themes"
```

---

### Task 6: Implement compute service (sentiment, breadth, factor aggregation)

**Files:**
- Create: `backend/services/compute_service.py`

- [ ] **Step 1: Write backend/services/compute_service.py**

```python
"""Compute market aggregates: sentiment, breadth, factor aggregation."""
import pandas as pd
import numpy as np


class ComputeService:
    def __init__(self, duckdb_service):
        self.duckdb = duckdb_service

    def compute_sentiment(self, top_list: pd.DataFrame, top_inst: pd.DataFrame) -> dict:
        """Compute composite sentiment score from Dragon Tiger List data."""
        if top_list.empty:
            return {"score": 50, "label": "NO DATA", "components": {}}

        # Capital flow signal (-1 to 1)
        net_amount = top_list["net_amount"].sum() if "net_amount" in top_list else 0
        l_amount = top_list["l_amount"].sum() if "l_amount" in top_list else 1
        capital_flow_score = min(1.0, max(-1.0, net_amount / max(abs(l_amount), 1e8))) if l_amount else 0
        capital_flow_normalized = (capital_flow_score + 1) / 2  # 0 to 1

        # Participation breadth (how many stocks hit the list)
        stock_count = len(top_list["ts_code"].unique()) if "ts_code" in top_list else 0
        breadth_score = min(1.0, stock_count / 100)
        breadth_normalized = breadth_score

        # Theme strength (from institutional participation)
        if not top_inst.empty and "net_buy" in top_inst:
            inst_net = top_inst["net_buy"].sum()
            inst_total = top_inst[["buy", "sell"]].sum().sum() if "buy" in top_inst else 1
            theme_score = min(1.0, max(-1.0, inst_net / max(abs(inst_total), 1e8)))
            theme_normalized = (theme_score + 1) / 2
        else:
            theme_normalized = 0.5

        composite = round(
            capital_flow_normalized * 0.4 + breadth_normalized * 0.3 + theme_normalized * 0.3,
            2,
        )
        score = int(composite * 100)

        label = "BEARISH"
        if score >= 70:
            label = "BULLISH"
        elif score >= 55:
            label = "NEUTRAL-BULLISH"
        elif score >= 45:
            label = "NEUTRAL"
        elif score >= 30:
            label = "NEUTRAL-BEARISH"

        return {
            "score": score,
            "label": label,
            "components": {
                "capital_flow": round(capital_flow_normalized, 2),
                "breadth": round(breadth_normalized, 2),
                "theme_strength": round(theme_normalized, 2),
            },
        }

    def compute_breadth(self) -> dict:
        """Compute market breadth indicators from factor panel."""
        try:
            latest_date = self.duckdb.get_latest_factor_date()
            if not latest_date:
                return self._empty_breadth()
            df = self.duckdb.get_factor_panel_for_date(latest_date)
            if df.empty:
                return self._empty_breadth()

            total = len(df)
            above_ma20 = int((df["dist_ma20"] > 0).sum()) if "dist_ma20" in df else 0
            above_ma60 = int((df["dist_ma60"] > 0).sum()) if "dist_ma60" in df else 0
            pct_above_ma20 = round(above_ma20 / total * 100, 1) if total else 0
            pct_above_ma60 = round(above_ma60 / total * 100, 1) if total else 0

            # Advance/decline from returns
            adv = int((df["ret_5d"] > 0).sum()) if "ret_5d" in df else 0
            dec = int((df["ret_5d"] < 0).sum()) if "ret_5d" in df else 0
            ad_ratio = round(adv / dec, 2) if dec else 1.0

            # Volume breadth: % of stocks where volume > 20d average
            vol_breadth = round(float((df["vol_ratio_5_20"] > 1).sum()) / total * 100, 1) if "vol_ratio_5_20" in df else 0

            return {
                "advance_decline_ratio": ad_ratio,
                "pct_above_ma20": pct_above_ma20,
                "pct_above_ma60": pct_above_ma60,
                "volume_breadth": vol_breadth,
            }
        except Exception:
            return self._empty_breadth()

    def _empty_breadth(self) -> dict:
        return {"advance_decline_ratio": 0, "pct_above_ma20": 0, "pct_above_ma60": 0, "volume_breadth": 0}

    def get_stock_factors(self, ts_code: str) -> dict:
        """Get multi-horizon factor data for a stock."""
        try:
            df = self.duckdb.query_parquet(
                "factor_panel.parquet",
                f"SELECT * FROM {{path}} WHERE ts_code = '{ts_code}' ORDER BY trade_date DESC LIMIT 120"
            )
            if df.empty:
                return {}

            horizons = {"5d": 5, "20d": 20, "60d": 60, "120d": 120}
            result = {}
            for horizon_name, days in horizons.items():
                subset = df.head(days)
                if subset.empty:
                    continue
                result[horizon_name] = {
                    "ret": round(float(subset["ret_5d"].iloc[-1]), 2) if "ret_5d" in subset else 0,
                    "rsi_14": round(float(subset["rsi_14"].iloc[0]), 1) if "rsi_14" in subset else 50,
                    "adx": round(float(subset["adx"].iloc[0]), 1) if "adx" in subset else 0,
                    "dist_ma20": round(float(subset["dist_ma20"].iloc[0]), 2) if "dist_ma20" in subset else 0,
                    "dist_ma60": round(float(subset["dist_ma60"].iloc[0]), 2) if "dist_ma60" in subset else 0,
                    "atr_pct": round(float(subset["atr_pct"].iloc[0]), 2) if "atr_pct" in subset else 0,
                    "bbw": round(float(subset["bbw"].iloc[0]), 2) if "bbw" in subset else 0,
                }
            return result
        except Exception:
            return {}

    def compute_top_movers(self, candidates: pd.DataFrame) -> list[dict]:
        """Extract top 10 from candidates, sorted by composite score."""
        if candidates.empty:
            return []
        score_col = "composite_score" if "composite_score" in candidates.columns else None
        if score_col is None:
            # Look for score-related columns
            for col in ["final_score", "total_score", "score"]:
                if col in candidates.columns:
                    score_col = col
                    break
        if score_col:
            df = candidates.nlargest(10, score_col)
        else:
            df = candidates.head(10)
        cols = ["ts_code", "name", "industry"]
        keep = [c for c in cols if c in df.columns]
        return df[keep].to_dict(orient="records")
```

- [ ] **Step 2: Quick smoke test**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run python -c "
from services.duckdb_service import DuckDBService
from services.compute_service import ComputeService
svc = DuckDBService('/home/robertpeng/trend-agent/data')
comp = ComputeService(svc)
breadth = comp.compute_breadth()
print(f'Breadth: {breadth}')
factors = comp.get_stock_factors('000001.SZ')
print(f'000001.SZ factors: {list(factors.keys())}')
"
```
Expected: Prints breadth dict and factor horizons.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/compute_service.py
git commit -m "feat: add compute service for sentiment, breadth, factors"
```

---

### Task 7: Implement fallback service (Sina/Eastmoney)

**Files:**
- Create: `backend/services/fallback_service.py`

- [ ] **Step 1: Write backend/services/fallback_service.py**

```python
"""Fallback live quote service using Sina and Eastmoney public endpoints."""
import asyncio
import httpx


class FallbackService:
    SINA_API = "http://hq.sinajs.cn/list={codes}"
    EASTMONEY_API = "https://push2.eastmoney.com/api/qt/stock/get"

    async def get_quote_sina(self, ts_code: str) -> dict | None:
        """Fetch from Sina finance API."""
        code = ts_code.replace(".SZ", "").replace(".SH", "")
        prefix = "sz" if ".SZ" in ts_code else "sh"
        sina_code = f"{prefix}{code}"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    self.SINA_API.format(codes=sina_code),
                    headers={"Referer": "https://finance.sina.com.cn"},
                )
                if resp.status_code != 200:
                    return None
                text = resp.text
                if "var hq_str_" not in text:
                    return None
                data = text.split('"')[1].split(",")
                if len(data) < 32:
                    return None
                return {
                    "price": float(data[3]),
                    "change_pct": round((float(data[3]) - float(data[2])) / float(data[2]) * 100, 2),
                    "high": float(data[4]),
                    "low": float(data[5]),
                    "open": float(data[1]),
                    "volume": int(float(data[8])),
                }
        except Exception:
            return None

    async def get_quote_eastmoney(self, ts_code: str) -> dict | None:
        """Fetch from Eastmoney push API."""
        code = ts_code.replace(".SZ", "").replace(".SH", "")
        market = "0" if ".SZ" in ts_code else "1"
        secid = f"{market}.{code}"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    self.EASTMONEY_API,
                    params={
                        "secid": secid,
                        "fields": "f43,f44,f45,f46,f47,f48,f60,f170",
                        "forcect": "1",
                    },
                )
                if resp.status_code != 200:
                    return None
                data = resp.json().get("data", {})
                if not data:
                    return None
                return {
                    "price": data.get("f43", 0) / 100 if data.get("f43") else 0,
                    "change_pct": data.get("f170", 0) / 100 if data.get("f170") else 0,
                    "high": data.get("f44", 0) / 100 if data.get("f44") else 0,
                    "low": data.get("f45", 0) / 100 if data.get("f45") else 0,
                    "open": data.get("f46", 0) / 100 if data.get("f46") else 0,
                    "volume": data.get("f47", 0),
                }
        except Exception:
            return None

    async def get_quote(self, ts_code: str) -> dict | None:
        """Try Sina first, fall back to Eastmoney."""
        result = await self.get_quote_sina(ts_code)
        if result:
            return result
        return await self.get_quote_eastmoney(ts_code)
```

- [ ] **Step 2: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/fallback_service.py
git commit -m "feat: add Sina/Eastmoney fallback for yfinance gaps"
```

---

## Phase 3: Backend LLM Service

### Task 8: Implement LLM service with vLLM client and SSE streaming

**Files:**
- Create: `backend/services/llm_service.py`

- [ ] **Step 1: Write backend/services/llm_service.py**

```python
"""Gemma 4 LLM service via DGX Spark vLLM endpoint.
Latency: ~12 tok/s. All generation uses SSE streaming.
Batch dispatch exploits vLLM continuous batching.
"""
import asyncio
import json
from typing import AsyncIterator
import httpx


class LLMService:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def health_check(self) -> bool:
        """Check if Gemma is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate_stream(self, prompt: str, max_tokens: int = 128) -> AsyncIterator[str]:
        """Generate text with SSE streaming. Yields tokens as they arrive."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """Non-streaming convenience wrapper."""
        parts = []
        async for token in self.generate_stream(prompt, max_tokens):
            parts.append(token)
        return "".join(parts)

    async def generate_batched(self, prompts: list[str], max_tokens: int = 128) -> list[str]:
        """Fire all prompts concurrently — vLLM continuous batching handles it.
        For N prompts, total time ≈ max(single_time) not sum(single_time)."""
        async def collect(prompt: str) -> str:
            parts = []
            async for token in self.generate_stream(prompt, max_tokens):
                parts.append(token)
            return "".join(parts)
        return list(await asyncio.gather(*[collect(p) for p in prompts]))

    # --- Prompt builders ---

    BRIEFING_PROMPT = """你是一个A股市场分析师。基于以下市场数据，用2-3句话总结今日市场概况（中文，80字以内）：

指数表现：{indices}
市场情绪：{sentiment}
热门板块：{themes}
资金流向：{capital_flow}
市场宽度：{breadth}

简洁有力地概括市场状态和关键主题。"""

    def build_briefing_prompt(self, indices: dict, sentiment: dict, themes: list, capital_flow: dict, breadth: dict) -> str:
        return self.BRIEFING_PROMPT.format(
            indices=json.dumps(indices, ensure_ascii=False),
            sentiment=json.dumps(sentiment, ensure_ascii=False),
            themes=json.dumps(themes[:5], ensure_ascii=False),
            capital_flow=json.dumps(capital_flow, ensure_ascii=False),
            breadth=json.dumps(breadth, ensure_ascii=False),
        )

    COMMENTARY_PROMPT = """你是一个技术分析师。基于以下技术指标，用2-3句话解读该股的走势（中文，60字以内）：

股票：{name} ({ts_code})
当前价格：{price}（{change_pct:+.2f}%）
各周期指标：{factors}

简洁解读多周期指标的矛盾或共振，给出关键观察点。不要给出买卖建议。"""

    def build_commentary_prompt(self, name: str, ts_code: str, price: float, change_pct: float, factors: dict) -> str:
        return self.COMMENTARY_PROMPT.format(
            name=name, ts_code=ts_code, price=price, change_pct=change_pct,
            factors=json.dumps(factors, ensure_ascii=False),
        )

    THEME_PROMPT = """你是一个A股主题分析师。为以下每个热门板块写1句话解读（中文，30字以内/板块）：

{themes}

对每个板块，解释资金逻辑和关注点。返回JSON数组：[{{"theme": "板块名", "text": "解读"}}, ...]"""

    def build_theme_prompt(self, themes: list[dict]) -> str:
        return self.THEME_PROMPT.format(themes=json.dumps(themes, ensure_ascii=False))

    SIGNAL_PROMPT = """你是一个交易信号分析师。为以下信号写1句话解读（中文，25字以内/条）：

{signals}

解读为什么这个信号值得关注。返回JSON数组：[{{"ts_code": "...", "text": "解读"}}, ...]"""

    def build_signal_prompt(self, signals: list[dict]) -> str:
        return self.SIGNAL_PROMPT.format(signals=json.dumps(signals, ensure_ascii=False))

    NL_SCREENER_PROMPT = """将用户的自然语言筛选条件转换为结构化过滤参数。

可用的过滤字段：industry（行业）, pe_max（最大PE）, pe_min（最小PE）, total_mv_min（最小总市值亿）, total_mv_max（最大总市值亿）, vol_ratio_min（最小量比）, min_score（最低综合得分）, horizon（周期：5/20/60/120）

返回JSON：{{"filters": {{...}}, "explanation": "解释转换逻辑，中文"}}

用户输入：{query}"""

    def build_nl_screener_prompt(self, query: str) -> str:
        return self.NL_SCREENER_PROMPT.format(query=query)
```

- [ ] **Step 2: Verify Gemma connectivity**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run python -c "
import asyncio
from services.llm_service import LLMService
async def test():
    llm = LLMService('http://192.168.3.46:8000/v1', 'gemma-4-31B-nvfp4')
    ok = await llm.health_check()
    print(f'Gemma reachable: {ok}')
    if ok:
        result = await llm.generate('Hello, say hi in Chinese in 5 words.')
        print(f'Response: {result}')
asyncio.run(test())
"
```
Expected: `Gemma reachable: True` and a Chinese greeting.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/services/llm_service.py
git commit -m "feat: add LLM service with vLLM Gemma, SSE streaming, batch dispatch"
```

---

## Phase 4: Backend API Routes

### Task 9: Implement macro API routes (8 per-panel endpoints)

**Files:**
- Create: `backend/routes/macro.py`

- [ ] **Step 1: Write backend/routes/macro.py**

```python
"""Macro dashboard API endpoints — one per panel."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/indices")
async def macro_indices(request: Request):
    indices = await request.app.state.yfinance.get_indices()
    return indices


@router.get("/sentiment")
async def macro_sentiment(request: Request):
    duckdb = request.app.state.duckdb
    compute = request.app.state.compute
    latest = duckdb.get_latest_top_list()
    latest_inst = duckdb.get_latest_top_inst()
    return compute.compute_sentiment(latest, latest_inst)


@router.get("/themes")
async def macro_themes(request: Request):
    pipeline = request.app.state.pipeline
    themes = pipeline.get_latest_themes()
    return themes[:8] if themes else []


@router.get("/capital-flow")
async def macro_capital_flow(request: Request):
    duckdb = request.app.state.duckdb
    latest = duckdb.get_latest_top_inst()
    if latest.empty:
        return {"northbound": 0, "institutional": 0, "retail_active": 0, "history": []}
    # Classify by exalter type
    northbound = float(latest[latest["exalter"].str.contains("深股通|沪股通|北上", na=False)]["net_buy"].sum()) if "exalter" in latest and "net_buy" in latest else 0
    institutional = float(latest[~latest["exalter"].str.contains("深股通|沪股通|北上|游资|散户", na=False)]["net_buy"].sum()) if "exalter" in latest and "net_buy" in latest else 0
    retail = float(latest[latest["exalter"].str.contains("游资", na=False)]["net_buy"].sum()) if "exalter" in latest and "net_buy" in latest else 0
    return {
        "northbound": round(northbound / 1e8, 1),
        "institutional": round(institutional / 1e8, 1),
        "retail_active": round(retail / 1e8, 1),
        "history": [],
    }


@router.get("/top-movers")
async def macro_top_movers(request: Request):
    pipeline = request.app.state.pipeline
    compute = request.app.state.compute
    candidates = pipeline.get_latest_candidates()
    return compute.compute_top_movers(candidates)


@router.get("/breadth")
async def macro_breadth(request: Request):
    compute = request.app.state.compute
    return compute.compute_breadth()


@router.get("/dragon-tiger")
async def macro_dragon_tiger(request: Request):
    duckdb = request.app.state.duckdb
    latest = duckdb.get_latest_top_list()
    if latest.empty:
        return []
    cols = ["ts_code", "name", "close", "pct_change", "net_amount", "reason"]
    keep = [c for c in cols if c in latest.columns]
    result = latest[keep].head(30).to_dict(orient="records")
    for r in result:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
    return result


@router.get("/signal-alerts")
async def macro_signal_alerts(request: Request):
    duckdb = request.app.state.duckdb
    labels = duckdb.get_signal_labels(latest_only=True)
    if labels.empty:
        return []
    cols = ["ts_code", "name", "entry_date", "industry"]
    keep = [c for c in cols if c in labels.columns]
    return labels[keep].head(8).to_dict(orient="records")
```

- [ ] **Step 2: Verify endpoints**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run uvicorn main:app --port 8000 &
sleep 3
for ep in indices sentiment themes "capital-flow" "top-movers" breadth "dragon-tiger" "signal-alerts"; do
  echo "=== /api/macro/$ep ==="
  curl -s "http://localhost:8000/api/macro/$ep" | head -c 200
  echo
done
kill %1 2>/dev/null
```
Expected: Each endpoint returns JSON.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/routes/macro.py
git commit -m "feat: add 8 per-panel macro API endpoints"
```

---

### Task 10: Implement stock API endpoint

**Files:**
- Create: `backend/routes/stock.py`

- [ ] **Step 1: Write backend/routes/stock.py**

```python
"""Stock detail API endpoint."""
from fastapi import APIRouter, Request, Query

router = APIRouter()


@router.get("/{ts_code:path}")
async def stock_detail(
    request: Request,
    ts_code: str,
    horizon: int = Query(default=20, ge=5, le=120),
):
    duckdb = request.app.state.duckdb
    yfinance = request.app.state.yfinance
    compute = request.app.state.compute
    pipeline = request.app.state.pipeline

    # Live quote
    live = await yfinance.get_quote(ts_code)

    # Historical OHLCV
    ticks = duckdb.get_stock_ticks(ts_code)
    ohlcv = []
    if not ticks.empty:
        for _, row in ticks.tail(250).iterrows():
            ohlcv.append({
                "date": str(row["trade_date"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["vol"]),
                "turnover_rate": float(row.get("turnover_rate", 0) or 0),
                "source": "history",
            })

    # Multi-horizon factors
    factors = compute.get_stock_factors(ts_code)

    # Audit result
    audit = pipeline.get_audit_for_stock(ts_code) or {}

    # Dragon Tiger presence
    latest_dt = duckdb.get_latest_top_list()
    dt_presence = {}
    if not latest_dt.empty and "ts_code" in latest_dt.columns:
        stock_dt = latest_dt[latest_dt["ts_code"] == ts_code]
        dt_presence = {
            "recent_appearances": len(stock_dt),
            "net_flow": round(float(stock_dt["net_amount"].sum() / 1e8), 1) if "net_amount" in stock_dt else 0,
            "last_reason": str(stock_dt["reason"].iloc[0]) if "reason" in stock_dt and len(stock_dt) > 0 else "",
        }

    # Stock name
    name = ""
    if not ticks.empty and "ts_code" in ticks.columns:
        # Try from latest candidates for name
        candidates = pipeline.get_latest_candidates()
        if not candidates.empty and "ts_code" in candidates.columns and "name" in candidates.columns:
            match = candidates[candidates["ts_code"] == ts_code]
            if not match.empty:
                name = str(match["name"].iloc[0])

    return {
        "ts_code": ts_code,
        "name": name,
        "live_price": live["price"],
        "live_change_pct": live["change_pct"],
        "ohlcv": ohlcv,
        "factors": factors,
        "audit": audit,
        "dragon_tiger_presence": dt_presence,
    }
```

- [ ] **Step 2: Verify endpoint**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run uvicorn main:app --port 8000 &
sleep 3
curl -s "http://localhost:8000/api/stock/000001.SZ" | python -m json.tool | head -30
kill %1 2>/dev/null
```
Expected: JSON with live_price, ohlcv, factors, audit fields.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/routes/stock.py
git commit -m "feat: add stock detail API endpoint with live+history merge"
```

---

### Task 11: Implement themes and screener API endpoints

**Files:**
- Create: `backend/routes/themes.py`
- Create: `backend/routes/screener.py`

- [ ] **Step 1: Write backend/routes/themes.py**

```python
"""Theme explorer API endpoint."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def theme_list(request: Request):
    pipeline = request.app.state.pipeline
    themes = pipeline.get_latest_themes()
    return {"themes": themes}
```

- [ ] **Step 2: Write backend/routes/screener.py**

```python
"""Stock screener API endpoints."""
from fastapi import APIRouter, Request, Query
from pydantic import BaseModel

router = APIRouter()


class NLQuery(BaseModel):
    query: str


@router.get("")
async def screener_list(
    request: Request,
    industry: str | None = Query(default=None),
    min_score: float = Query(default=0.0),
    horizon: int = Query(default=20, ge=5, le=120),
):
    pipeline = request.app.state.pipeline
    df = pipeline.get_latest_candidates()
    if df.empty:
        return {"candidates": []}

    if industry:
        df = df[df["industry"].str.contains(industry, na=False)] if "industry" in df else df

    # Pick score column
    score_col = None
    for col in ["composite_score", "final_score", "total_score", "score"]:
        if col in df.columns:
            score_col = col
            break
    if score_col and min_score > 0:
        df = df[df[score_col] >= min_score]

    cols = ["ts_code", "name", "industry"]
    if score_col:
        cols.append(score_col)
    keep = [c for c in cols if c in df.columns]
    result = df[keep].to_dict(orient="records")
    return {"candidates": result}


@router.post("/nl")
async def screener_nl(request: Request, body: NLQuery):
    llm = request.app.state.llm
    prompt = llm.build_nl_screener_prompt(body.query)
    import json
    response = await llm.generate(prompt, max_tokens=80)
    try:
        parsed = json.loads(response)
        filters = parsed.get("filters", {})
        explanation = parsed.get("explanation", "")
    except json.JSONDecodeError:
        filters = {}
        explanation = "无法解析筛选条件"

    # Apply LLM-derived filters by reusing the GET logic
    pipeline = request.app.state.pipeline
    df = pipeline.get_latest_candidates()
    if df.empty:
        return {"filters": filters, "explanation": explanation, "candidates": []}

    if filters.get("industry"):
        df = df[df["industry"].str.contains(filters["industry"], na=False)] if "industry" in df else df
    if filters.get("pe_max") and "pe" in df.columns:
        df = df[df["pe"] <= filters["pe_max"]]
    if filters.get("total_mv_max") and "total_mv" in df.columns:
        df = df[df["total_mv"] / 1e8 <= filters["total_mv_max"]]
    if filters.get("vol_ratio_min") and "volume_ratio" in df.columns:
        df = df[df["volume_ratio"] >= filters["vol_ratio_min"]]

    cols = ["ts_code", "name", "industry"]
    keep = [c for c in cols if c in df.columns]
    return {
        "filters": filters,
        "explanation": explanation,
        "candidates": df[keep].head(20).to_dict(orient="records"),
    }
```

- [ ] **Step 3: Verify endpoints**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run uvicorn main:app --port 8000 &
sleep 3
echo "=== Themes ===" && curl -s "http://localhost:8000/api/themes" | head -c 200
echo
echo "=== Screener ===" && curl -s "http://localhost:8000/api/screener" | head -c 200
echo
kill %1 2>/dev/null
```
Expected: Both endpoints return JSON.

- [ ] **Step 4: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/routes/themes.py ../contrarian-dashboard/backend/routes/screener.py
git commit -m "feat: add themes and screener API endpoints (with NL search)"
```

---

### Task 12: Implement LLM API routes

**Files:**
- Create: `backend/routes/llm.py`

- [ ] **Step 1: Write backend/routes/llm.py**

```python
"""LLM API endpoints — all return SSE streams."""
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class CommentaryRequest(BaseModel):
    name: str = ""
    ts_code: str = ""
    price: float = 0
    change_pct: float = 0
    factors: dict = {}


async def sse_stream(generator):
    """Wrap async token generator as SSE response."""
    async def event_stream():
        async for token in generator:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/briefing")
async def llm_briefing(request: Request):
    """Generate daily briefing. Fetches live data, builds prompt, streams result."""
    yfinance = request.app.state.yfinance
    duckdb = request.app.state.duckdb
    compute = request.app.state.compute
    pipeline = request.app.state.pipeline
    llm = request.app.state.llm

    indices = await yfinance.get_indices()
    sentiment = compute.compute_sentiment(duckdb.get_latest_top_list(), duckdb.get_latest_top_inst())
    themes = pipeline.get_latest_themes()
    capital_flow = {"northbound": 0, "institutional": 0, "retail_active": 0}  # computed from macro/capital-flow
    breadth = compute.compute_breadth()

    prompt = llm.build_briefing_prompt(indices, sentiment, themes, capital_flow, breadth)
    return await sse_stream(llm.generate_stream(prompt, max_tokens=120))


@router.post("/commentary/{ts_code}")
async def llm_commentary(ts_code: str, request: Request, body: CommentaryRequest = None):
    """Generate per-stock commentary. Streams result."""
    llm = request.app.state.llm
    yfinance = request.app.state.yfinance
    compute = request.app.state.compute

    if body and body.factors:
        factors = body.factors
    else:
        factors = compute.get_stock_factors(ts_code)

    if body and body.price:
        price = body.price
        change_pct = body.change_pct
    else:
        live = await yfinance.get_quote(ts_code)
        price = live["price"]
        change_pct = live["change_pct"]

    prompt = llm.build_commentary_prompt(
        name=body.name if body else "",
        ts_code=ts_code,
        price=price,
        change_pct=change_pct,
        factors=factors,
    )
    return await sse_stream(llm.generate_stream(prompt, max_tokens=100))


@router.post("/themes")
async def llm_themes(request: Request):
    """Generate theme narratives (batched). Streams each narrative as SSE event."""
    pipeline = request.app.state.pipeline
    llm = request.app.state.llm

    themes = pipeline.get_latest_themes()[:6]
    if not themes:
        return await sse_stream(_empty_stream())

    prompt = llm.build_theme_prompt(themes)
    return await sse_stream(llm.generate_stream(prompt, max_tokens=200))


@router.post("/signals")
async def llm_signals(request: Request):
    """Generate signal interpretations (batched). Streams SSE."""
    duckdb = request.app.state.duckdb
    llm = request.app.state.llm

    labels = duckdb.get_signal_labels(latest_only=True)
    if labels.empty:
        return await sse_stream(_empty_stream())

    signals = labels.head(8).to_dict(orient="records")
    prompt = llm.build_signal_prompt([
        {"ts_code": s.get("ts_code", ""), "name": s.get("name", "")}
        for s in signals
    ])
    return await sse_stream(llm.generate_stream(prompt, max_tokens=180))


@router.get("/health")
async def llm_health(request: Request):
    """Check if Gemma is reachable."""
    llm = request.app.state.llm
    ok = await llm.health_check()
    return {"available": ok}


async def _empty_stream():
    yield "无数据"
```

- [ ] **Step 2: Test SSE streaming**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run uvicorn main:app --port 8000 &
sleep 3
echo "=== LLM Health ===" && curl -s "http://localhost:8000/api/llm/health"
echo
# Test briefing (SSE stream, grab first few events)
timeout 20 curl -s "http://localhost:8000/api/llm/briefing" -X POST || true
kill %1 2>/dev/null
```
Expected: Health check returns `{"available": true}`, briefing streams token events.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/backend/routes/llm.py
git commit -m "feat: add LLM API routes with SSE streaming for briefing, commentary, themes, signals"
```

---

## Phase 5: Frontend Core Components

### Task 13: Create TopNav and DashboardGrid layout components

**Files:**
- Create: `frontend/src/components/layout/TopNav.tsx`
- Create: `frontend/src/components/layout/DashboardGrid.tsx`

- [ ] **Step 1: Write frontend/src/components/layout/TopNav.tsx**

```tsx
import { NavLink } from 'react-router-dom'

export default function TopNav() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
      isActive
        ? 'bg-[var(--color-bullish)] text-white'
        : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'
    }`

  return (
    <nav className="border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-card)]">
      <div className="max-w-[1600px] mx-auto px-4 h-12 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <span className="text-lg font-bold text-[var(--color-text-primary)]">
            Contrarian Agent
          </span>
          <div className="flex gap-2">
            <NavLink to="/" className={linkClass} end>大盘</NavLink>
            <NavLink to="/themes" className={linkClass}>主题</NavLink>
            <NavLink to="/screener" className={linkClass}>选股</NavLink>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-[var(--color-text-muted)]">
            数据更新: {new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[var(--color-bullish)]" />
            <span className="text-[var(--color-bullish)]">Live</span>
          </span>
        </div>
      </div>
    </nav>
  )
}
```

- [ ] **Step 2: Write frontend/src/components/layout/DashboardGrid.tsx**

```tsx
import { ReactNode } from 'react'

interface DashboardGridProps {
  briefing: ReactNode
  topRow: [ReactNode, ReactNode]
  middleRow: [ReactNode, ReactNode]
  bottomRow: [ReactNode, ReactNode]
  breadth: ReactNode
  feed: ReactNode
}

export default function DashboardGrid({
  briefing, topRow, middleRow, bottomRow, breadth, feed,
}: DashboardGridProps) {
  return (
    <div className="space-y-3">
      {/* Briefing banner — full width */}
      <div>{briefing}</div>

      {/* Top row: Market Snapshot (2/3) + Hot Themes (1/3) */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">{topRow[0]}</div>
        <div className="col-span-1">{topRow[1]}</div>
      </div>

      {/* Middle row: Sentiment (2/3) + Capital Flow (1/3) */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">{middleRow[0]}</div>
        <div className="col-span-1">{middleRow[1]}</div>
      </div>

      {/* Bottom row: Top Movers (2/3) + Signal Alerts (1/3) */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">{bottomRow[0]}</div>
        <div className="col-span-1">{bottomRow[1]}</div>
      </div>

      {/* Market Breadth — full width */}
      <div>{breadth}</div>

      {/* Dragon Tiger Feed — full width */}
      <div>{feed}</div>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/components/layout/TopNav.tsx ../contrarian-dashboard/frontend/src/components/layout/DashboardGrid.tsx
git commit -m "feat: add TopNav and DashboardGrid layout components"
```

---

### Task 14: Create API client and data hooks

**Files:**
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useApiData.ts`
- Create: `frontend/src/hooks/useSSE.ts`
- Create: `frontend/src/hooks/usePolling.ts`

- [ ] **Step 1: Write frontend/src/lib/api.ts**

```tsx
const BASE = 'http://localhost:8000/api'

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`)
  return res.json()
}

export function fetchSSE(
  path: string,
  options: { method?: string; body?: string },
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): AbortController {
  const controller = new AbortController()
  fetch(`${BASE}${path}`, {
    method: options.method || 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: options.body,
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`SSE ${res.status}`)
      const reader = res.body?.getReader()
      if (!reader) throw new Error('No body')
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') { onDone(); return }
            try {
              const parsed = JSON.parse(data)
              if (parsed.token) onToken(parsed.token)
            } catch {}
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err)
    })
  return controller
}
```

- [ ] **Step 2: Write frontend/src/hooks/useApiData.ts**

```tsx
import { useState, useEffect, useCallback } from 'react'

interface UseApiDataResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
}

export function useApiData<T>(path: string, pollIntervalMs?: number): UseApiDataResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`http://localhost:8000/api${path}`)
      if (!res.ok) throw new Error(`${res.status}`)
      const json = await res.json()
      setData(json)
      setError(null)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [path])

  useEffect(() => {
    setLoading(true)
    fetchData()
    if (pollIntervalMs) {
      const interval = setInterval(fetchData, pollIntervalMs)
      return () => clearInterval(interval)
    }
  }, [fetchData, pollIntervalMs])

  return { data, loading, error, refetch: fetchData }
}
```

- [ ] **Step 3: Write frontend/src/hooks/useSSE.ts**

```tsx
import { useState, useEffect, useRef, useCallback } from 'react'

interface UseSSEResult {
  text: string
  streaming: boolean
  error: string | null
  start: (path: string, method?: string, body?: string) => void
  cancel: () => void
}

export function useSSE(): UseSSEResult {
  const [text, setText] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const start = useCallback((path: string, method = 'GET', body?: string) => {
    setText('')
    setStreaming(true)
    setError(null)

    const controller = new AbortController()
    controllerRef.current = controller

    fetch(`http://localhost:8000/api${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`SSE ${res.status}`)
        const reader = res.body?.getReader()
        if (!reader) throw new Error('No body')
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') { setStreaming(false); return }
              try {
                const parsed = JSON.parse(data)
                if (parsed.token) setText((prev) => prev + parsed.token)
              } catch {}
            }
          }
        }
        setStreaming(false)
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          setError(err.message)
          setStreaming(false)
        }
      })
  }, [])

  const cancel = useCallback(() => {
    controllerRef.current?.abort()
    setStreaming(false)
  }, [])

  return { text, streaming, error, start, cancel }
}
```

- [ ] **Step 4: Write frontend/src/hooks/usePolling.ts**

```tsx
import { useEffect, useRef } from 'react'

export function usePolling(callback: () => void, intervalMs: number | null) {
  const savedCallback = useRef(callback)
  savedCallback.current = callback

  useEffect(() => {
    if (intervalMs === null) return
    savedCallback.current()
    const id = setInterval(() => savedCallback.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
}
```

- [ ] **Step 5: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/lib/api.ts ../contrarian-dashboard/frontend/src/hooks/useApiData.ts ../contrarian-dashboard/frontend/src/hooks/useSSE.ts ../contrarian-dashboard/frontend/src/hooks/usePolling.ts
git commit -m "feat: add API client, useApiData, useSSE, usePolling hooks"
```

---

## Phase 6: Frontend Panel Components

### Task 15: Implement MarketSnapshot and SentimentGauge panels

**Files:**
- Create: `frontend/src/components/panels/MarketSnapshotPanel.tsx`
- Create: `frontend/src/components/panels/SentimentGaugePanel.tsx`

- [ ] **Step 1: Write types/frontend/src/types/index.ts**

```tsx
// Create the types file: frontend/src/types/index.ts
export interface IndexData {
  price: number
  change_pct: number
  sparkline: number[]
}

export interface IndicesData {
  shanghai: IndexData
  shenzhen: IndexData
  chinext: IndexData
  star50: IndexData
}

export interface SentimentData {
  score: number
  label: string
  components: { capital_flow: number; breadth: number; theme_strength: number }
}

export interface ThemeData {
  name: string
  change_pct?: number
  net_flow?: number
  status?: string
}

export interface CapitalFlowData {
  northbound: number
  institutional: number
  retail_active: number
}

export interface BreadthData {
  advance_decline_ratio: number
  pct_above_ma20: number
  pct_above_ma60: number
  volume_breadth: number
}

export interface StockFactors {
  [horizon: string]: {
    ret: number
    rsi_14: number
    adx: number
    dist_ma20: number
    dist_ma60: number
    atr_pct: number
    bbw: number
  }
}

export interface StockData {
  ts_code: string
  name: string
  live_price: number
  live_change_pct: number
  ohlcv: { date: string; open: number; high: number; low: number; close: number; volume: number; source: string }[]
  factors: StockFactors
  audit: any
  dragon_tiger_presence: any
}
```

- [ ] **Step 2: Write frontend/src/components/panels/MarketSnapshotPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface Props {
  data: {
    shanghai?: { price: number; change_pct: number }
    shenzhen?: { price: number; change_pct: number }
    chinext?: { price: number; change_pct: number }
    star50?: { price: number; change_pct: number }
  } | null
  loading: boolean
}

const INDEX_LABELS: Record<string, string> = {
  shanghai: '上证指数',
  shenzhen: '深证成指',
  chinext: '创业板指',
  star50: '科创50',
}

export default function MarketSnapshotPanel({ data, loading }: Props) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">MARKET SNAPSHOT</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-16 bg-[var(--color-bg-hover)]" />
            ))}
          </div>
        ) : data ? (
          <div className="grid grid-cols-4 gap-4">
            {Object.entries(INDEX_LABELS).map(([key, label]) => {
              const d = (data as any)[key]
              const color = (d?.change_pct ?? 0) >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)'
              return (
                <div key={key} className="text-center">
                  <div className="text-xs text-[var(--color-text-muted)]">{label}</div>
                  <div className="text-xl font-bold mt-1" style={{ color }}>
                    {(d?.change_pct ?? 0) >= 0 ? '+' : ''}{d?.change_pct?.toFixed(2) ?? '—'}%
                  </div>
                  <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                    {d?.price?.toFixed(2) ?? '—'}
                  </div>
                </div>
              )
            })}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Write frontend/src/components/panels/SentimentGaugePanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface Props {
  data: { score: number; label: string; components: Record<string, number> } | null
  loading: boolean
}

const LABEL_COLORS: Record<string, string> = {
  BULLISH: 'var(--color-bullish)',
  'NEUTRAL-BULLISH': 'var(--color-info)',
  NEUTRAL: 'var(--color-warning)',
  'NEUTRAL-BEARISH': 'var(--color-bearish)',
  BEARISH: 'var(--color-bearish)',
}

export default function SentimentGaugePanel({ data, loading }: Props) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">SENTIMENT GAUGE</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-20 bg-[var(--color-bg-hover)]" />
        ) : data ? (
          <div className="flex items-center gap-6">
            <div className="text-center">
              <div
                className="text-4xl font-bold"
                style={{ color: LABEL_COLORS[data.label] || 'var(--color-text-primary)' }}
              >
                {data.score}
              </div>
              <div className="text-xs mt-1" style={{ color: LABEL_COLORS[data.label] || 'var(--color-text-muted)' }}>
                {data.label}
              </div>
            </div>
            <div className="flex-1 space-y-1.5">
              {Object.entries(data.components).map(([key, val]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-xs text-[var(--color-text-muted)] w-20">{key}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-[var(--color-bg-hover)]">
                    <div
                      className="h-1.5 rounded-full bg-[var(--color-info)]"
                      style={{ width: `${val * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-[var(--color-text-muted)]">{(val * 100).toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/types/index.ts ../contrarian-dashboard/frontend/src/components/panels/MarketSnapshotPanel.tsx ../contrarian-dashboard/frontend/src/components/panels/SentimentGaugePanel.tsx
git commit -m "feat: add MarketSnapshot and SentimentGauge panels"
```

---

### Task 16: Implement HotThemes, CapitalFlow, TopMovers, SignalAlerts panels

**Files:**
- Create: `frontend/src/components/panels/HotThemesPanel.tsx`
- Create: `frontend/src/components/panels/CapitalFlowPanel.tsx`
- Create: `frontend/src/components/panels/TopMoversPanel.tsx`
- Create: `frontend/src/components/panels/SignalAlertsPanel.tsx`

- [ ] **Step 1: Write frontend/src/components/panels/HotThemesPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Theme { name: string; change_pct?: number; net_flow?: number; status?: string }

export default function HotThemesPanel({ data, loading }: { data: Theme[] | null; loading: boolean }) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">HOT THEMES</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-8 bg-[var(--color-bg-hover)]" />)}</div>
        ) : data ? (
          <div className="space-y-2">
            {data.map((theme: Theme) => (
              <div key={theme.name} className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-primary)]">{theme.name}</span>
                <div className="flex items-center gap-2">
                  {theme.change_pct != null && (
                    <span style={{ color: theme.change_pct >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' }}>
                      {theme.change_pct >= 0 ? '+' : ''}{theme.change_pct.toFixed(1)}%
                    </span>
                  )}
                  {theme.status && (
                    <Badge variant="outline" className="text-[10px] border-[var(--color-border-subtle)]">
                      {theme.status}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: Write frontend/src/components/panels/CapitalFlowPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowUp, ArrowDown } from 'lucide-react'

interface Props { data: { northbound: number; institutional: number; retail_active: number } | null; loading: boolean }

export default function CapitalFlowPanel({ data, loading }: Props) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">CAPITAL FLOW</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? <Skeleton className="h-20 bg-[var(--color-bg-hover)]" /> : data ? (
          <div className="space-y-2">
            {[
              { label: '北上资金', value: data.northbound },
              { label: '机构', value: data.institutional },
              { label: '游资', value: data.retail_active },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">{item.label}</span>
                <span className="flex items-center gap-1" style={{ color: item.value >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' }}>
                  {item.value >= 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                  {item.value >= 0 ? '+' : ''}{item.value.toFixed(1)}亿
                </span>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Write frontend/src/components/panels/TopMoversPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useNavigate } from 'react-router-dom'

interface Mover { ts_code: string; name: string; composite_score?: number }

export default function TopMoversPanel({ data, loading }: { data: Mover[] | null; loading: boolean }) {
  const navigate = useNavigate()
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">TOP MOVERS</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-6 bg-[var(--color-bg-hover)]" />)}</div>
        ) : data ? (
          <div className="space-y-1">
            {data.slice(0, 10).map((m) => (
              <div
                key={m.ts_code}
                className="flex items-center justify-between text-sm py-1 px-2 rounded hover:bg-[var(--color-bg-hover)] cursor-pointer"
                onClick={() => navigate(`/stock/${m.ts_code}`)}
              >
                <span className="text-[var(--color-text-primary)]">{m.name || m.ts_code}</span>
                <span className="text-[var(--color-text-muted)] text-xs">{m.ts_code}</span>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Write frontend/src/components/panels/SignalAlertsPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useNavigate } from 'react-router-dom'

interface Signal { ts_code: string; name?: string; entry_date?: string; signal?: string }

export default function SignalAlertsPanel({ data, loading }: { data: Signal[] | null; loading: boolean }) {
  const navigate = useNavigate()
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">SIGNAL ALERTS</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-8 bg-[var(--color-bg-hover)]" />)}</div>
        ) : data && data.length > 0 ? (
          <div className="space-y-2">
            {data.slice(0, 6).map((s) => (
              <div
                key={s.ts_code}
                className="flex items-center justify-between text-sm cursor-pointer hover:bg-[var(--color-bg-hover)] rounded px-2 py-1"
                onClick={() => navigate(`/stock/${s.ts_code}`)}
              >
                <div>
                  <span className="text-[var(--color-text-primary)]">{s.name || s.ts_code}</span>
                  <span className="text-[var(--color-text-muted)] text-xs ml-2">{s.ts_code}</span>
                </div>
                <Badge variant="outline" className="text-[10px] border-[var(--color-bullish)] text-[var(--color-bullish)]">
                  {s.signal || s.entry_date || 'signal'}
                </Badge>
              </div>
            ))}
          </div>
        ) : data && data.length === 0 ? (
          <div className="text-sm text-[var(--color-text-muted)]">暂无信号</div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 5: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/components/panels/HotThemesPanel.tsx ../contrarian-dashboard/frontend/src/components/panels/CapitalFlowPanel.tsx ../contrarian-dashboard/frontend/src/components/panels/TopMoversPanel.tsx ../contrarian-dashboard/frontend/src/components/panels/SignalAlertsPanel.tsx
git commit -m "feat: add HotThemes, CapitalFlow, TopMovers, SignalAlerts panels"
```

---

### Task 17: Implement MarketBreadth, DragonTigerFeed panels

**Files:**
- Create: `frontend/src/components/panels/MarketBreadthPanel.tsx`
- Create: `frontend/src/components/panels/DragonTigerFeedPanel.tsx`

- [ ] **Step 1: Write frontend/src/components/panels/MarketBreadthPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface Props {
  data: {
    advance_decline_ratio: number
    pct_above_ma20: number
    pct_above_ma60: number
    volume_breadth: number
  } | null
  loading: boolean
}

export default function MarketBreadthPanel({ data, loading }: Props) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">MARKET BREADTH</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-16 bg-[var(--color-bg-hover)]" />
        ) : data ? (
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: '涨跌比', value: data.advance_decline_ratio.toFixed(2), color: data.advance_decline_ratio > 1 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
              { label: '>MA20', value: `${data.pct_above_ma20}%`, color: data.pct_above_ma20 > 50 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
              { label: '>MA60', value: `${data.pct_above_ma60}%`, color: data.pct_above_ma60 > 50 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
              { label: '量能宽度', value: `${data.volume_breadth}%`, color: data.volume_breadth > 50 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
            ].map((item) => (
              <div key={item.label} className="text-center">
                <div className="text-xs text-[var(--color-text-muted)]">{item.label}</div>
                <div className="text-lg font-bold mt-1" style={{ color: item.color }}>{item.value}</div>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: Write frontend/src/components/panels/DragonTigerFeedPanel.tsx**

```tsx
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

interface Entry {
  ts_code: string
  name: string
  close?: number
  pct_change?: number
  net_amount?: number
  reason?: string
}

export default function DragonTigerFeedPanel({ data, loading }: { data: Entry[] | null; loading: boolean }) {
  return (
    <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-[var(--color-text-muted)]">DRAGON TIGER FEED</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-40 bg-[var(--color-bg-hover)]" />
        ) : data && data.length > 0 ? (
          <ScrollArea className="h-40">
            <div className="space-y-1">
              {data.map((entry, i) => (
                <div key={`${entry.ts_code}-${i}`} className="flex items-center justify-between text-xs py-1 border-b border-[var(--color-border-subtle)] last:border-0">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-[var(--color-text-primary)] font-medium truncate">{entry.name}</span>
                    <span className="text-[var(--color-text-muted)]">{entry.ts_code}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    {entry.pct_change != null && (
                      <span style={{ color: entry.pct_change >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' }}>
                        {entry.pct_change >= 0 ? '+' : ''}{entry.pct_change.toFixed(1)}%
                      </span>
                    )}
                    {entry.net_amount != null && (
                      <span style={{ color: entry.net_amount >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' }}>
                        {(entry.net_amount / 1e8).toFixed(2)}亿
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        ) : (
          <div className="text-sm text-[var(--color-text-muted)]">暂无龙虎榜数据</div>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/components/panels/MarketBreadthPanel.tsx ../contrarian-dashboard/frontend/src/components/panels/DragonTigerFeedPanel.tsx
git commit -m "feat: add MarketBreadth and DragonTigerFeed panels"
```

---

### Task 18: Implement BriefingBanner with SSE streaming

**Files:**
- Create: `frontend/src/components/panels/BriefingBanner.tsx`

- [ ] **Step 1: Write frontend/src/components/panels/BriefingBanner.tsx**

```tsx
import { useEffect, useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useSSE } from '@/hooks/useSSE'

export default function BriefingBanner() {
  const { text, streaming, error, start } = useSSE()
  const [gemmaDown, setGemmaDown] = useState(false)

  useEffect(() => {
    // Check LLM health first, then fetch briefing
    fetch('http://localhost:8000/api/llm/health')
      .then(r => r.json())
      .then(data => {
        if (data.available) {
          start('/llm/briefing', 'POST')
        } else {
          setGemmaDown(true)
        }
      })
      .catch(() => setGemmaDown(true))
  }, [start])

  if (gemmaDown) {
    return (
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-warning)]">
        <CardContent className="py-2 px-4">
          <p className="text-sm text-[var(--color-warning)]">
            Gemma LLM 不可用，AI 注释已暂停 — 检查 DGX Spark 服务状态
          </p>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-bearish)]">
        <CardContent className="py-2 px-4">
          <p className="text-sm text-[var(--color-bearish)]">AI 简报生成失败: {error}</p>
        </CardContent>
      </Card>
    )
  }

  if (!text && streaming) {
    return (
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardContent className="py-3 px-4">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-3/4 bg-[var(--color-bg-hover)]" />
            <span className="text-xs text-[var(--color-text-muted)]">生成中...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (text) {
    return (
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardContent className="py-3 px-4">
          <p className="text-sm text-[var(--color-text-primary)] leading-relaxed">
            {text}
            {streaming && <span className="inline-block w-1.5 h-4 bg-[var(--color-text-muted)] ml-0.5 animate-pulse align-middle" />}
          </p>
        </CardContent>
      </Card>
    )
  }

  return null
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/components/panels/BriefingBanner.tsx
git commit -m "feat: add BriefingBanner with SSE streaming and LLM-down warning"
```

---

## Phase 7: Frontend Pages

### Task 19: Assemble MacroDashboard page

**Files:**
- Create: `frontend/src/pages/MacroDashboard.tsx`

- [ ] **Step 1: Write frontend/src/pages/MacroDashboard.tsx**

```tsx
import DashboardGrid from '@/components/layout/DashboardGrid'
import BriefingBanner from '@/components/panels/BriefingBanner'
import MarketSnapshotPanel from '@/components/panels/MarketSnapshotPanel'
import SentimentGaugePanel from '@/components/panels/SentimentGaugePanel'
import HotThemesPanel from '@/components/panels/HotThemesPanel'
import CapitalFlowPanel from '@/components/panels/CapitalFlowPanel'
import TopMoversPanel from '@/components/panels/TopMoversPanel'
import MarketBreadthPanel from '@/components/panels/MarketBreadthPanel'
import DragonTigerFeedPanel from '@/components/panels/DragonTigerFeedPanel'
import SignalAlertsPanel from '@/components/panels/SignalAlertsPanel'
import { useApiData } from '@/hooks/useApiData'

export default function MacroDashboard() {
  const { data: indices, loading: iLoad } = useApiData('/macro/indices', 60000)
  const { data: sentiment, loading: sLoad } = useApiData('/macro/sentiment', 300000)
  const { data: themes, loading: tLoad } = useApiData('/macro/themes', 300000)
  const { data: capFlow, loading: cLoad } = useApiData('/macro/capital-flow', 300000)
  const { data: movers, loading: mLoad } = useApiData('/macro/top-movers', 300000)
  const { data: breadth, loading: bLoad } = useApiData('/macro/breadth', 300000)
  const { data: dragonTiger, loading: dLoad } = useApiData('/macro/dragon-tiger', 60000)
  const { data: signals, loading: sigLoad } = useApiData('/macro/signal-alerts', 300000)

  return (
    <DashboardGrid
      briefing={<BriefingBanner />}
      topRow={[
        <MarketSnapshotPanel key="snapshot" data={indices} loading={iLoad} />,
        <HotThemesPanel key="themes" data={themes} loading={tLoad} />,
      ]}
      middleRow={[
        <SentimentGaugePanel key="sentiment" data={sentiment} loading={sLoad} />,
        <CapitalFlowPanel key="flow" data={capFlow} loading={cLoad} />,
      ]}
      bottomRow={[
        <TopMoversPanel key="movers" data={movers} loading={mLoad} />,
        <SignalAlertsPanel key="signals" data={signals} loading={sigLoad} />,
      ]}
      breadth={<MarketBreadthPanel data={breadth} loading={bLoad} />}
      feed={<DragonTigerFeedPanel data={dragonTiger} loading={dLoad} />}
    />
  )
}
```

- [ ] **Step 2: Verify the app renders (even if data is missing, panels show loading states)**

Run: `cd /home/robertpeng/contrarian-dashboard/frontend && timeout 10 npm run dev || true`
Expected: Vite starts without TypeScript errors.

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/pages/MacroDashboard.tsx
git commit -m "feat: assemble MacroDashboard with 8 independent-loading panels"
```

---

### Task 20: Implement StockDetail page with K-line chart and LLM commentary

**Files:**
- Create: `frontend/src/pages/StockDetail.tsx`
- Create: `frontend/src/components/charts/KLineChart.tsx`
- Create: `frontend/src/components/charts/HorizonFactorCards.tsx`

- [ ] **Step 1: Write frontend/src/components/charts/KLineChart.tsx**

```tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

interface OHLCV {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export default function KLineChart({ data }: { data: OHLCV[] }) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current || !data.length) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#131b2e' },
        textColor: '#64748b',
      },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      width: containerRef.current.clientWidth,
      height: 400,
      crosshair: { mode: 0 },
      timeScale: { borderColor: '#1e293b', timeVisible: true },
      rightPriceScale: { borderColor: '#1e293b' },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })

    const candleData = data.map((d) => ({
      time: d.date.replace(/-/g, '/'),
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))
    const volumeData = data.map((d) => ({
      time: d.date.replace(/-/g, '/'),
      value: d.volume,
      color: d.close >= d.open ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)',
    }))

    candleSeries.setData(candleData)
    volumeSeries.setData(volumeData)

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  return <div ref={containerRef} className="w-full rounded-md overflow-hidden" />
}
```

- [ ] **Step 2: Write frontend/src/components/charts/HorizonFactorCards.tsx**

```tsx
interface FactorRow {
  ret: number; rsi_14: number; adx: number; dist_ma20: number; dist_ma60: number; atr_pct: number; bbw: number
}

export default function HorizonFactorCards({ factors, horizon }: { factors: Record<string, FactorRow>; horizon: string }) {
  const f = factors[horizon]
  if (!f) return <div className="text-sm text-[var(--color-text-muted)]">无数据</div>

  const items = [
    { label: 'RSI(14)', value: f.rsi_14.toFixed(1), color: f.rsi_14 > 70 ? 'var(--color-bearish)' : f.rsi_14 < 30 ? 'var(--color-bullish)' : 'var(--color-warning)' },
    { label: 'ADX', value: f.adx.toFixed(1), color: f.adx > 25 ? 'var(--color-bullish)' : 'var(--color-text-muted)' },
    { label: 'Ret', value: `${f.ret >= 0 ? '+' : ''}${f.ret.toFixed(1)}%`, color: f.ret >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
    { label: 'Dist→MA20', value: `${f.dist_ma20 >= 0 ? '+' : ''}${f.dist_ma20.toFixed(1)}%`, color: f.dist_ma20 >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' },
    { label: 'ATR%', value: f.atr_pct.toFixed(2), color: 'var(--color-text-primary)' },
    { label: 'BBW', value: f.bbw.toFixed(3), color: 'var(--color-text-primary)' },
  ]

  return (
    <div className="grid grid-cols-6 gap-2">
      {items.map((item) => (
        <div key={item.label} className="text-center bg-[var(--color-bg-card)] rounded-md p-3 border border-[var(--color-border-subtle)]">
          <div className="text-[10px] text-[var(--color-text-muted)] uppercase">{item.label}</div>
          <div className="text-sm font-bold mt-1" style={{ color: item.color }}>{item.value}</div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Write frontend/src/pages/StockDetail.tsx**

```tsx
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import KLineChart from '@/components/charts/KLineChart'
import HorizonFactorCards from '@/components/charts/HorizonFactorCards'
import { useApiData } from '@/hooks/useApiData'
import { useSSE } from '@/hooks/useSSE'
import { useEffect } from 'react'

const HORIZONS = ['5d', '20d', '60d', '120d']

export default function StockDetail() {
  const { code } = useParams<{ code: string }>()
  const [horizon, setHorizon] = useState('20d')
  const { data, loading, error } = useApiData<any>(`/stock/${code}`, 60000)
  const { text: commentary, streaming: llmStreaming, error: llmError, start: startLLM } = useSSE()

  useEffect(() => {
    if (data && code) {
      startLLM(
        `/llm/commentary/${code}`,
        'POST',
        JSON.stringify({
          name: data.name,
          ts_code: code,
          price: data.live_price,
          change_pct: data.live_change_pct,
          factors: data.factors,
        })
      )
    }
  }, [data, code, startLLM])

  if (loading) {
    return <div className="space-y-4">{Array.from({length:3}).map((_,i) => <Skeleton key={i} className="h-40 bg-[var(--color-bg-card)]" />)}</div>
  }

  if (error || !data) {
    return <div className="text-[var(--color-bearish)]">加载失败: {error || '无数据'}</div>
  }

  const color = data.live_change_pct >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)'

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardContent className="py-3 px-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-[var(--color-text-primary)]">
              {code} {data.name && <span className="text-[var(--color-text-muted)]">{data.name}</span>}
            </h1>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold" style={{ color }}>{data.live_price?.toFixed(2) || '—'}</div>
            <div className="text-sm" style={{ color }}>
              {data.live_change_pct >= 0 ? '+' : ''}{data.live_change_pct?.toFixed(2)}%
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Horizon Tabs */}
      <div className="flex gap-2">
        {HORIZONS.map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
              horizon === h
                ? 'bg-[var(--color-bullish)] text-white'
                : 'bg-[var(--color-bg-card)] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'
            }`}
          >
            {h}
          </button>
        ))}
      </div>

      {/* K-line Chart */}
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardContent className="p-0">
          <KLineChart data={data.ohlcv || []} />
        </CardContent>
      </Card>

      {/* Horizon Factors */}
      <HorizonFactorCards factors={data.factors || {}} horizon={horizon} />

      {/* LLM Commentary */}
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-[var(--color-text-muted)]">LLM ANALYSIS</CardTitle>
        </CardHeader>
        <CardContent>
          {llmError && <p className="text-sm text-[var(--color-bearish)]">Gemma 不可用: {llmError}</p>}
          {!commentary && llmStreaming && <Skeleton className="h-12 bg-[var(--color-bg-hover)]" />}
          {commentary && (
            <p className="text-sm text-[var(--color-text-primary)] leading-relaxed">
              {commentary}
              {llmStreaming && <span className="inline-block w-1.5 h-4 bg-[var(--color-text-muted)] ml-0.5 animate-pulse align-middle" />}
            </p>
          )}
          {!commentary && !llmStreaming && !llmError && (
            <p className="text-sm text-[var(--color-text-muted)]">加载AI分析中...</p>
          )}
        </CardContent>
      </Card>

      {/* Audit Summary */}
      {data.audit?.verdict && (
        <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-[var(--color-text-muted)]">AUDIT</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-3">
              <Badge
                style={{
                  background:
                    data.audit.verdict === 'pass' ? 'var(--color-bullish)'
                      : data.audit.verdict === 'warn' ? 'var(--color-warning)'
                      : 'var(--color-bearish)',
                }}
              >
                {data.audit.verdict?.toUpperCase() || '?'}
              </Badge>
              {data.audit.confidence != null && (
                <span className="text-sm text-[var(--color-text-muted)]">
                  Confidence: {(data.audit.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Dragon Tiger */}
      {data.dragon_tiger_presence?.recent_appearances > 0 && (
        <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-[var(--color-text-muted)]">DRAGON TIGER</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-text-primary)]">
              近期待上榜 {data.dragon_tiger_presence.recent_appearances} 次，
              净流入 {data.dragon_tiger_presence.net_flow} 亿
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/components/charts/KLineChart.tsx ../contrarian-dashboard/frontend/src/components/charts/HorizonFactorCards.tsx ../contrarian-dashboard/frontend/src/pages/StockDetail.tsx
git commit -m "feat: add StockDetail page with K-line chart, horizon tabs, LLM commentary"
```

---

### Task 21: Implement ThemeExplorer and StockScreener pages

**Files:**
- Create: `frontend/src/pages/ThemeExplorer.tsx`
- Create: `frontend/src/pages/StockScreener.tsx`

- [ ] **Step 1: Write frontend/src/pages/ThemeExplorer.tsx**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { useApiData } from '@/hooks/useApiData'
import { useNavigate } from 'react-router-dom'

interface Theme {
  name: string
  net_flow?: number
  appearances?: number
  capital_structure?: { northbound?: number; institutional?: number; retail?: number }
  constituents?: string[]
  trend?: string
}

export default function ThemeExplorer() {
  const { data, loading } = useApiData<{ themes: Theme[] }>('/themes')
  const navigate = useNavigate()

  if (loading) return <div className="space-y-3">{Array.from({length:5}).map((_,i)=> <Skeleton key={i} className="h-24 bg-[var(--color-bg-card)]" />)}</div>

  const themes = data?.themes || []

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Theme Explorer</h1>
      <div className="grid grid-cols-2 gap-3">
        {themes.map((theme: Theme) => (
          <Card
            key={theme.name}
            className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-hover)] cursor-pointer transition-colors"
            onClick={() => navigate(`/screener?industry=${encodeURIComponent(theme.name)}`)}
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                {theme.name}
                {theme.trend && (
                  <Badge variant="outline" className="text-[10px] border-[var(--color-border-subtle)]">{theme.trend}</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-2 text-xs">
                {theme.net_flow != null && (
                  <div>
                    <div className="text-[var(--color-text-muted)]">净流入</div>
                    <div style={{ color: theme.net_flow >= 0 ? 'var(--color-bullish)' : 'var(--color-bearish)' }}>
                      {theme.net_flow >= 0 ? '+' : ''}{theme.net_flow.toFixed(1)}亿
                    </div>
                  </div>
                )}
                {theme.appearances != null && (
                  <div>
                    <div className="text-[var(--color-text-muted)]">上榜次数</div>
                    <div className="text-[var(--color-text-primary)]">{theme.appearances}</div>
                  </div>
                )}
                {theme.capital_structure && (
                  <div>
                    <div className="text-[var(--color-text-muted)]">资金结构</div>
                    <div className="text-[var(--color-text-primary)] text-[10px]">
                      北{theme.capital_structure.northbound || 0}% 机{theme.capital_structure.institutional || 0}%
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Write frontend/src/pages/StockScreener.tsx**

```tsx
import { useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { useApiData } from '@/hooks/useApiData'
import { useSSE } from '@/hooks/useSSE'

interface Candidate {
  ts_code: string
  name: string
  industry?: string
  composite_score?: number
}

export default function StockScreener() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const industry = searchParams.get('industry') || ''
  const [nlQuery, setNlQuery] = useState('')
  const [nlResults, setNlResults] = useState<Candidate[] | null>(null)
  const [nlExplanation, setNlExplanation] = useState('')

  const { data, loading } = useApiData<{ candidates: Candidate[] }>(
    industry ? `/screener?industry=${encodeURIComponent(industry)}` : '/screener'
  )

  const handleNlSearch = async () => {
    if (!nlQuery.trim()) return
    try {
      const res = await fetch('http://localhost:8000/api/screener/nl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: nlQuery }),
      })
      const json = await res.json()
      setNlResults(json.candidates || [])
      setNlExplanation(json.explanation || '')
    } catch {}
  }

  const candidates = nlResults || data?.candidates || []

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Stock Screener</h1>

      {/* NL Search */}
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardContent className="py-3 flex gap-2">
          <input
            type="text"
            value={nlQuery}
            onChange={(e) => setNlQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleNlSearch()}
            placeholder="自然语言筛选：低估值小市值科技股放量突破..."
            className="flex-1 bg-[var(--color-bg-primary)] border border-[var(--color-border-subtle)] rounded-md px-3 py-2 text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-info)]"
          />
          <button
            onClick={handleNlSearch}
            className="px-4 py-2 bg-[var(--color-info)] text-white rounded-md text-sm hover:opacity-90"
          >
            搜索
          </button>
        </CardContent>
        {nlExplanation && (
          <div className="px-4 pb-3 text-xs text-[var(--color-text-muted)]">
            解释: {nlExplanation}
          </div>
        )}
      </Card>

      {/* Results Table */}
      <Card className="bg-[var(--color-bg-card)] border-[var(--color-border-subtle)]">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-[var(--color-text-muted)]">
            {candidates.length} 个结果 {industry && `· ${industry}`}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-64 bg-[var(--color-bg-hover)]" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-[var(--color-border-subtle)]">
                  <TableHead className="text-[var(--color-text-muted)]">代码</TableHead>
                  <TableHead className="text-[var(--color-text-muted)]">名称</TableHead>
                  <TableHead className="text-[var(--color-text-muted)]">行业</TableHead>
                  <TableHead className="text-[var(--color-text-muted)]">评分</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {candidates.map((c) => (
                  <TableRow
                    key={c.ts_code}
                    className="border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-hover)] cursor-pointer"
                    onClick={() => navigate(`/stock/${c.ts_code}`)}
                  >
                    <TableCell className="text-[var(--color-text-primary)] font-mono text-xs">{c.ts_code}</TableCell>
                    <TableCell className="text-[var(--color-text-primary)]">{c.name}</TableCell>
                    <TableCell className="text-[var(--color-text-muted)] text-xs">{c.industry || '—'}</TableCell>
                    <TableCell>
                      {c.composite_score != null ? (
                        <Badge variant="outline" className="border-[var(--color-bullish)] text-[var(--color-bullish)]">
                          {(c.composite_score * 100).toFixed(0)}
                        </Badge>
                      ) : '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/frontend/src/pages/ThemeExplorer.tsx ../contrarian-dashboard/frontend/src/pages/StockScreener.tsx
git commit -m "feat: add ThemeExplorer and StockScreener pages with NL search"
```

---

## Phase 8: Integration & Polish

### Task 22: Wire up data symlink and verify end-to-end

**Files:**
- Create: symlink `contrarian-dashboard/data -> trend-agent/data`
- Modify: None

- [ ] **Step 1: Create data symlink**

Run: `ln -s /home/robertpeng/trend-agent/data /home/robertpeng/contrarian-dashboard/data`

- [ ] **Step 2: Start backend and verify all API endpoints**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/backend && uv run uvicorn main:app --port 8000 &
sleep 3
echo "=== Health ===" && curl -s http://localhost:8000/api/health
echo
echo "=== Indices ===" && curl -s http://localhost:8000/api/macro/indices | python -c "import sys,json; d=json.load(sys.stdin); print(f'Shanghai: {d.get(\"shanghai\",{}).get(\"price\",\"N/A\")}')"
echo "=== Sentiment ===" && curl -s http://localhost:8000/api/macro/sentiment
echo
echo "=== Themes ===" && curl -s http://localhost:8000/api/macro/themes | python -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} themes')"
echo "=== Stock ===" && curl -s http://localhost:8000/api/stock/000001.SZ | python -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"name\",\"?\")} price={d.get(\"live_price\")} factors={list(d.get(\"factors\",{}).keys())}')"
kill %1 2>/dev/null
```
Expected: All endpoints return data. Stock endpoint returns OHLCV and factors.

- [ ] **Step 3: Start frontend and verify no build errors**

Run:
```bash
cd /home/robertpeng/contrarian-dashboard/frontend && npx tsc --noEmit 2>&1 | head -30
```
Expected: No TypeScript errors (or only minor ones to fix).

- [ ] **Step 4: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/data
git commit -m "feat: add data symlink and verify end-to-end API"
```

---

### Task 23: Write README and final polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

Write `/home/robertpeng/contrarian-dashboard/README.md`:

```markdown
# Contrarian Agent Dashboard

Web dashboard for the Contrarian Agent A-share research system.

## Quick Start

```bash
# Terminal 1: Backend
cd backend && uv run uvicorn main:app --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open http://localhost:5173 in Chrome.

## Architecture

- **Backend**: FastAPI + DuckDB + yfinance + Gemma 4 (vLLM)
- **Frontend**: React + TypeScript + shadcn/ui + lightweight-charts + Tailwind CSS
- **Data**: Reads Parquet from `../trend-agent/data/` via symlink

## Pages

- `/` — Macro Dashboard (8-panel grid + AI briefing)
- `/themes` — Theme Explorer
- `/screener` — Stock Screener with NL search
- `/stock/:code` — Stock Detail with K-line chart and LLM commentary

## Environment

- Gemma 4 on DGX Spark at `http://192.168.3.46:8000/v1`
- yfinance for intraday quotes (~15min delay)
- Sina/Eastmoney as fallback for missing tickers
```

- [ ] **Step 2: Commit**

```bash
cd /home/robertpeng/trend-agent
git add ../contrarian-dashboard/README.md
git commit -m "docs: add README with quick start and architecture overview"
```

---

## Plan Summary

**Total tasks**: 23  
**Estimated time**: ~2-3 hours (assuming clean tooling and no major debugging)  
**Key deliverables**: Working dashboard with 4 pages, 8-panel macro grid, K-line charts, LLM-powered commentary, NL screener  
**Test strategy**: Manual verification at each stage. Backend: curl endpoints. Frontend: `tsc --noEmit` + browser. LLM: health check + SSE streaming test.
