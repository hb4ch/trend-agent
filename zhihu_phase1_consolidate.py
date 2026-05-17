#!/usr/bin/env python3
"""
Phase 1: Consolidate raw tick data into a single parquet.
Filters to stocks with >= MIN_DAYS, normalizes dates, basic cleaning only.
No window functions — fast bulk read + write.
"""
import os
import duckdb
import pandas as pd
from pathlib import Path

MIN_DAYS = 120
N_THREADS = 8
INPUT_GLOB = "data/stock_ticks/*.parquet"
OUTPUT = Path("data/zhihu_ticks_consolidated.parquet")

print(f"[Phase 1] Consolidating {INPUT_GLOB}...")

con = duckdb.connect(":memory:")
con.execute(f"SET threads TO {N_THREADS}")

# Step A: Count rows per stock to filter
print("  Counting rows per stock...")
counts = con.execute(f"""
    SELECT ts_code, COUNT(*) AS n
    FROM read_parquet('{INPUT_GLOB}', union_by_name=true)
    GROUP BY ts_code
    HAVING COUNT(*) >= {MIN_DAYS}
""").df()
qualified = set(counts["ts_code"])
print(f"  {len(qualified):,} stocks qualify (>= {MIN_DAYS} days)")

# Step B: Load only qualified stocks with date conversion
print("  Loading qualified stocks with date conversion...")
df = con.execute(f"""
    SELECT
        ts_code,
        strptime(trade_date, '%Y%m%d') AS trade_date_d,
        CAST(open AS DOUBLE)   AS open,
        CAST(high AS DOUBLE)   AS high,
        CAST(low AS DOUBLE)    AS low,
        CAST(close AS DOUBLE)  AS close,
        CAST(pre_close AS DOUBLE) AS pre_close,
        CAST(vol AS DOUBLE)    AS vol,
        CAST(amount AS DOUBLE) AS amount,
        CAST(turnover_rate AS DOUBLE) AS turnover_rate
    FROM read_parquet('{INPUT_GLOB}', union_by_name=true)
    WHERE ts_code IN ({','.join(repr(c) for c in sorted(qualified))})
      AND close > 0 AND high > 0 AND low > 0 AND open > 0
    ORDER BY ts_code, trade_date_d
""").df()

con.close()

df.to_parquet(OUTPUT, index=False)
print(f"  Saved {len(df):,} rows × {len(df.columns)} cols to {OUTPUT}")
print(f"  Stocks: {df['ts_code'].nunique():,}")
print(f"  Date range: {df['trade_date_d'].min().date()} to {df['trade_date_d'].max().date()}")
print("[Phase 1] Done.")
