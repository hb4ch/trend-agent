#!/usr/bin/env python3
"""
Look-ahead bias check: recompute BOS, True BOS, Gap Hold with strictly
lagged data and compare ICs against the original inflated versions.
"""
import numpy as np
import pandas as pd

PANEL = "data/zhihu_factor_panel.parquet"
SW = 10  # swing window

print("Loading panel...")
df = pd.read_parquet(PANEL)
df["trade_date_d"] = pd.to_datetime(df["trade_date_d"])
print(f"  {len(df):,} rows, {df['ts_code'].nunique()} stocks")

# Results collector
comparisons = []

for ts, grp in df.groupby("ts_code", sort=False):
    n = len(grp)
    if n < SW * 3:
        continue

    hi = grp["high"].values.astype(float)
    lo = grp["low"].values.astype(float)
    cl = grp["close"].values.astype(float)
    op = grp["open"].values.astype(float)
    pc = grp["pre_close"].values.astype(float)
    vo = grp["vol"].values.astype(float)

    # ── ORIGINAL (inflated) swing highs ──
    swing_hi_orig = np.zeros(n, dtype=bool)
    for i in range(SW, n - SW):
        if hi[i] > max(hi[i-SW:i].max(), hi[i+1:i+SW+1].max()):
            swing_hi_orig[i] = True

    # ── LAGGED (clean) swing highs: only use swing highs confirmed
    #    by day i — i.e., swing_hi_orig at day j is only known at day j+SW.
    #    At day i, we know swing highs at j where j+SW <= i, i.e., j <= i-SW.
    swing_hi_clean = np.zeros(n, dtype=bool)
    for i in range(SW, n - SW):
        # Original detection at day i — only usable from day i+SW onward
        is_swing = hi[i] > max(hi[i-SW:i].max(), hi[i+1:i+SW+1].max())
        if is_swing:
            swing_hi_clean[min(i + SW, n - 1)] = True
    # Propagate: once confirmed, stays known
    for i in range(1, n):
        swing_hi_clean[i] = swing_hi_clean[i] or swing_hi_clean[i-1]

    # ── Rolling last swing high (original, inflated) ──
    last_sh_orig = np.full(n, np.nan)
    cur = np.nan
    for i in range(n):
        if swing_hi_orig[i]:
            cur = hi[i]
        last_sh_orig[i] = cur

    # ── Rolling last swing high (lagged, clean) ──
    last_sh_clean = np.full(n, np.nan)
    cur = np.nan
    for i in range(n):
        if swing_hi_clean[i] and i > 0 and swing_hi_clean[i] > swing_hi_clean[i-1]:
            # This is the confirmation day of a swing high that happened at i-SW
            cur = hi[i - SW]
        last_sh_clean[i] = cur

    # ── BOS ──
    bos_orig = np.where((~np.isnan(last_sh_orig)) & (cl > last_sh_orig), 1.0, 0.0)
    bos_clean = np.where((~np.isnan(last_sh_clean)) & (cl > last_sh_clean), 1.0, 0.0)

    # ── Gap Hold ──
    # Original: gh[i]=1 if gap at day i AND all lows from i to end are above pc[i]
    gh_orig = np.zeros(n)
    for i in range(1, n):
        if (op[i] - pc[i]) / max(pc[i], 0.001) > 0.005:
            end = min(n, i + 11)
            if np.all(lo[i:end] >= pc[i] * 0.999):
                gh_orig[i] = 1.0

    # Clean: only gap days 10+ days ago where the 10-day hold is CONFIRMED
    gh_clean = np.zeros(n)
    for i in range(11, n):
        gap_day = i - 10  # gap was 10 days ago
        if (op[gap_day] - pc[gap_day]) / max(pc[gap_day], 0.001) > 0.005:
            # Check if the 10-day window after the gap held
            gap_end = min(n, gap_day + 11)
            if np.all(lo[gap_day:gap_end] >= pc[gap_day] * 0.999):
                # Gap confirmed held — signal on ALL subsequent days
                gh_clean[i] = 1.0

    # ── True BOS ──
    # Original: tb[j]=1 if BOS at i<j and retest at j
    atr_arr = pd.Series(
        np.maximum(hi - lo, np.maximum(
            np.abs(hi - np.roll(pc, 1)), np.abs(lo - np.roll(pc, 1)))
        )
    ).rolling(14, min_periods=1).mean().values
    atr_arr[0] = hi[0] - lo[0]

    tb_orig = np.zeros(n)
    for i in range(n):
        if bos_orig[i] == 0:
            continue
        sw_p = last_sh_orig[i]
        if np.isnan(sw_p):
            continue
        zb, zt = sw_p - atr_arr[i] * 0.3, sw_p + atr_arr[i] * 0.2
        for j in range(i + 1, min(n, i + 21)):
            if zb <= lo[j] <= zt and cl[j] > zb:
                tb_orig[j] = 1.0
                break

    tb_clean = np.zeros(n)
    for i in range(n):
        if bos_clean[i] == 0:
            continue
        sw_p = last_sh_clean[i]
        if np.isnan(sw_p):
            continue
        zb, zt = sw_p - atr_arr[i] * 0.3, sw_p + atr_arr[i] * 0.2
        for j in range(i + 1, min(n, i + 21)):
            if zb <= lo[j] <= zt and cl[j] > zb:
                tb_clean[j] = 1.0
                break

    # ── Store per-row comparison ──
    for i in range(SW * 3, n):
        for fwd_tag, col in [("1d", "fwd_ret_1d"), ("3d", "fwd_ret_3d"),
                              ("5d", "fwd_ret_5d"), ("10d", "fwd_ret_10d"),
                              ("20d", "fwd_ret_20d")]:
            fwd = grp[col].iloc[i]
            if pd.isna(fwd):
                continue
            comparisons.append({
                "ts_code": ts,
                "trade_date_d": grp["trade_date_d"].iloc[i],
                "horizon": fwd_tag,
                "fwd_ret": fwd,
                "bos_orig": bos_orig[i],
                "bos_clean": bos_clean[i],
                "gh_orig": gh_orig[i],
                "gh_clean": gh_clean[i],
                "tb_orig": tb_orig[i],
                "tb_clean": tb_clean[i],
            })

comp = pd.DataFrame(comparisons)
print(f"  Comparison rows: {len(comp):,}")

# ── Compute ICs ──
print("\n=== IC Comparison: Original vs Clean (Look-Ahead Fixed) ===\n")
print(f"{'Signal':<20} {'Hrz':>5} {'Orig IC':>10} {'Orig T':>8} {'Clean IC':>10} {'Clean T':>8} {'IC Delta':>10}")
print("-" * 75)

for signal_name, orig_col, clean_col in [
    ("BOS", "bos_orig", "bos_clean"),
    ("Gap Hold", "gh_orig", "gh_clean"),
    ("True BOS", "tb_orig", "tb_clean"),
]:
    for fwd_tag in ["1d", "3d", "5d", "10d", "20d"]:
        horizon_num = int(fwd_tag.replace("d", ""))
        sub = comp[comp["horizon"] == fwd_tag].dropna(subset=[orig_col, clean_col, "fwd_ret"])
        if len(sub) < 1000:
            continue

        dates = sorted(sub["trade_date_d"].unique())
        ic_orig_list, ic_clean_list = [], []

        for d in dates:
            cross = sub[sub["trade_date_d"] == d]
            if len(cross) < 30:
                continue
            xo = cross[orig_col].astype(float)
            xc = cross[clean_col].astype(float)
            y = cross["fwd_ret"].astype(float)

            if xo.std() < 1e-12 and xc.std() < 1e-12:
                continue

            mask = np.isfinite(y)
            if mask.sum() < 30:
                continue

            if xo.std() > 1e-12:
                ro = xo[mask].rank().corr(y[mask].rank())
                if np.isfinite(ro):
                    ic_orig_list.append(ro)
            if xc.std() > 1e-12:
                rc = xc[mask].rank().corr(y[mask].rank())
                if np.isfinite(rc):
                    ic_clean_list.append(rc)

        if len(ic_orig_list) < 20 and len(ic_clean_list) < 20:
            continue

        mo = float(np.mean(ic_orig_list)) if ic_orig_list else np.nan
        mc = float(np.mean(ic_clean_list)) if ic_clean_list else np.nan
        so = float(np.std(ic_orig_list, ddof=1)) if ic_orig_list else 1.0
        sc = float(np.std(ic_clean_list, ddof=1)) if ic_clean_list else 1.0
        to_val = mo / (so / np.sqrt(len(ic_orig_list))) if so > 0 and ic_orig_list else np.nan
        tc_val = mc / (sc / np.sqrt(len(ic_clean_list))) if sc > 0 and ic_clean_list else np.nan

        delta = mo - mc if (not np.isnan(mo) and not np.isnan(mc)) else np.nan

        print(f"{signal_name:<20} {fwd_tag:>5} {mo:>10.4f} {to_val:>8.1f} {mc:>10.4f} {tc_val:>8.1f} {delta:>10.4f}")

print("\n=== Trigger Rate Comparison ===\n")
for signal_name, orig_col, clean_col in [
    ("BOS", "bos_orig", "bos_clean"),
    ("Gap Hold", "gh_orig", "gh_clean"),
    ("True BOS", "tb_orig", "tb_clean"),
]:
    orig_rate = comp[orig_col].mean()
    clean_rate = comp[clean_col].mean()
    print(f"  {signal_name:20s}: orig={orig_rate:.2%}  clean={clean_rate:.2%}")

print("\nDone.")
