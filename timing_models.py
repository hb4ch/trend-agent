from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _swing_highs(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Boolean mask: True where high is the max of `window` bars to each side."""
    high = df["high"].astype(float)
    mask = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        left = high.iloc[i - window : i]
        right = high.iloc[i + 1 : i + window + 1]
        if high.iloc[i] > left.max() and high.iloc[i] > right.max():
            mask.iloc[i] = True
    return mask


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _volume_profile_poc(df: pd.DataFrame, lookback: int = 60, bins: int = 30) -> float | None:
    subset = df.tail(lookback)
    if len(subset) < 20:
        return None
    price = subset["close"].astype(float)
    vol = subset.get("vol", subset.get("volume", pd.Series(0, index=subset.index)))
    vol = pd.to_numeric(vol, errors="coerce").fillna(0)
    if vol.sum() == 0:
        return None
    price_min, price_max = float(price.min()), float(price.max())
    if price_max <= price_min:
        return float(price_min)
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)
    for _, row_data in subset.iterrows():
        p = float(row_data["close"])
        v = float(vol.loc[row_data.name]) if row_data.name in vol.index else 0.0
        idx = int(min(bins - 1, max(0, (p - price_min) / (price_max - price_min) * bins - 1e-9)))
        vol_profile[idx] += v
    best = int(np.argmax(vol_profile))
    return float((bin_edges[best] + bin_edges[best + 1]) / 2.0)


def detect_bos(df: pd.DataFrame, swing_window: int = 10) -> bool:
    """Break of Structure: close above the most recent swing high."""
    if len(df) < swing_window * 2 + 1:
        return False
    swings = _swing_highs(df, window=swing_window)
    swing_indices = swings[swings].index
    if len(swing_indices) == 0:
        return False
    last_swing_loc = int(swing_indices[-1])
    if last_swing_loc >= len(df) - 1:
        return False
    swing_price = float(df["high"].iloc[last_swing_loc])
    last_close = float(df["close"].iloc[-1])
    return last_close > swing_price


def detect_true_bos(df: pd.DataFrame, swing_window: int = 10) -> bool:
    """True BOS: BOS occurred, then price retested the broken level and held."""
    if len(df) < swing_window * 2 + 1:
        return False
    swings = _swing_highs(df, window=swing_window)
    swing_indices = swings[swings].index
    if len(swing_indices) < 2:
        return False
    prev_swing_loc = int(swing_indices[-2])
    prev_swing_price = float(df["high"].iloc[prev_swing_loc])
    close = df["close"].astype(float)
    after_swing = close.iloc[prev_swing_loc + 1 :]
    if after_swing.empty:
        return False
    broke = bool((after_swing > prev_swing_price).any())
    if not broke:
        return False
    after_break = df.iloc[prev_swing_loc + 1 :].copy()
    break_idx = int(after_break["close"].gt(prev_swing_price).idxmax())
    atr_val = _atr(df).iloc[-1]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return False
    zone_bottom = prev_swing_price - atr_val * 0.3
    zone_top = prev_swing_price + atr_val * 0.2
    retest = df.iloc[break_idx + 1 :] if break_idx + 1 < len(df) else df.iloc[break_idx:break_idx]
    if retest.empty:
        return False
    retest_lows = retest["low"].astype(float)
    touched_zone = bool(((retest_lows >= zone_bottom) & (retest_lows <= zone_top)).any())
    last_close_val = float(close.iloc[-1])
    held = last_close_val > zone_bottom
    return touched_zone and held


def detect_joc(df: pd.DataFrame, swing_window: int = 10) -> bool:
    """Jump Over Creek: volume surge through a swing high resistance."""
    if len(df) < swing_window * 2 + 1:
        return False
    swings = _swing_highs(df, window=swing_window)
    swing_indices = swings[swings].index
    if len(swing_indices) == 0:
        return False
    last_swing_loc = int(swing_indices[-1])
    if last_swing_loc >= len(df) - 1:
        return False
    swing_price = float(df["high"].iloc[last_swing_loc])
    bos = detect_bos(df, swing_window)
    if not bos:
        return False
    vol_col = "vol" if "vol" in df.columns else "volume"
    vol = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    if avg_vol <= 0:
        return False
    last_vol = float(vol.iloc[-1])
    return last_vol > avg_vol * 1.5


def detect_poc(df: pd.DataFrame, lookback: int = 60) -> bool:
    """Point of Control: close is near the volume-weighted fair value level."""
    poc = _volume_profile_poc(df, lookback=lookback)
    if poc is None:
        return False
    atr_val = float(_atr(df).iloc[-1])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return False
    last_close = float(df["close"].iloc[-1])
    return abs(last_close - poc) <= atr_val


def detect_poc_retest(df: pd.DataFrame, lookback: int = 60) -> bool:
    """POC Retest: price was above POC, now revisits with contracting volume."""
    poc = _volume_profile_poc(df, lookback=lookback)
    if poc is None:
        return False
    atr_val = float(_atr(df).iloc[-1])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return False
    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])
    if abs(last_close - poc) > atr_val:
        return False
    if len(close) < 6:
        return False
    above_poc = close.iloc[-6:-1] > (poc + atr_val * 0.5)
    if not above_poc.any():
        return False
    vol_col = "vol" if "vol" in df.columns else "volume"
    vol = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    if len(vol) < 20:
        return False
    last_vol = float(vol.iloc[-1])
    avg_vol = float(vol.tail(20).mean())
    return last_vol < avg_vol


def detect_gap_hold(df: pd.DataFrame, min_gap_pct: float = 0.005, min_confirmation_days: int = 5) -> bool:
    """Gap up that held: gap day ≥ min_confirmation_days ago, all subsequent lows stayed above prior close."""
    if len(df) < min_confirmation_days + 2:
        return False
    close = df["close"].astype(float)
    open_p = df["open"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)
    gap_up = (open_p - prev_close) / (prev_close + 1e-9) > min_gap_pct
    if not gap_up.any():
        return False
    last_idx = len(df) - 1
    gap_days = gap_up[gap_up].index
    for gap_day in gap_days:
        gap_loc = int(df.index.get_loc(gap_day))
        days_since = last_idx - gap_loc
        if days_since < min_confirmation_days:
            continue
        gap_prev_close = float(prev_close.iloc[gap_loc])
        subsequent_lows = low.iloc[gap_loc:]
        if (subsequent_lows >= gap_prev_close * 0.999).all():
            return True
    return False


def _bos_recently(df: pd.DataFrame, lookback: int = 20, swing_window: int = 10) -> bool:
    """Check if BOS triggered on any day within the last `lookback` bars."""
    if len(df) < swing_window * 2 + 1:
        return False
    swings = _swing_highs(df, window=swing_window)
    swing_indices = swings[swings].index
    if len(swing_indices) == 0:
        return False
    close = df["close"].astype(float)
    last_idx = len(df) - 1
    for i in range(max(0, last_idx - lookback), last_idx + 1):
        sh_idx = int(swing_indices[swing_indices < i][-1]) if any(swing_indices < i) else None
        if sh_idx is None:
            continue
        if sh_idx >= i:
            continue
        if float(close.iloc[i]) > float(df["high"].iloc[sh_idx]):
            return True
    return False


def detect_lps(df: pd.DataFrame, swing_window: int = 10) -> bool:
    """Last Point of Support: pullback to MA20 in an uptrend with contracting volume.

    Requires: (1) MA20 trending up, (2) close near MA20, (3) BOS triggered
    in the past 20 days (uptrend confirmed), (4) volume below average.
    """
    if len(df) < 60:
        return False
    close = df["close"].astype(float)
    ma20 = close.rolling(20).mean()
    atr_val = float(_atr(df).iloc[-1])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return False
    # MA20 trending up
    if ma20.iloc[-1] <= ma20.iloc[-6]:
        return False
    # Close within 1 ATR of MA20 (the pullback zone)
    if abs(float(close.iloc[-1]) - float(ma20.iloc[-1])) > atr_val:
        return False
    # Uptrend confirmed by recent BOS
    if not _bos_recently(df, lookback=20, swing_window=swing_window):
        return False
    # Volume contracting (below 20-day average)
    vol_col = "vol" if "vol" in df.columns else "volume"
    vol = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    if len(vol) < 20:
        return False
    last_vol = float(vol.iloc[-1])
    avg_vol = float(vol.tail(20).mean())
    return last_vol < avg_vol


def compute_timing_signals(df: pd.DataFrame) -> Dict[str, object]:
    """Run all 7 timing detectors and return triggered models + aggregate score.

    Note: IC analysis showed these models have no standalone predictive power
    in the A-share market (mean reversion dominates). They are ENTRY FILTERS
    meant to be used on top of fundamentally-screened stocks, not alpha factors.
    Equal weighting reflects this — we have no statistical basis to weight
    one pattern above another.
    """
    bos = detect_bos(df)
    true_bos = detect_true_bos(df) if bos else False
    joc = detect_joc(df)
    poc = detect_poc(df)
    poc_retest = detect_poc_retest(df)
    gap_hold = detect_gap_hold(df)
    lps = detect_lps(df)
    triggered = sum([bos, true_bos, joc, poc, poc_retest, gap_hold, lps])
    return {
        "timing_bos": bos,
        "timing_true_bos": true_bos,
        "timing_joc": joc,
        "timing_poc": poc,
        "timing_poc_retest": poc_retest,
        "timing_gap_hold": gap_hold,
        "timing_lps": lps,
        "timing_score": triggered / 7.0,
    }
