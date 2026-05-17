#!/usr/bin/env python3
"""
A股潜力成长组合筛选脚本
基于"基本面成长+资金面博弈"双重分析体系
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

from math import exp

from utils import EPSILON

# Setup logging
logger = logging.getLogger(__name__)

# ============ 筛选参数配置 ============
# Relaxed parameters to allow more candidates through to research phase
MARKET_CAP_MIN = 10e8   # 10亿市值 (relaxed from 20e8 to allow smaller growth stocks)
MARKET_CAP_MAX = 500e8  # 500亿市值 (relaxed from 300e8 to include more mature growth)
MIN_DATA_DAYS = 180     # 最少交易日数据 (relaxed from 250 to allow newer IPOs)
MA_PERIODS = [60, 120, 250]  # 均线周期
MA_RECENT_DAYS = 50     # 均线观察窗口
CONSOLIDATION_DAYS = 120  # 横盘观察天数
VOLATILITY_THRESHOLD = 0.50  # 横盘波动幅度阈值（50%）(relaxed from 0.35 to allow more dynamic stocks)

# 排除条件
EXCLUDE_ST = False       # 排除ST股票

# 估值参数
VALUATION_MODE = "blend"
VALUATION_PEER_MIN_SAMPLES = 5
VALUATION_OUTLIER_PERCENTILE = 0.97
VALUATION_ABSOLUTE_CAPS = {
    "pe": 250.0,
    "pb": 25.0,
    "ps_ttm": 40.0,
}
VALUATION_METRICS = ("pe", "pb", "ps_ttm")
VALUATION_WEIGHTS = {
    "pe": 0.5,
    "pb": 0.3,
    "ps_ttm": 0.2,
}

OBV_DIVERGENCE_MIN_NORM = 0.05
OBV_DIVERGENCE_MAX_PRICE_CHANGE = 0.03


def load_stock_basic() -> pd.DataFrame:
    """Load stock basic information."""
    logger.info("Loading stock basic info...")
    df = pd.read_parquet('data/stock_basic/stock_basic.parquet')
    return df


def load_stock_company() -> pd.DataFrame:
    """Load company information."""
    logger.info("Loading stock company info...")
    try:
        df = pd.read_parquet('data/stock_company/stock_company.parquet')
        return df
    except (OSError, IOError) as e:
        logger.warning(f"Could not load company info: {e}")
        return pd.DataFrame()


# ============ Technical Indicator Functions ============

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14):
    """
    Average Directional Index.

    Returns:
        (adx, plus_di, minus_di) as pd.Series
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + EPSILON))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + EPSILON))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + EPSILON))
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


def compute_bollinger_width(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger Band Width = (upper - lower) / middle."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return (upper - lower) / (middle + EPSILON)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    vol_col = "volume" if "volume" in df.columns else "vol"
    return (direction * df[vol_col]).cumsum()


def compute_obv_features(df: pd.DataFrame, window: int = 20) -> Dict[str, float | bool]:
    """Return raw and volume-normalized OBV accumulation features."""
    vol_col = "volume" if "volume" in df.columns else "vol"
    obv_series = compute_obv(df)
    obv_recent = obv_series.tail(window).dropna()
    if len(obv_recent) >= 2:
        obv_slope = float(np.polyfit(range(len(obv_recent)), obv_recent.values, 1)[0])
    else:
        obv_slope = 0.0

    avg_volume = float(df[vol_col].tail(window).mean()) if vol_col in df.columns else 0.0
    obv_slope_norm = obv_slope / (avg_volume + EPSILON) if avg_volume > EPSILON else 0.0

    if len(df) > window:
        base_close = float(df["close"].iloc[-(window + 1)])
        price_change = (float(df["close"].iloc[-1]) / (base_close + EPSILON)) - 1.0 if base_close > EPSILON else 0.0
    else:
        price_change = 0.0

    obv_divergence = obv_slope_norm >= OBV_DIVERGENCE_MIN_NORM and price_change <= OBV_DIVERGENCE_MAX_PRICE_CHANGE
    return {
        "obv_slope": obv_slope,
        "obv_slope_norm": float(obv_slope_norm),
        "price_change_20d": float(price_change),
        "obv_divergence": bool(obv_divergence),
    }


def compute_trend_emergence_score(
    return_5d: float,
    return_10d: float,
    return_20d: float,
    volume_ratio_5d_vs_60d: float,
    fresh_breakout: bool,
    near_breakout: bool,
    obv_accumulation_score: float = 0.0,
) -> float:
    """Score early trend emergence from returns, volume, breakout proximity, and OBV."""
    r5 = finite_float(return_5d)
    r10 = finite_float(return_10d)
    r20 = finite_float(return_20d)
    vol_ratio = finite_float(volume_ratio_5d_vs_60d, 1.0)

    if r20 > 0.45:
        return_score = min(8.0, linear_score(r20, 0.0, 0.45, 25.0))
    elif r20 <= 0.25:
        return_score = linear_score(r20, 0.0, 0.05, 25.0)
    else:
        return_score = max(0.0, 25.0 * (1.0 - (r20 - 0.25) / 0.20))

    acceleration_score = 0.0
    if r5 > 0:
        acceleration_score += linear_score(r5, 0.0, 0.08, 10.0)
    if r10 > 0:
        acceleration_score += linear_score(r10, 0.0, 0.15, 6.0)
    if r5 > r20 / 4.0:
        acceleration_score += 4.0

    volume_score = linear_score(vol_ratio, 1.0, 2.5, 20.0)
    if vol_ratio > 4.0:
        volume_score = min(volume_score, 12.0)

    breakout_score = 20.0 if fresh_breakout else (12.0 if near_breakout else 0.0)
    obv_score = linear_score(finite_float(obv_accumulation_score), 0.0, 100.0, 15.0)
    return min(100.0, return_score + acceleration_score + volume_score + breakout_score + obv_score)


def compute_trend_emergence_features(
    df: pd.DataFrame,
    obv_accumulation_score: float = 0.0,
) -> Dict[str, float | bool]:
    """Return simple features for stocks that are starting to trend now."""
    if df is None or df.empty:
        return {
            "return_5d": 0.0,
            "return_10d": 0.0,
            "return_20d": 0.0,
            "volume_ratio_5d_vs_60d": 1.0,
            "fresh_breakout": False,
            "near_breakout": False,
            "trend_emergence_score": 0.0,
        }

    data = df.sort_values("trade_date").reset_index(drop=True) if "trade_date" in df.columns else df.reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")
    latest_close = finite_float(close.iloc[-1])

    def period_return(days: int) -> float:
        if len(close) <= days:
            return 0.0
        base = finite_float(close.iloc[-(days + 1)])
        if base <= EPSILON or latest_close <= EPSILON:
            return 0.0
        return float(latest_close / (base + EPSILON) - 1.0)

    return_5d = period_return(5)
    return_10d = period_return(10)
    return_20d = period_return(20)

    vol_col = "volume" if "volume" in data.columns else "vol"
    if vol_col in data.columns and len(data) >= 60:
        volume = pd.to_numeric(data[vol_col], errors="coerce")
        avg_5 = finite_float(volume.tail(5).mean())
        avg_60 = finite_float(volume.tail(60).mean())
        volume_ratio = avg_5 / (avg_60 + EPSILON) if avg_60 > EPSILON else 1.0
    else:
        volume_ratio = 1.0

    fresh_breakout = False
    near_breakout = False
    if len(data) >= 61 and "high" in data.columns:
        prior_60d_high = finite_float(pd.to_numeric(data["high"], errors="coerce").iloc[-61:-1].max())
        if prior_60d_high > EPSILON:
            fresh_breakout = latest_close >= prior_60d_high
            near_breakout = latest_close >= 0.95 * prior_60d_high

    trend_score = compute_trend_emergence_score(
        return_5d,
        return_10d,
        return_20d,
        volume_ratio,
        fresh_breakout,
        near_breakout,
        obv_accumulation_score,
    )

    return {
        "return_5d": float(return_5d),
        "return_10d": float(return_10d),
        "return_20d": float(return_20d),
        "volume_ratio_5d_vs_60d": float(volume_ratio),
        "fresh_breakout": bool(fresh_breakout),
        "near_breakout": bool(near_breakout),
        "trend_emergence_score": float(trend_score),
    }


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPSILON)
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP over `period` days."""
    vol_col = "volume" if "volume" in df.columns else "vol"
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df[vol_col]
    return tp_vol.rolling(period).sum() / (df[vol_col].rolling(period).sum() + EPSILON)


# ============ Continuous Scoring Utilities ============

def gaussian_score(value: float, optimal: float, width: float, max_points: float) -> float:
    """Bell-curve scoring centered on optimal value."""
    return max_points * exp(-0.5 * ((value - optimal) / (width + EPSILON)) ** 2)


def linear_score(value: float, min_val: float, max_val: float, max_points: float, invert: bool = False) -> float:
    """Linear ramp between min and max, optionally inverted."""
    if max_val <= min_val:
        return 0.0
    t = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    if invert:
        t = 1.0 - t
    return max_points * t


def range_score(value: float, lo: float, hi: float, max_points: float) -> float:
    """Full points inside [lo, hi], linear taper outside."""
    if lo <= value <= hi:
        return max_points
    if value < lo:
        return max(0.0, max_points * (1.0 - (lo - value) / (hi - lo + EPSILON)))
    return max(0.0, max_points * (1.0 - (value - hi) / (hi - lo + EPSILON)))


def finite_float(value, default: float = 0.0) -> float:
    """Return a finite float, otherwise a default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def compute_volume_quality_score(
    volume_boost: float,
    avg_turnover: float,
    obv_accumulation_score: float = 0.0,
) -> float:
    """Score volume quality with calibrated OBV accumulation instead of raw OBV sign."""
    vq_obv = linear_score(finite_float(obv_accumulation_score), 0.0, 100.0, 40.0)
    vq_boost = range_score(finite_float(volume_boost, 1.0), 1.2, 3.0, 35.0)
    vq_turnover = range_score(finite_float(avg_turnover), 1.0, 5.0, 25.0)
    return min(100.0, vq_obv + vq_boost + vq_turnover)


def safe_positive_float(value) -> Optional[float]:
    """Return a positive finite float, otherwise None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0:
        return None
    return out


def _metric_percentile(valid_values: pd.Series, value: Optional[float]) -> float:
    """Return percentile rank in [0, 1] for a positive metric value."""
    if value is None:
        return np.nan
    sample = pd.Series(valid_values).dropna()
    if sample.empty:
        return np.nan
    return float((sample <= value).mean())


def classify_valuation_label(stretch_score: Optional[float]) -> str:
    """Map valuation stretch score to a readable label."""
    if stretch_score is None or pd.isna(stretch_score):
        return "估值待补充"
    if stretch_score >= 85:
        return "显著高估"
    if stretch_score >= 65:
        return "偏贵"
    if stretch_score >= 40:
        return "适中溢价"
    return "合理"


def compute_industry_relative_valuation(
    df: pd.DataFrame,
    *,
    outlier_percentile: float = VALUATION_OUTLIER_PERCENTILE,
    peer_min_samples: int = VALUATION_PEER_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Add industry-relative valuation features.

    Higher stretch score means more expensive vs peers.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    industry_col = out["industry"].fillna("Unknown").astype(str) if "industry" in out.columns else pd.Series(["Unknown"] * len(out), index=out.index)
    out["industry"] = industry_col

    valid_metric_values: Dict[str, pd.Series] = {}
    global_metric_values: Dict[str, pd.Series] = {}
    for metric in VALUATION_METRICS:
        if metric not in out.columns:
            out[metric] = np.nan
        valid_col = out[metric].apply(safe_positive_float)
        out[f"{metric}_valid"] = valid_col
        valid_metric_values[metric] = valid_col
        global_metric_values[metric] = valid_col.dropna()
        out[f"{metric}_percentile_industry"] = np.nan
        out[f"{metric}_baseline_source"] = "none"

    out["valuation_data_points"] = 0

    industry_groups = out.groupby("industry", dropna=False).groups
    for metric in VALUATION_METRICS:
        valid_col = valid_metric_values[metric]
        for industry, idx in industry_groups.items():
            idx_list = list(idx)
            industry_values = valid_col.loc[idx_list].dropna()
            baseline = industry_values if len(industry_values) >= peer_min_samples else global_metric_values[metric]
            baseline_source = "industry" if len(industry_values) >= peer_min_samples else ("market" if not global_metric_values[metric].empty else "none")
            if baseline_source == "none":
                continue
            for row_idx in idx_list:
                pct = _metric_percentile(baseline, valid_col.loc[row_idx])
                out.at[row_idx, f"{metric}_percentile_industry"] = pct
                out.at[row_idx, f"{metric}_baseline_source"] = baseline_source

    percentile_cols: List[str] = []
    for metric in VALUATION_METRICS:
        pct_col = f"{metric}_percentile_industry"
        percentile_cols.append(pct_col)
        out["valuation_data_points"] += out[pct_col].notna().astype(int)

    weight_sum = sum(VALUATION_WEIGHTS.values())
    weighted = []
    for metric in VALUATION_METRICS:
        pct_col = f"{metric}_percentile_industry"
        weighted.append(out[pct_col].fillna(0.5) * VALUATION_WEIGHTS[metric])
    out["valuation_stretch_score"] = (sum(weighted) / max(weight_sum, EPSILON)) * 100.0
    out.loc[out["valuation_data_points"] == 0, "valuation_stretch_score"] = 50.0
    out["valuation_quality_score"] = 100.0 - out["valuation_stretch_score"]
    out["valuation_label"] = out["valuation_stretch_score"].apply(classify_valuation_label)

    absolute_outlier = pd.Series(False, index=out.index)
    for metric, cap in VALUATION_ABSOLUTE_CAPS.items():
        valid_col = out[f"{metric}_valid"] if f"{metric}_valid" in out.columns else pd.Series(np.nan, index=out.index)
        absolute_outlier = absolute_outlier | (valid_col > cap).fillna(False)

    percentile_outlier = out["valuation_stretch_score"] >= float(outlier_percentile) * 100.0
    unsupported = (out["valuation_data_points"] == 0) & out["pe"].apply(lambda x: safe_positive_float(x) is None)
    out["valuation_outlier"] = (absolute_outlier | percentile_outlier | unsupported).astype(bool)
    out["valuation_has_peer_context"] = out["valuation_data_points"] > 0
    return out.drop(columns=[f"{metric}_valid" for metric in VALUATION_METRICS], errors="ignore")


def analyze_stock_technical(df, current_date=None):
    """
    分析单只股票的技术形态

    返回指标:
    - consolidation_score: 横盘整理得分（0-100）
    - volatility: 波动率
    - volume_trend: 量能趋势
    - ma_alignment: 均线粘合度
    - recent_strength: 近期强度
    """
    if len(df) < MIN_DATA_DAYS:
        return None

    df = df.sort_values('trade_date').reset_index(drop=True)
    latest = df.iloc[-1]

    # 1. 市值过滤（单位：元，需要转换）
    market_cap = latest['total_mv']  # 已经是万元
    market_cap_yuan = market_cap * 10000  # 转换为元

    if not (MARKET_CAP_MIN <= market_cap_yuan <= MARKET_CAP_MAX):
        return None

    # 2. 计算技术指标
    recent = df.tail(CONSOLIDATION_DAYS)

    # 波动率/振幅分析（120日箱体振幅）
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()
    recent_volatility = (recent_high - recent_low) / (recent_low + EPSILON)

    # 均线系统（MA60/120/250 纠缠度：最近MA_RECENT_DAYS天的平均粘合程度）
    for period in MA_PERIODS:
        df[f"ma{period}"] = df["close"].rolling(period).mean()

    ma_cols = [f"ma{p}" for p in MA_PERIODS]
    ma_recent = df[ma_cols].tail(MA_RECENT_DAYS).dropna()
    if ma_recent.empty:
        return None
    daily_spread = (ma_recent.max(axis=1) - ma_recent.min(axis=1)) / (ma_recent.min(axis=1) + EPSILON)
    ma_spread = float(daily_spread.mean())
    ma_spread_std = float(daily_spread.std(ddof=0)) if len(daily_spread) > 1 else 0.0
    ma60 = float(df["ma60"].iloc[-1])
    ma120 = float(df["ma120"].iloc[-1])
    ma250 = float(df["ma250"].iloc[-1])

    # 量能分析
    avg_turnover = recent['turnover_rate'].mean()
    recent_turnover = df['turnover_rate'].tail(10).mean()
    volume_boost = recent_turnover / (avg_turnover + EPSILON)

    # 价格位置
    price_position = (latest['close'] - recent_low) / (recent_high - recent_low + EPSILON)

    # 3. Compute new technical indicators
    ema20 = float(compute_ema(df["close"], 20).iloc[-1]) if len(df) >= 20 else float(latest['close'])

    atr_series = compute_atr(df)
    atr_now = float(atr_series.iloc[-1]) if atr_series.notna().any() else None
    atr_60d_ago = float(atr_series.iloc[-60]) if len(atr_series) >= 60 and atr_series.iloc[-60] == atr_series.iloc[-60] else None
    atr_ratio = (atr_now / (atr_60d_ago + EPSILON)) if (atr_now is not None and atr_60d_ago is not None and atr_60d_ago > EPSILON) else 1.0

    adx_series, _, _ = compute_adx(df)
    adx_now = float(adx_series.iloc[-1]) if adx_series.notna().any() else 20.0
    adx_5d_ago = float(adx_series.iloc[-5]) if len(adx_series) >= 5 and adx_series.iloc[-5] == adx_series.iloc[-5] else adx_now
    adx_slope = adx_now - adx_5d_ago

    bbw_series = compute_bollinger_width(df)
    bbw_now = float(bbw_series.iloc[-1]) if bbw_series.notna().any() else None
    bbw_120 = bbw_series.tail(120).dropna()
    if len(bbw_120) > 1 and bbw_now is not None:
        bbw_min = float(bbw_120.min())
        bbw_max = float(bbw_120.max())
        bbw_percentile = (bbw_now - bbw_min) / (bbw_max - bbw_min + EPSILON)
    else:
        bbw_percentile = 0.5

    obv_features = compute_obv_features(df, window=20)
    obv_slope = float(obv_features["obv_slope"])
    obv_slope_norm = float(obv_features["obv_slope_norm"])
    price_change_20d = float(obv_features["price_change_20d"])
    obv_divergence = bool(obv_features["obv_divergence"])
    trend_features = compute_trend_emergence_features(df, obv_accumulation_score=0.0)

    rsi_series = compute_rsi(df)
    rsi_now = float(rsi_series.iloc[-1]) if rsi_series.notna().any() else 50.0

    vwap_series = compute_vwap(df, period=20)
    vwap_now = float(vwap_series.iloc[-1]) if vwap_series.notna().any() else float(latest['close'])
    vwap_ratio = float(latest['close']) / (vwap_now + EPSILON)

    # 3b. Build scores dict
    scores = {
        'volatility': recent_volatility,
        'ma_spread': ma_spread,
        'ma_spread_std': ma_spread_std,
        'avg_turnover': avg_turnover,
        'volume_boost': volume_boost,
        'price_position': price_position,
        'ma60': ma60,
        'ma120': ma120,
        'ma250': ma250,
        'ema20': ema20,
        'current_price': latest['close'],
        'market_cap': market_cap_yuan,
        'pe': latest.get('pe_ttm', latest.get('pe', 0)),
        'pb': latest.get('pb', 0),
        'ps': latest.get('ps', 0),
        'ps_ttm': latest.get('ps_ttm', latest.get('ps', 0)),
        'turnover_rate': latest['turnover_rate'],
        'data_days': len(df),
        'atr_ratio': atr_ratio,
        'adx': adx_now,
        'adx_slope': adx_slope,
        'bbw_percentile': bbw_percentile,
        'obv_slope': obv_slope,
        'obv_slope_norm': obv_slope_norm,
        'price_change_20d': price_change_20d,
        'obv_divergence': obv_divergence,
        'obv_accumulation_score': 0.0,
        'return_5d': float(trend_features["return_5d"]),
        'return_10d': float(trend_features["return_10d"]),
        'return_20d': float(trend_features["return_20d"]),
        'volume_ratio_5d_vs_60d': float(trend_features["volume_ratio_5d_vs_60d"]),
        'fresh_breakout': bool(trend_features["fresh_breakout"]),
        'near_breakout': bool(trend_features["near_breakout"]),
        'trend_emergence_score': float(trend_features["trend_emergence_score"]),
        'rsi': rsi_now,
        'vwap_ratio': vwap_ratio,
    }

    # 4. 横盘得分 (ORIGINAL - preserved for backward-compatible tier filtering)
    vol_score = max(0.0, 1.0 - (recent_volatility / VOLATILITY_THRESHOLD)) * 40
    ma_score = max(0.0, 1.0 - (ma_spread / 0.15)) * 25
    ma_std_score = max(0.0, 1.0 - (ma_spread_std / 0.03)) * 5
    price_score = max(0.0, 1.0 - (abs(price_position - 0.5) / 0.2)) * 20
    volume_score = max(0.0, (volume_boost - 1.0) / 0.5) * 10
    consolidation_score = int(min(100, vol_score + ma_score + ma_std_score + price_score + volume_score))

    scores['consolidation_score'] = consolidation_score

    # 4b. Squeeze Readiness Score (0-100) - NEW
    sq_atr = linear_score(atr_ratio, 0.5, 1.0, 40.0, invert=True)  # lower ratio = more squeeze
    sq_bbw = linear_score(bbw_percentile, 0.0, 0.5, 30.0, invert=True)  # lower percentile = tighter
    sq_adx = gaussian_score(adx_now, 15.0, 10.0, 30.0)  # ADX near 15 = quiescent, max squeeze
    squeeze_readiness = min(100.0, sq_atr + sq_bbw + sq_adx)
    scores['squeeze_readiness'] = squeeze_readiness

    # 4c. Volume Quality Score (0-100) - recomputed with universe OBV percentiles later
    volume_quality_score = compute_volume_quality_score(volume_boost, avg_turnover, 0.0)
    scores['volume_quality_score'] = volume_quality_score

    # 5. 趋势判断
    ma_trend = "neutral"
    if ma60 > ma120 > ma250:
        ma_trend = "bullish"
    elif ma60 < ma120 < ma250:
        ma_trend = "bearish"
    scores['ma_trend'] = ma_trend

    # 6. Momentum scoring (enhanced with new indicators)
    momentum_score = compute_momentum_score(
        df, latest, ema20, ma60, ma120, ma250,
        recent_high, recent_low, volume_boost,
        adx_now, adx_slope, obv_slope, vwap_ratio,
        obv_slope_norm=obv_slope_norm,
        obv_divergence=obv_divergence,
    )
    scores['momentum_score'] = momentum_score

    return scores


def compute_momentum_score(
    df: pd.DataFrame,
    latest: pd.Series,
    ema20: float,
    ma60: float,
    ma120: float,
    ma250: float,
    box_top: float,
    box_bottom: float,
    volume_boost: float,
    adx: float = 20.0,
    adx_slope: float = 0.0,
    obv_slope: float = 0.0,
    vwap_ratio: float = 1.0,
    obv_slope_norm: Optional[float] = None,
    obv_accumulation_score: Optional[float] = None,
    obv_divergence: bool = False,
    box_position: Optional[float] = None,
) -> float:
    """
    Compute momentum score for a stock (enhanced with continuous scoring).

    Components:
    - MA Alignment (EMA20 > MA60 > MA120): 0-25 points
    - ADX Inflection (rising from < 25):    0-20 points
    - Box Position (0.6-0.95 ideal):        0-15 points
    - OBV Accumulation/Divergence:          0-20 points
    - Volume Confirmation (1.2-3.0x):       0-10 points
    - VWAP Confirmation (close near/above): 0-10 points

    Returns:
        Momentum score from 0-100
    """
    close = finite_float(latest.get('close', latest.get('current_price', 0.0)))
    score = 0.0

    # 1. MA Alignment Score (0-25 points) - continuous
    alignment = 0.0
    if close > ema20:
        alignment += 8.0
    if ema20 > ma60:
        alignment += 8.0
    if ma60 > ma120:
        alignment += 5.0
    if ma120 > ma250:
        alignment += 4.0
    score += alignment

    # 2. ADX Inflection Score (0-20 points) - ADX rising from below 25
    if adx < 25 and adx_slope > 0:
        score += linear_score(adx_slope, 0.0, 3.0, 20.0)
    elif adx < 20:
        score += 5.0  # Low ADX about to inflect

    # 3. Box Position Score (0-15 points) - continuous taper
    position = box_position
    if position is None:
        box_range = box_top - box_bottom
        position = (close - box_bottom) / box_range if box_range > EPSILON else None
    if position is not None:
        score += range_score(position, 0.25, 0.65, 15.0)

    # 4. OBV Accumulation/Divergence Score (0-20 points)
    if obv_accumulation_score is not None:
        score += linear_score(finite_float(obv_accumulation_score), 0.0, 100.0, 15.0)
        if obv_divergence:
            score += 5.0
    else:
        norm = obv_slope_norm if obv_slope_norm is not None else (0.0 if obv_slope <= 0 else None)
        if norm is not None:
            score += linear_score(finite_float(norm), 0.0, 0.20, 20.0)
        elif obv_slope > 0:
            score += 5.0

    # 5. Volume Confirmation Score (0-10 points) - continuous
    score += range_score(volume_boost, 1.2, 3.0, 10.0)

    # 6. VWAP Confirmation Score (0-10 points) - close near/above VWAP
    score += gaussian_score(vwap_ratio, 1.02, 0.05, 10.0)

    return min(100.0, score)


def screen_all_stocks() -> pd.DataFrame:
    """
    Screen all stocks using technical criteria.

    Returns:
        DataFrame of screened stocks with technical scores
    """
    logger.info("="*60)
    logger.info("STEP 1: Initial screening - 30-50 candidates")
    logger.info("="*60)

    # 加载基础数据
    stock_basic = load_stock_basic()
    stock_company = load_stock_company()

    # 获取所有股票tick文件
    tick_dir = Path('data/stock_ticks')
    tick_files = list(tick_dir.glob('*.parquet'))

    logger.info(f"Found {len(tick_files)} stock data files")

    results = []
    skipped = {
        'no_data': 0,
        'market_cap': 0,
        'insufficient_data': 0,
    }

    for i, tick_file in enumerate(tick_files):
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(tick_files)} files...")

        ts_code = tick_file.stem

        # 排除ST股票
        stock_info = stock_basic[stock_basic['ts_code'] == ts_code]
        if stock_info.empty:
            skipped['no_data'] += 1
            continue

        name = stock_info.iloc[0]['name']
        if EXCLUDE_ST and ('ST' in name or '*ST' in name):
            skipped['no_data'] += 1
            continue

        # 加载并分析tick数据
        try:
            df = pd.read_parquet(tick_file)
        except Exception as e:
            skipped['no_data'] += 1
            continue

        # 技术分析
        analysis = analyze_stock_technical(df)
        if analysis is None:
            if len(df) < MIN_DATA_DAYS:
                skipped['insufficient_data'] += 1
            else:
                skipped['market_cap'] += 1
            continue

        # 整合结果
        result = {
            'ts_code': ts_code,
            'name': name,
            'industry': stock_info.iloc[0]['industry'],
            'market': stock_info.iloc[0]['market'],
            'exchange': stock_info.iloc[0]['exchange'],
            'area': stock_info.iloc[0]['area'],
            **analysis
        }

        results.append(result)

    logger.info(f"Processing complete! Qualified: {len(results)}, Skipped: {sum(skipped.values())}")
    for k, v in skipped.items():
        logger.debug(f"  - {k}: {v}")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.warning("No stocks passed the initial filter!")
        return None

    results_df = compute_industry_relative_valuation(results_df)
    before_outlier_filter = len(results_df)
    results_df = results_df[~results_df["valuation_outlier"]].copy()
    skipped["market_cap"] += max(0, before_outlier_filter - len(results_df))
    if results_df.empty:
        logger.warning("No stocks remained after valuation filter!")
        return None

    results_df["obv_slope_norm"] = results_df.get("obv_slope_norm", pd.Series(0.0, index=results_df.index)).apply(finite_float)
    results_df["obv_divergence"] = results_df.get("obv_divergence", pd.Series(False, index=results_df.index)).astype(bool)
    results_df["obv_slope_norm_percentile"] = 0.0
    positive_obv = results_df["obv_slope_norm"] > 0
    if positive_obv.any():
        results_df.loc[positive_obv, "obv_slope_norm_percentile"] = results_df.loc[positive_obv, "obv_slope_norm"].rank(pct=True)
    divergence_bonus = results_df["obv_divergence"].map(lambda value: 15.0 if value else 0.0)
    results_df["obv_accumulation_score"] = (
        results_df["obv_slope_norm_percentile"] * 85.0 + divergence_bonus
    ).clip(lower=0.0, upper=100.0)

    results_df["volume_quality_score"] = results_df.apply(
        lambda row: compute_volume_quality_score(
            row.get("volume_boost", 1.0),
            row.get("avg_turnover", 0.0),
            row.get("obv_accumulation_score", 0.0),
        ),
        axis=1,
    )
    results_df["momentum_score"] = results_df.apply(
        lambda row: compute_momentum_score(
            pd.DataFrame(),
            row,
            finite_float(row.get("ema20")),
            finite_float(row.get("ma60")),
            finite_float(row.get("ma120")),
            finite_float(row.get("ma250")),
            0.0,
            0.0,
            row.get("volume_boost", 1.0),
            row.get("adx", 20.0),
            row.get("adx_slope", 0.0),
            row.get("obv_slope", 0.0),
            row.get("vwap_ratio", 1.0),
            obv_slope_norm=row.get("obv_slope_norm", 0.0),
            obv_accumulation_score=row.get("obv_accumulation_score", 0.0),
            obv_divergence=bool(row.get("obv_divergence", False)),
            box_position=row.get("price_position", None),
        ),
        axis=1,
    )
    results_df["trend_emergence_score"] = results_df.apply(
        lambda row: compute_trend_emergence_score(
            row.get("return_5d", 0.0),
            row.get("return_10d", 0.0),
            row.get("return_20d", 0.0),
            row.get("volume_ratio_5d_vs_60d", 1.0),
            bool(row.get("fresh_breakout", False)),
            bool(row.get("near_breakout", False)),
            row.get("obv_accumulation_score", 0.0),
        ),
        axis=1,
    )

    # Composite ranking: trend-first, valuation-aware
    results_df['composite_score'] = (
        results_df['consolidation_score'] * 0.40 +
        results_df['momentum_score'] * 0.15 +
        results_df['volume_quality_score'] * 0.18 +
        results_df['squeeze_readiness'] * 0.15 +
        results_df['valuation_quality_score'] * 0.12
    )
    results_df["technical_selection_score"] = (
        results_df["composite_score"] * 0.70 +
        results_df["trend_emergence_score"] * 0.30
    )
    results_df = results_df.sort_values('composite_score', ascending=False)

    # Distribution logging for calibration
    for col in ['composite_score', 'consolidation_score', 'momentum_score', 'squeeze_readiness', 'volume_quality_score', 'trend_emergence_score']:
        if col in results_df.columns:
            vals = results_df[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col}: p25={vals.quantile(0.25):.1f} p50={vals.quantile(0.50):.1f} p75={vals.quantile(0.75):.1f}")

    # 添加公司信息
    if not stock_company.empty:
        company_cols = [
            'ts_code',
            'main_business',
            'business_scope',
            'introduction',
            'employees',
            'province',
            'city',
        ]
        company_info = stock_company[company_cols]
        results_df = results_df.merge(company_info, on='ts_code', how='left')

    return results_df


def save_results(df, output_file='screening_results.json'):
    """Save screening results to file."""
    # 转换为可序列化的格式
    records = df.head(50).to_dict('records')

    # 格式化数值
    for record in records:
        record['market_cap_fmt'] = f"{record['market_cap'] / 1e8:.2f}亿"
        pe = safe_positive_float(record.get('pe'))
        pb = safe_positive_float(record.get('pb'))
        ps_ttm = safe_positive_float(record.get('ps_ttm'))
        record['pe_fmt'] = f"{pe:.2f}" if pe is not None else "N/A"
        record['pb_fmt'] = f"{pb:.2f}" if pb is not None else "N/A"
        record['ps_ttm_fmt'] = f"{ps_ttm:.2f}" if ps_ttm is not None else "N/A"
        record['composite_score_fmt'] = f"{record['composite_score']:.2f}"
        record['valuation_quality_score_fmt'] = f"{record.get('valuation_quality_score', 0.0):.2f}"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


def print_summary(df, top_n=30):
    """Print summary of screening results."""
    logger.info("="*100)
    logger.info(f"TOP {top_n} 潜力成长股 - 初筛宽名单")
    logger.info("="*100)

    summary = df.head(top_n)[[
        'ts_code', 'name', 'industry', 'market_cap',
        'pe', 'pb', 'ps_ttm', 'valuation_label', 'valuation_quality_score',
        'consolidation_score', 'volume_boost', 'avg_turnover', 'ma_trend', 'exchange'
    ]].copy()

    # 格式化列
    summary['market_cap_fmt'] = summary['market_cap'].apply(lambda x: f"{x/1e8:.2f}亿" if pd.notna(x) else "N/A")
    summary['pe_fmt'] = summary['pe'].apply(lambda x: f"{x:.2f}" if safe_positive_float(x) is not None else "N/A")
    summary['valuation_score_fmt'] = summary['valuation_quality_score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    logger.info("{:<12} {:<8} {:<12} {:<10} {:<8} {:<10} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
        "代码", "名称", "行业", "市值", "PE", "估值", "估值分", "横盘分", "量能倍", "趋势", "交易所"
    ))
    logger.info("-" * 100)

    for _, row in summary.iterrows():
        logger.info("{:<12} {:<8} {:<12} {:<10} {:<8} {:<10} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
            row['ts_code'], row['name'], row['industry'][:10],
            row['market_cap_fmt'], row['pe_fmt'],
            row['valuation_label'],
            row['valuation_score_fmt'],
            f"{row['consolidation_score']:.0f}",
            f"{row['volume_boost']:.2f}",
            row['ma_trend'], row['exchange']
        ))

    # 统计信息
    logger.info("="*100)
    logger.info("统计摘要")
    logger.info("="*100)

    logger.info(f"行业分布:")
    industry_dist = df.head(top_n)['industry'].value_counts().head(10)
    for industry, count in industry_dist.items():
        logger.info(f"  {industry}: {count}只")

    logger.info(f"交易所分布:")
    exchange_dist = df.head(top_n)['exchange'].value_counts()
    for exchange, count in exchange_dist.items():
        logger.info(f"  {exchange}: {count}只")

    logger.info(f"市值分布:")
    mc_bins = [50, 100, 200, 300]
    mc_labels = ['50-100亿', '100-200亿', '200-300亿']
    df['mc_bin'] = pd.cut(df['market_cap'] / 1e8, bins=mc_bins, labels=mc_labels, include_lowest=True)
    mc_dist = df.head(top_n)['mc_bin'].value_counts().sort_index()
    for mc_range, count in mc_dist.items():
        logger.info(f"  {mc_range}: {count}只")

    logger.info(f"横盘得分分布:")
    score_bins = [0, 40, 60, 80, 100]
    score_labels = ['0-40分', '40-60分', '60-80分', '80-100分']
    df['score_bin'] = pd.cut(df['consolidation_score'], bins=score_bins, labels=score_labels, include_lowest=True)
    score_dist = df.head(top_n)['score_bin'].value_counts().sort_index()
    for score_range, count in score_dist.items():
        logger.info(f"  {score_range}: {count}只")


if __name__ == '__main__':
    # Setup logging when run as script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 执行筛选
    results_df = screen_all_stocks()

    if results_df is not None and not results_df.empty:
        # 保存结果
        save_results(results_df)

        # 打印摘要
        print_summary(results_df, top_n=50)

        # 导出详细数据
        output_csv = f'screening_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.head(50).to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"详细数据已导出至: {output_csv}")
    else:
        logger.error("筛选失败或无符合条件的股票")
