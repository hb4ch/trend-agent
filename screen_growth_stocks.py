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
EXCLUDE_HIGH_PE = 100   # 排除PE过高的股票


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

    obv_series = compute_obv(df)
    obv_recent = obv_series.tail(20)
    if len(obv_recent) >= 2:
        obv_slope = float(np.polyfit(range(len(obv_recent)), obv_recent.values, 1)[0])
    else:
        obv_slope = 0.0

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
        'turnover_rate': latest['turnover_rate'],
        'data_days': len(df),
        'atr_ratio': atr_ratio,
        'adx': adx_now,
        'adx_slope': adx_slope,
        'bbw_percentile': bbw_percentile,
        'obv_slope': obv_slope,
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

    # 4c. Volume Quality Score (0-100) - NEW
    vq_obv = linear_score(obv_slope, 0.0, abs(obv_slope) * 2 + EPSILON, 40.0) if obv_slope > 0 else 0.0
    vq_boost = range_score(volume_boost, 1.2, 3.0, 35.0)
    vq_turnover = range_score(avg_turnover, 1.0, 5.0, 25.0)
    volume_quality_score = min(100.0, vq_obv + vq_boost + vq_turnover)
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
) -> float:
    """
    Compute momentum score for a stock (enhanced with continuous scoring).

    Components:
    - MA Alignment (EMA20 > MA60 > MA120): 0-25 points
    - ADX Inflection (rising from < 25):    0-20 points
    - Box Position (0.6-0.95 ideal):        0-15 points
    - OBV Divergence (positive slope):      0-20 points
    - Volume Confirmation (1.2-3.0x):       0-10 points
    - VWAP Confirmation (close near/above): 0-10 points

    Returns:
        Momentum score from 0-100
    """
    close = float(latest['close'])
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
    box_range = box_top - box_bottom
    if box_range > EPSILON:
        position = (close - box_bottom) / box_range
        score += range_score(position, 0.6, 0.95, 15.0)

    # 4. OBV Divergence Score (0-20 points) - positive OBV slope = accumulation
    if obv_slope > 0:
        score += min(20.0, linear_score(obv_slope, 0.0, abs(obv_slope) * 2 + EPSILON, 20.0))

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

        # 排除PE过高的股票（需要有效的PE值）
        pe_val = analysis.get('pe')
        if pe_val is not None and (pe_val > EXCLUDE_HIGH_PE or pe_val < 0):
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

    # Composite ranking: consolidation 35%, momentum 25%, volume quality 20%, squeeze readiness 20%
    results_df['composite_score'] = (
        results_df['consolidation_score'] * 0.35 +
        results_df['momentum_score'] * 0.25 +
        results_df['volume_quality_score'] * 0.20 +
        results_df['squeeze_readiness'] * 0.20
    )
    results_df = results_df.sort_values('composite_score', ascending=False)

    # Distribution logging for calibration
    for col in ['composite_score', 'consolidation_score', 'momentum_score', 'squeeze_readiness', 'volume_quality_score']:
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
        record['pe_fmt'] = f"{record['pe']:.2f}"
        record['pb_fmt'] = f"{record['pb']:.2f}"
        record['composite_score_fmt'] = f"{record['composite_score']:.2f}"

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
        'pe', 'consolidation_score', 'volume_boost',
        'avg_turnover', 'ma_trend', 'exchange'
    ]].copy()

    # 格式化列
    summary['market_cap_fmt'] = summary['market_cap'].apply(lambda x: f"{x/1e8:.2f}亿" if pd.notna(x) else "N/A")
    summary['pe_fmt'] = summary['pe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    logger.info("{:<12} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
        "代码", "名称", "行业", "市值", "PE", "横盘分", "量能倍", "换手", "趋势", "交易所"
    ))
    logger.info("-" * 100)

    for _, row in summary.iterrows():
        logger.info("{:<12} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
            row['ts_code'], row['name'], row['industry'][:10],
            row['market_cap_fmt'], row['pe_fmt'],
            f"{row['consolidation_score']:.0f}",
            f"{row['volume_boost']:.2f}",
            f"{row['avg_turnover']:.2f}",
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
