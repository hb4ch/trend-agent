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

    # 3. 计算综合得分
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
        'current_price': latest['close'],
        'market_cap': market_cap_yuan,
        'pe': latest.get('pe_ttm', latest.get('pe', 0)),
        'pb': latest.get('pb', 0),
        'turnover_rate': latest['turnover_rate'],
        'data_days': len(df),
    }

    # 4. 横盘得分（连续评分，避免满分扎堆）
    vol_score = max(0.0, 1.0 - (recent_volatility / VOLATILITY_THRESHOLD)) * 40
    ma_score = max(0.0, 1.0 - (ma_spread / 0.15)) * 25
    ma_std_score = max(0.0, 1.0 - (ma_spread_std / 0.03)) * 5
    price_score = max(0.0, 1.0 - (abs(price_position - 0.5) / 0.2)) * 20
    volume_score = max(0.0, (volume_boost - 1.0) / 0.5) * 10
    consolidation_score = int(min(100, vol_score + ma_score + ma_std_score + price_score + volume_score))

    scores['consolidation_score'] = consolidation_score

    # 5. 趋势判断
    ma_trend = "neutral"
    if ma60 > ma120 > ma250:
        ma_trend = "bullish"
    elif ma60 < ma120 < ma250:
        ma_trend = "bearish"
    scores['ma_trend'] = ma_trend

    # 6. Momentum scoring (new)
    momentum_score = compute_momentum_score(df, latest, ma60, ma120, ma250, recent_high, recent_low, volume_boost)
    scores['momentum_score'] = momentum_score

    return scores


def compute_momentum_score(
    df: pd.DataFrame,
    latest: pd.Series,
    ma60: float,
    ma120: float,
    ma250: float,
    box_top: float,
    box_bottom: float,
    volume_boost: float,
) -> float:
    """
    Compute momentum score for a stock.

    Scores:
    - MA alignment (price above MA20/MA60): 0-40 points
    - Box position (proximity to breakout): 0-30 points
    - Volume confirmation (recent vs historical): 0-30 points

    Returns:
        Momentum score from 0-100
    """
    close = latest['close']
    score = 0.0

    # 1. MA Alignment Score (0-40 points)
    # Price above short-term MAs is bullish
    ma20 = df["close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else close
    ma_alignment = 0.0
    if close > ma20:
        ma_alignment += 15.0
    if close > ma60:
        ma_alignment += 15.0
    if ma20 > ma60:
        ma_alignment += 10.0  # Short above medium is bullish
    score += ma_alignment

    # 2. Box Position Score (0-30 points)
    # Proximity to box top indicates potential breakout
    box_range = box_top - box_bottom
    if box_range > EPSILON:
        position = (close - box_bottom) / box_range
        # Higher position = closer to breakout (but not already broken out)
        if 0.7 <= position <= 0.95:
            score += 30.0  # Ideal breakout zone
        elif 0.5 <= position < 0.7:
            score += 20.0  # Good position
        elif 0.3 <= position < 0.5:
            score += 10.0  # Neutral
        # Below 0.3 or above 0.95 = 0 points

    # 3. Volume Confirmation Score (0-30 points)
    # Volume boost between 1.2-3.0x is ideal (accumulation without distribution)
    if 1.2 <= volume_boost <= 3.0:
        score += 30.0  # Ideal volume
    elif 1.0 <= volume_boost < 1.2:
        score += 15.0  # Slight increase
    elif 3.0 < volume_boost <= 5.0:
        score += 10.0  # High volume (may indicate distribution)
    # Below 1.0 or above 5.0 = 0 points

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

    # 排序：按横盘得分 + 动量 + 换手率综合排序
    # Rebalanced weights: consolidation 40%, momentum 20%, volume 25%, turnover 15%
    results_df['composite_score'] = (
        results_df['consolidation_score'] * 0.40 +
        results_df['momentum_score'] * 0.20 +
        results_df['volume_boost'] * 10 * 0.25 +
        results_df['avg_turnover'] * 2 * 0.15
    )
    results_df = results_df.sort_values('composite_score', ascending=False)

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
