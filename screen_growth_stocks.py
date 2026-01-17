#!/usr/bin/env python3
"""
A股潜力成长组合筛选脚本
基于"基本面成长+资金面博弈"双重分析体系
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# ============ 筛选参数配置 ============
MARKET_CAP_MIN = 20e8   # 50亿市值
MARKET_CAP_MAX = 300e8  # 300亿市值
MIN_DATA_DAYS = 120     # 最少交易日数据
CONSOLIDATION_DAYS = 120  # 横盘观察天数
VOLATILITY_THRESHOLD = 0.35  # 横盘波动幅度阈值（35%）
MIN_AVG_TURNOVER = 1.0  # 最低平均换手率
RECENT_VOLUME_BOOST = 1.5  # 近期放量倍数

# 排除条件
EXCLUDE_ST = False       # 排除ST股票
EXCLUDE_HIGH_PE = 100   # 排除PE过高的股票


def load_stock_basic():
    """加载股票基本信息"""
    print("Loading stock basic info...")
    df = pd.read_parquet('data/stock_basic/stock_basic.parquet')
    return df


def load_stock_company():
    """加载公司信息"""
    print("Loading stock company info...")
    try:
        df = pd.read_parquet('data/stock_company/stock_company.parquet')
        return df
    except Exception as e:
        print(f"Warning: Could not load company info: {e}")
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
    recent_volatility = (recent_high - recent_low) / (recent_low + 1e-9)

    # 均线系统（MA20/60/120 纠缠度：最近20天的平均粘合程度）
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()
    ma_recent = df[["ma20", "ma60", "ma120"]].tail(20).dropna()
    if ma_recent.empty:
        return None
    daily_spread = (ma_recent.max(axis=1) - ma_recent.min(axis=1)) / (ma_recent.min(axis=1) + 1e-9)
    ma_spread = float(daily_spread.mean())
    ma_spread_std = float(daily_spread.std(ddof=0)) if len(daily_spread) > 1 else 0.0
    ma20 = float(df["ma20"].iloc[-1])
    ma60 = float(df["ma60"].iloc[-1])
    ma120 = float(df["ma120"].iloc[-1])

    # 量能分析
    avg_turnover = recent['turnover_rate'].mean()
    recent_turnover = df['turnover_rate'].tail(10).mean()
    volume_boost = recent_turnover / (avg_turnover + 0.001)

    # 价格位置
    price_position = (latest['close'] - recent_low) / (recent_high - recent_low + 0.001)

    # 3. 计算综合得分
    scores = {
        'volatility': recent_volatility,
        'ma_spread': ma_spread,
        'ma_spread_std': ma_spread_std,
        'avg_turnover': avg_turnover,
        'volume_boost': volume_boost,
        'price_position': price_position,
        'ma20': ma20,
        'ma60': ma60,
        'ma120': ma120,
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
    if ma20 > ma60 > ma120:
        ma_trend = "bullish"
    elif ma20 < ma60 < ma120:
        ma_trend = "bearish"
    scores['ma_trend'] = ma_trend

    return scores


def screen_all_stocks():
    """
    对所有股票进行筛选
    """
    print("\n" + "="*60)
    print("STEP 1: 初步筛选 - 30-50只宽名单")
    print("="*60 + "\n")

    # 加载基础数据
    stock_basic = load_stock_basic()
    stock_company = load_stock_company()

    # 获取所有股票tick文件
    tick_dir = Path('data/stock_ticks')
    tick_files = list(tick_dir.glob('*.parquet'))

    print(f"Found {len(tick_files)} stock data files")

    results = []
    skipped = {
        'no_data': 0,
        'market_cap': 0,
        'insufficient_data': 0,
    }

    for i, tick_file in enumerate(tick_files):
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(tick_files)} files...")

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

    print(f"\nProcessing complete!")
    print(f"  Qualified: {len(results)}")
    print(f"  Skipped: {sum(skipped.values())}")
    for k, v in skipped.items():
        print(f"    - {k}: {v}")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo stocks passed the initial filter!")
        return None

    # 排序：按横盘得分 + 换手率综合排序
    results_df['composite_score'] = (
        results_df['consolidation_score'] * 0.6 +
        results_df['volume_boost'] * 10 * 0.2 +
        results_df['avg_turnover'] * 2 * 0.2
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
    """保存筛选结果"""
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

    print(f"\nResults saved to {output_file}")


def print_summary(df, top_n=30):
    """打印筛选结果摘要"""
    print("\n" + "="*100)
    print(f"TOP {top_n} 潜力成长股 - 初筛宽名单")
    print("="*100)

    summary = df.head(top_n)[[
        'ts_code', 'name', 'industry', 'market_cap',
        'pe', 'consolidation_score', 'volume_boost',
        'avg_turnover', 'ma_trend', 'exchange'
    ]].copy()

    # 格式化列
    summary['market_cap_fmt'] = summary['market_cap'].apply(lambda x: f"{x/1e8:.2f}亿" if pd.notna(x) else "N/A")
    summary['pe_fmt'] = summary['pe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    print("\n{:<12} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
        "代码", "名称", "行业", "市值", "PE", "横盘分", "量能倍", "换手", "趋势", "交易所"
    ))
    print("-" * 100)

    for _, row in summary.iterrows():
        print("{:<12} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<6}".format(
            row['ts_code'], row['name'], row['industry'][:10],
            row['market_cap_fmt'], row['pe_fmt'],
            f"{row['consolidation_score']:.0f}",
            f"{row['volume_boost']:.2f}",
            f"{row['avg_turnover']:.2f}",
            row['ma_trend'], row['exchange']
        ))

    # 统计信息
    print("\n" + "="*100)
    print("统计摘要")
    print("="*100)

    print(f"\n行业分布:")
    industry_dist = df.head(top_n)['industry'].value_counts().head(10)
    for industry, count in industry_dist.items():
        print(f"  {industry}: {count}只")

    print(f"\n交易所分布:")
    exchange_dist = df.head(top_n)['exchange'].value_counts()
    for exchange, count in exchange_dist.items():
        print(f"  {exchange}: {count}只")

    print(f"\n市值分布:")
    mc_bins = [50, 100, 200, 300]
    mc_labels = ['50-100亿', '100-200亿', '200-300亿']
    df['mc_bin'] = pd.cut(df['market_cap'] / 1e8, bins=mc_bins, labels=mc_labels, include_lowest=True)
    mc_dist = df.head(top_n)['mc_bin'].value_counts().sort_index()
    for mc_range, count in mc_dist.items():
        print(f"  {mc_range}: {count}只")

    print(f"\n横盘得分分布:")
    score_bins = [0, 40, 60, 80, 100]
    score_labels = ['0-40分', '40-60分', '60-80分', '80-100分']
    df['score_bin'] = pd.cut(df['consolidation_score'], bins=score_bins, labels=score_labels, include_lowest=True)
    score_dist = df.head(top_n)['score_bin'].value_counts().sort_index()
    for score_range, count in score_dist.items():
        print(f"  {score_range}: {count}只")


if __name__ == '__main__':
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
        print(f"\n详细数据已导出至: {output_csv}")
    else:
        print("\n筛选失败或无符合条件的股票")
