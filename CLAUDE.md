# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### Data Storage

- **Parquet format** for efficient storage in `data/`:
  - `stock_basic/`: Basic stock info (codes, names, industries)
  - `stock_company/`: Company details (introduction, business scope, location)
  - `stock_ticks/`: Historical price data per stock
  - `financial/`: Financial statements and metrics
- **Reports**: Timestamped HTML files in `reports/report_YYYYMMDD_HHMMSS/`

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

**ETL:** `DataFetcher.fetch_stock_basic_info()` - Fetched during `sync` and `fetch` commands

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

**ETL:** `DataFetcher.fetch_stock_company_info()` - Fetched during `fetch` command

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

**ETL:** `DataFetcher.fetch_stock_daily_data()` - Fetched per stock, with incremental updates based on freshness check

---

### 4. Financial Data (`data/financial/income/{ts_code}.parquet`)

**Source:** `pro.income()` API - Fetched on-demand during stock selection

| Column | Type | Description |
|--------|------|-------------|
| ts_code | string | Stock code |
| ann_date | string | Announcement date |
| f_ann_date | string | First announcement date |
| end_date | string | Report end date (YYYYMMDD) |
| report_type | string | Report type (1=合并报表) |
| comp_type | string | Company type |
| basic_eps | float | Basic EPS per share |
| diluted_eps | float | Diluted EPS per share |
| total_revenue | float | Total revenue |
| revenue | float | Operating revenue |
| oper_cost | float | Operating cost |
| operate_profit | float | Operating profit |
| total_profit | float | Total profit |
| n_income | float | Net income |
| n_income_attr_p | float | Net income attributable to parent |
| minority_gain | float | Minority interest gain |
| period_info | dict | Derived period type (Q1/Q2/Q3/Q4) |
| period_type | string | Period type (annual/semi_annual/quarterly) |
| period_name | string | Period name (年报/半年报/季报) |
| report_year | int | Report year |
| report_quarter | int | Quarter number (1-4) |
| end_date_dt | datetime | Converted end_date for calculations |
| revenue_yoy | float | YoY revenue growth (%) |
| n_income_yoy | float | YoY net income growth (%) |
| oper_profit_yoy | float | YoY operating profit growth (%) |
| revenue_qoq | float | QoQ revenue growth (%) |
| n_income_qoq | float | QoQ net income growth (%) |
| gross_margin | float | Gross margin (%) |
| net_margin | float | Net margin (%) |
| revenue_consecutive_growth | int | Consecutive quarters of revenue growth |
| n_income_consecutive_growth | int | Consecutive quarters of profit growth |
| financial_health_score | float | Financial health score (0-100) |
| revenue_stability_score | float | Revenue stability score (0-100) |
| profitability_score | float | Profitability score (0-100) |
| growth_momentum_score | float | Growth momentum score (0-100) |

**ETL:** `FinancialManager.fetch_financial_data()` - Fetched on-demand, with derived metrics calculated in `calculate_financial_trends()`

---

### 5. Enriched Stock Analysis (In-Memory During Selection)

**Source:** `StockSelector.analyze_stock()` - Computed during selection

| Field | Description | Source |
|-------|-------------|--------|
| ts_code | Stock code | From ticks |
| current_price | Latest closing price | From ticks |
| ma_value | 777-day moving average | Computed |
| price_diff | Price - MA | Computed |
| price_diff_pct | (Price - MA) / MA * 100 | Computed |
| pe, pb, pe_ttm | Valuation ratios | From ticks |
| market_cap | Total market cap (converted to RMB) | From ticks (total_mv * 10000) |
| name | Company name | From stock_basic |
| industry | Industry | From stock_basic |
| market | Market code | From stock_basic |
| exchange | Exchange | From stock_basic |
| company_introduction | Company intro | From stock_company |
| main_business | Main business | From stock_company |
| business_scope | Business scope | From stock_company |
| employees | Employee count | From stock_company |
| province, city | Location | From stock_company |
| financial_summary | Dict of financial metrics | From FinancialManager |
| llm_task_id | Task ID for LLM analysis | From TaskQueue |
| llm_outlook | LLM analysis result | From LLM API |
| llm_rating | LLM rating (1-5) | From LLM API |
| data_days | Number of days of data | Computed |
| latest_date | Latest trade date | From ticks |
| price_history | Last 360 days OHLCV + MA | From ticks (prepared for charts) |


## A股潜力成长组合筛选流程

### 概述

此流程基于"基本面成长+资金面博弈"双重分析体系，挖掘具有高成长潜力的A股标的。筛选标准结合估值安全边际、题材热度、技术形态和基本面质量四个维度。

### 执行流程

```bash
# Step 1: 初步筛选 - 生成30-50只宽名单
python screen_growth_stocks.py

# Step 2: 绘制Top N股票的K线技术分析图
python draw_charts.py
```

### 脚本说明

#### 1. screen_growth_stocks.py - 初步筛选脚本

**筛选参数：**
```python
MARKET_CAP_MIN = 20e8   # 20亿市值
MARKET_CAP_MAX = 300e8  # 300亿市值
MIN_DATA_DAYS = 120     # 最少交易日数据
CONSOLIDATION_DAYS = 60  # 横盘观察天数
VOLATILITY_THRESHOLD = 0.30  # 横盘波动幅度阈值（30%）
```

**横盘得分计算（满分100）：**
- 40分：60日波动率 < 30%
- 30分：MA20/MA60/MA120粘合度 < 15%
- 20分：价格处于横盘区间中部(30%-70%)
- 10分：近期量能倍数 > 1.2倍

**输出文件：**
- `screening_results.json` - 完整筛选结果（JSON格式）
- `screening_results_YYYYMMDD_HHMMSS.csv` - 完整筛选结果（CSV格式）

#### 2. draw_charts.py - K线技术分析图绘制

**功能特性：**
- 绘制近120个交易日K线图（红涨绿跌）
- 叠加MA20、MA60、MA120三条均线
- 成交量副图显示
- 量能倍数曲线（橙色，显示放量信号）
- 标题显示关键技术指标：收盘价、均线、均线粘合度、趋势状态、量能倍数

**输出文件：**
- `charts/{代码}_{名称}_技术分析.png` - 单只股票K线图
- `charts/技术形态分析汇总.txt` - 技术指标汇总报告

**依赖安装：**
```bash
pip install mplfinance matplotlib
```

### 筛选标准四维度

1. **估值与市值 (Safety Margin)**
   - 市值区间：50亿-300亿人民币
   - 剔除处于历史高位或短期翻倍的股票

2. **题材与资金 (Market Sentiment & Story)**
   - 关联当前市场高热度概念
   - 优先选择"有故事但未完全兑现"的公司

3. **技术形态 (Technical Patterns)**
   - 长周期横盘（3-6个月，波动幅度<30%）
   - 均线粘合（MA20/MA60/MA120相互纠缠或刚呈多头排列）
   - 底部放量信号

4. **基本面底色 (Fundamentals)**
   - 题材正宗标的（有技术、有产品、有份额）
   - 排除ST、退市风险、财务造假嫌疑

### 常见热点题材参考

| 题材类别 | 核心概念 |
|:---|:---|
| AI应用 | AI营销、端侧AI、物理AI、AI大模型 |
| 人形机器人 | 减速器、电机、执行器 |
| 脑机接口 | 神经科技、脑机设备 |
| 低空经济 | eVTOL、无人机、低空飞行 |
| 半导体/国产替代 | 存储芯片、国产算力、半导体设备 |
| 并购重组 | 产业整合、国企改革 |
| 商业航天 | 航天制造、卫星应用 |
| 智能驾驶 | L3级自动驾驶、车路协同 |
| 新型储能 | 固态电池、储能系统 |

### 实战案例（2026-01-12）

筛选结果Top 5：

| 代码 | 名称 | 核心题材 | 均线粘合度 | 趋势状态 |
|:---|:---|:---|:---|:---|
| 002527.SZ | 新时达 | 人形机器人 | 0.0434 | 空头排列 |
| 000099.SZ | 中信海直 | 低空经济 | 0.0387 | 空头排列 |
| 002657.SZ | 中科金财 | AI应用/AIGC | 0.0698 | 空头排列 |
| 000582.SZ | 北部湾港 | 一带一路/RCEP | 0.0346 | 多头排列 |
| 600761.SH | 安徽合力 | 智能物流/AGV | 0.0166 | 多头排列 |

### 投资建议

1. **分批建仓**：建议分3-4批建仓，避免追高
2. **仓位配置**：核心仓位（60%）+ 卫星仓位（40%）
3. **止盈止损**：
   - 止盈：单票涨幅超过30%分批止盈
   - 止损：跌破-15%严格止损
4. **持仓周期**：建议3-6个月中期持仓

### 风险提示

- 题材炒作风险：部分标的概念成分较重，需关注实际落地收入
- 行业周期性：港口、工程机械、半导体设备等受宏观经济影响
- 技术不确定性：人形机器人、低空经济、AI应用等新兴技术商业化存在不确定性
- 市值较小：大部分标的市值在50-200亿区间，股价波动较大
