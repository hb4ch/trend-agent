# 龙虎榜数据集成方案

> **核心理念**: 龙虎榜数据是Web Search热点的**补充验证**，而非**替换**。最终整合依赖DeepSeek做多源数据融合。

---

## 1. 数据关系架构

### 1.1 多源数据融合架构

```
┌─────────────────────────────────────────────────────────┐
│                    DeepSeek 整合层                      │
│  • 多源数据融合                                          │
│  • 交叉验证判断                                          │
│  • 最终综合评估                                          │
└─────────────────────────────────────────────────────────┘
                    ▲           ▲           ▲
                    │           │           │
        ┌───────────┴─────┐   │   ┌───────┴──────────┐
        │  Web Search     │   │   │  Dragon Tiger    │
        │  • 新闻情绪      │   │   │  • 资金真实流向  │
        │  • 政策催化      │   │   │  • 游资偏好      │
        │  • 市场叙事      │   │   │  • 机构动向      │
        └──────────────────┘   │   └──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  技术形态筛选        │
                    │  • 待时机标的        │
                    │  • 横盘震荡          │
                    │  • 均线粘合          │
                    └─────────────────────┘
```

### 1.2 两种数据源的互补性

| 维度 | Web Search | 龙虎榜 | 结合后的价值 |
|------|------------|--------|-------------|
| **时效性** | 实时新闻 | T+1数据 | 互补覆盖 |
| **信息类型** | 政策/催化/情绪 | 真实资金行为 | 相互验证 |
| **覆盖范围** | 所有热点股 | 仅异动股 | 过滤噪音 |
| **噪音水平** | 媒体夸大 | 游资一日游 | 交叉过滤 |
| **数据质量** | 定性为主 | 定量精确 | 互补增强 |

### 1.3 交叉验证的威力

| 场景 | Web Search | 龙虎榜 | DeepSeek判断 | 操作建议 |
|------|------------|--------|---------------|----------|
| **A: 主线机会** ✅ | 半导体政策利好 | 上榜45次，净买入150亿 | CONFIRMED | 重点布局 |
| **B: 纯炒作** ⚠️ | 脑机接口火热 | 仅3次游资一日游 | WEB_ONLY | 观望为主 |
| **C: 潜在机会** 🔍 | 无热点 | 某股机构持续买入 | CAPITAL_ONLY | 深入挖掘 |
| **D: 不关注** ❌ | 无热点 | 无上榜 | WEAK | 不关注 |

### 1.4 选股逻辑优化

**之前的理解（偏差）**:
- 追高已经上榜的股票（被动跟随）

**正确的理解**:
- 观察游资在做什么**题材方向**
- 在这些题材内找**尚未启动**的标的（提前布局）
- 等待资金轮动到这个标的（待时机）

```
龙虎榜扫描 → 识别游资活跃题材 → 在题材内找"待时机"标的 → 提前布局
```

---

## 2. 数据概览

### 2.1 top_list (龙虎榜榜单)

**文件**: `data/top_list/YYYYMMDD.parquet`
**日均记录**: ~84条

| 字段 | 类型 | 说明 | Alpha信号 |
|------|------|------|-----------|
| ts_code | string | 股票代码 | 主键 |
| trade_date | string | 交易日期 | 时间序列 |
| name | string | 股票名称 | - |
| close | float | 收盘价 | - |
| pct_change | float | 涨跌幅(%) | 强弱信号 |
| turnover_rate | float | 换手率(%) | 活跃度 |
| l_sell | float | 卖五营业部总卖出额 | - |
| l_buy | float | 买五营业部总买入额 | - |
| l_amount | float | 买卖总成交额 | - |
| net_amount | float | 净买入额(买入-卖出) | **资金方向** |
| net_rate | float | 净买入率(%) | **资金强度** |
| amount_rate | float | 买入额占成交比(%) | **资金占比** |
| float_values | float | 流通市值 | - |
| reason | string | 上榜原因 | 触发条件 |

### 2.2 top_inst (龙虎榜机构明细)

**文件**: `data/top_inst/YYYYMMDD.parquet`
**日均记录**: ~888条

| 字段 | 类型 | 说明 | Alpha信号 |
|------|------|------|-----------|
| ts_code | string | 股票代码 | 主键 |
| trade_date | string | 交易日期 | 时间序列 |
| exalter | string | 机构/营业部名称 | **资金身份** |
| buy | float | 买入金额(元) | - |
| buy_rate | float | 买入占成交量比(%) | 买入力度 |
| sell | float | 卖出金额(元) | - |
| sell_rate | float | 卖出占成交量比(%) | 卖出力度 |
| net_buy | float | 净买入额(买入-卖出) | **资金态度** |
| side | int | 0=买五, 1=卖五 | 方向 |
| reason | string | 上榜原因 | 触发条件 |

**"聪明钱"识别**:
- `深股通专用` / `沪股通专用`: 北上资金
- `机构专用`: 机构席位
- 知名游资营业部: `宁波桑田路`, `上海溧阳路`, `作手新一`, `消闲派` 等

---

## 3. 时间窗口选择策略

| 场景 | 推荐窗口 | 理由 |
|------|----------|------|
| **识别游资题材趋势** | **30日** | 观察资金持续流入，识别题材主线 |
| **题材热度排序** | **10日** | 捕捉近期加速的题材（可能轮动） |
| **排除已大涨股票** | **60日** | 避免买到已经炒作过的 |
| **"待时机"筛选** | **实时** | 基于当前技术形态 |

---

## 4. 集成策略

### 4.1 Phase 1: 市场情报增强 (多源融合)

**核心**: 让DeepSeek做最终整合

```python
def phase1_market_intel() -> list[Theme]:
    """
    Phase 1: 市场情报 - 多源数据融合
    """
    # 1. Web Search: 捕捉新闻情绪、政策催化
    web_themes = search_market_themes()

    # 2. 龙虎榜: 识别资金真实流向 (30日窗口)
    dtl = DragonTigerList()
    capital_themes = dtl.identify_hot_themes(days=30)

    # 3. 融合: 交给DeepSeek做综合判断
    themes = deepseek_merge_themes({
        "web_search": web_themes,
        "dragon_tiger": capital_themes,
        "instruction": """
        请综合以下两类数据，判断真正的市场主线：

        1. Web Search热点（新闻情绪、政策催化）
        2. 龙虎榜资金流向（真实行为、游资偏好）

        融合原则：
        - 两者都确认 = 主线题材 (CONFIRMED)，重点布局
        - 仅Web有 = 观察中 (WEB_ONLY)，需资金验证
        - 仅龙虎榜有 = 潜在机会 (CAPITAL_ONLY)，深入挖掘
        - 两者都没有 = 不关注 (WEAK)

        输出每个主题的：
        - name: 主题名称
        - validation_status: 验证状态 (confirmed/web_only/capital_only/weak)
        - evidence: 整合后的证据
        - capital_signal: 资金信号总结
        """
    })

    return themes
```

### 4.2 Phase 2: 候选股筛选 (在热门题材内)

**核心**: 在确认的热门题材内，找"待时机"标的

```python
def phase2_quant_mining(themes: list[Theme]) -> pd.DataFrame:
    """
    Phase 2: 选股 - 在确认的热门题材内筛选
    """
    candidates = []

    # 只关注验证通过的主题
    confirmed_themes = [t for t in themes if t.validation_status == "confirmed"]

    for theme in confirmed_themes:
        # 1. 获取题材内所有股票
        theme_stocks = get_theme_stocks(theme.name)

        # 2. 排除: 已经在龙虎榜大涨过的 (避免追高)
        hot_stocks = get_recent_toplist_stocks(days=60)
        available_stocks = theme_stocks[~theme_stocks.isin(hot_stocks)]

        # 3. 筛选: 技术形态 + 待时机
        filtered = filter_by_consolidation(available_stocks)

        # 4. 优先级排序:
        #    - 龙虎榜有资金关注但未大涨的 > 完全没有上榜的
        filtered = rank_by_capital_interest(filtered, dtl)

        candidates.extend(filtered)

    return pd.DataFrame(candidates)
```

### 4.3 Phase 5: 报告增强

**核心**: 展示题材的资金趋势，而非仅个股

```python
def phase5_generate_report():
    """
    Phase 5: 报告生成 - 展示题材的资金验证
    """
    dtl = DragonTigerList()

    # 为每个主题准备资金分析
    theme_capital_analysis = {}
    for theme in themes:
        if theme.validation_status == "confirmed":
            # 展示该题材的资金流向趋势
            analysis = dtl.get_theme_capital_analysis(theme.name, days=30)
            theme_capital_analysis[theme.name] = analysis

    # 传递给DeepSeek，在报告中展示
    context = {
        "themes": themes,
        "theme_capital_analysis": theme_capital_analysis,  # 新增
        "candidates": candidates,
        ...
    }
```

**报告中的展示**:
```
**【资金验证】**<font color='purple'>
- **题材资金趋势** (近30日):
  - 半导体板块龙虎榜上榜45次
  - 累计净买入150亿，其中北上资金净买入80亿，机构净买入50亿
  - 资金结构: 北向53% + 机构34% + 游资13%
  - 趋势: 加速流入 (近10日上榜频率提升)

- **个股资金表现**:
  - 中芯国际: 上榜3次，累计净买入12亿
  - 北方华创: 上榜2次，累计净买入8亿
  - [其他标的]: 尚未上榜，但板块资金活跃，值得关注
</font>
```

---

## 5. 实现步骤

### Step 1: 创建DragonTigerList类 (utils.py)

```python
class DragonTigerList:
    """龙虎榜数据管理"""

    def __init__(self, data_dir: str = "data"):
        self.top_list_dir = Path(data_dir) / "top_list"
        self.top_inst_dir = Path(data_dir) / "top_inst"

    def load_recent_toplist(self, days: int = 30) -> pd.DataFrame:
        """加载最近N天的龙虎榜数据"""

    def load_stock_institutions(self, ts_code: str, days: int = 30) -> pd.DataFrame:
        """加载指定股票的机构交易明细"""

    def identify_hot_themes(self, days: int = 30) -> dict:
        """
        识别游资活跃题材 (30日窗口)

        Returns:
            {
                "半导体": {
                    "hit_count": 45,
                    "net_buy": 15.2e9,
                    "hot_stocks": ["中芯国际", "北方华创"],
                    "institution_mix": {"北上": 0.45, "机构": 0.35, "游资": 0.20},
                    "trend": "accelerating"
                },
                ...
            }
        """

    def get_theme_capital_analysis(self, theme_name: str, days: int = 30) -> dict:
        """获取题材的资金流向分析"""

    def get_recent_toplist_stocks(self, days: int = 60) -> list:
        """获取最近上榜的股票（用于排除）"""

    def identify_smart_money(self, df: pd.DataFrame) -> dict:
        """识别聪明钱(北上、机构、知名游资)"""
```

### Step 2: 新增DeepSeek融合函数

```python
def deepseek_merge_themes(data_sources: dict) -> list[Theme]:
    """
    让DeepSeek做多源数据融合
    """
    prompt = f"""
    你是资深A股策略分析师，请融合以下数据源做判断：

    ## 数据源1: Web Search热点
    {format_themes(data_sources["web_search"])}

    ## 数据源2: 龙虎榜资金流向
    {format_capital_flows(data_sources["dragon_tiger"])}

    ## 任务
    输出JSON: {{"themes": [...]}}

    每个主题包含：
    - name: 主题名称
    - validation_status: 验证状态
      - "confirmed": Web+龙虎榜都确认，主线机会
      - "web_only": Web热点但无资金验证，观望
      - "capital_only": 有资金但无热点，潜在机会
      - "weak": 两者都弱，不关注
    - evidence: 整合后的证据（Web + 龙虎榜数据）
    - capital_signal: 资金信号总结
    """

    result = deepseek_chat(prompt)
    return parse_themes(result)
```

### Step 3: 更新CLAUDE.md文档

添加 `top_list` schema 到 `CLAUDE.md`

---

## 6. 预期效果

### 6.1 市场情报更准确

- **Before**: "半导体板块活跃" (web search可能夸大/滞后)
- **After**: "半导体近30日上榜45次，净买入150亿，北上+机构持续流入，CONFIRMED主线"

### 6.2 选股更精准

- **Before**: 技术形态好的标的可能不在热门题材
- **After**: 在CONFIRMED热门题材内，找"待时机"标的，提前布局

### 6.3 报告更有说服力

- **Before**: "龙虎榜显示机构关注" (模糊)
- **After**: "板块资金加速流入，近30日净买入150亿，北上资金占比53%" (精确)

---

## 7. 优先级建议

| 优先级 | 任务 | 预计工作量 | ROI |
|--------|------|------------|-----|
| **P0** | DragonTigerList类实现 | 2h | 高 |
| **P0** | Phase 1 DeepSeek融合 | 3h | 高 |
| **P1** | Phase 2 在题材内筛选 | 2h | 中 |
| **P2** | Phase 5 报告增强 | 2h | 中 |
| **P3** | 自动同步脚本 | 1h | 低 |
