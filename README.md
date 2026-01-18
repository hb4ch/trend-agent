# Trend Agent（A股趋势跟踪 & 游资审计选股）

一个将 **“重势、通过滤、待时机”** 固化为流水线的 A 股选股/研报工具：

1. **定方向**：联网检索 → 提炼当前市场主线白名单（3–5 个题材）。
2. **选标的**：本地 Parquet 数据 → 挖掘“120 日箱体横盘 + 均线粘合 + 温和放量”的候选。
3. **做尽调**：审计式检索 → 验真题材、排除雷区（时效性监管信息、立案/诉讼/减持等）。
4. **交付**：K 线图（标注异动）+ Markdown/PDF 研报（含信源链接）。

> 免责声明：仅供研究与学习，不构成投资建议。

---

## 特性

- **本地数据为主**：读取 `data/` 下的 Parquet（见 `CLAUDE.md` 的 Schema）。
- **强“主线白名单”**：先定主线，再做筛选；题材不匹配会被降权/剔除（可启用语义匹配）。
- **“审计级排雷”**：
  - 立案调查 / 退市风险 / 重大诉讼 / 巨额减持等：一票否决。
  - 监管函/问询函/处罚：引入时效性（默认仅近 `REGULATORY_MAX_AGE_DAYS` 天有效）。
- **可视化 & 报告**：`mplfinance` 绘图，`pandoc + xelatex` 输出 PDF。

---

## 目录结构

```
.
├── trend_agent.py              # 主流水线（Phase 1-5）
├── screen_growth_stocks.py     # 技术面/市值过滤 + 初筛宽名单
├── deep_researcher.py          # Zhipu 原生 web_search + DeepSeek/Qwen（SiliconFlow）
├── CLAUDE.md                   # 本地数据 Schema（Parquet 字段）
├── data -> ../tushare-stock-select/data   # 你的本地数据（符号链接）
├── charts/                     # 输出 K 线图（默认忽略）
└── reports/                    # 输出报告与 trace（默认忽略）
```

---

## 环境准备

### 1) Python venv

项目默认使用本地 `.venv`：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖（按你环境现有依赖为准）：

```bash
pip install -U pandas numpy duckdb mplfinance matplotlib langchain-core langchain-community python-dotenv
```

### 2) 系统依赖（PDF）

需要 `pandoc` 与 `xelatex`（用于中文 PDF）：

```bash
pandoc --version
xelatex --version
```

若缺失，请按你的系统包管理器安装（如 `texlive-xetex`）。

---

## 配置（.env）

在项目根目录创建 `.env`（已在 `.gitignore` 中忽略）：

### Zhipu（联网检索）

```env
ZHIPUAI_API_KEY=xxx
DEBUG_ZHIPU_SEARCH=0
```

### SiliconFlow（DeepSeek/Qwen，同一 endpoint）

```env
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=sk-xxx

DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=Pro/deepseek-ai/DeepSeek-V3.2

QWEN_MODEL=Qwen/Qwen3-8B
USE_QWEN_THEME_MATCH=1

DEBUG_DEEPSEEK=0
DEBUG_SILICONFLOW=0
```

### 策略参数（可选）

```env
REGULATORY_MAX_AGE_DAYS=730
```

---

## 运行

```bash
source .venv/bin/activate
python trend_agent.py
```

输出：

- `charts/{ts_code}.png`
- `reports/report_YYYYMMDD_HHMMSS.md`
- `reports/report_YYYYMMDD_HHMMSS.pdf`（若系统已安装 `pandoc + xelatex`）
- `reports/audit_trace_*.jsonl`、`reports/deepseek_trace_*.jsonl`（debug/回放用）

---

## 调试

打开更多日志：

```env
DEBUG_DEEPSEEK=1
DEBUG_ZHIPU_SEARCH=1
DEBUG_SILICONFLOW=1
```

---

## 脚本说明

- `trend_agent.py`
  - Phase 1：市场主线（搜索→归纳）
  - Phase 2：技术面初筛（DuckDB + 形态/量能过滤；可选 Qwen 语义题材归属）
  - Phase 3：审计式尽调（带 cninfo 优先检索、时效性监管判断、trace）
  - Phase 4：绘图（K线+均线+量能/换手异动标注）
  - Phase 5：研报（DeepSeek 写作 + 工具调用；失败则 fallback）
- `screen_growth_stocks.py`
  - 120 日箱体振幅、MA20/60/120 粘合度与稳定性、量能倍数等打分
- `deep_researcher.py`
  - Zhipu 原生 `web_search.web_search()` 输出结构化 results
  - SiliconFlow OpenAI 兼容 `chat/completions`：DeepSeek/Qwen

---

## Roadmap（可选）

- 更精确的“题材归属”：把公司业务信息 + cninfo 公告摘要 → 结构化标签。
- 更严格的“点火信号”：结合涨停/连板/龙虎榜本地数据（若你有落库）。
- 报告模板升级：更明确的“待时机”入场条件、风险分层、备选路径。

