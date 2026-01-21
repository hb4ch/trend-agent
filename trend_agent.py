#!/usr/bin/env python3
"""
Trend Agent - A-share stock research and screening pipeline.

Implements a 5-phase pipeline:
1. Market Intelligence - Extract market themes
2. Quantitative Mining - Screen stocks by technical criteria
3. Deep Research - Audit and due diligence
4. Visualization - Generate charts
5. Report Generation - Create research reports
"""

import io
import json
import logging
import os
import re
import subprocess
import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import matplotlib as mpl
from matplotlib import font_manager
import mplfinance as mpf
import numpy as np
import pandas as pd
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deep_researcher import deepseek_chat, deepseek_plan_queries, qwen_chat, zhipu_search
from screen_growth_stocks import screen_all_stocks
from utils import (
    DragonTigerList,
    extract_urls,
    is_recent,
    normalize_verdict,
    parse_result_date,
    pretty_print,
    safe_json_loads,
    setup_logging,
    stock_symbol,
    trace_append,
    truncate,
)
from llm_provider import get_llm_provider, LLMProvider

# Setup logging
logger = logging.getLogger(__name__)


DATA_ROOT = Path("data")
CHART_DIR = Path("charts")
REPORT_DIR = Path("reports")
CHART_DAYS = 240
CHART_FONT = "Source Han Sans CN"
CHART_FONT_FALLBACKS = [
    "Source Han Sans CN",
    "思源黑体 CN",
    "Noto Sans CJK SC",
    "Noto Sans CJK",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Arial Unicode MS",
]

# Configuration from environment
DEBUG_DEEPSEEK = os.environ.get("DEBUG_DEEPSEEK", "").strip() in {"1", "true", "True", "YES", "yes"}
USE_QWEN_THEME_MATCH = os.environ.get("USE_QWEN_THEME_MATCH", "").strip() in {"1", "true", "True", "YES", "yes"}
REGULATORY_MAX_AGE_DAYS = int(os.environ.get("REGULATORY_MAX_AGE_DAYS", "730"))

def setup_matplotlib_chinese_fonts() -> None:
    """Configure Chinese fonts for matplotlib and mplfinance."""
    candidate_font_files = [
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Normal.otf",
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Light.otf",
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Heavy.otf",
    ]
    for file_path in candidate_font_files:
        try:
            if Path(file_path).exists():
                font_manager.fontManager.addfont(file_path)
        except Exception:
            pass

    # Configure matplotlib for Chinese text
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = CHART_FONT_FALLBACKS
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.figsize"] = (16, 10)

    # mplfinance uses matplotlib's rcParams internally, so the above settings
    # should apply. However, mplfinance's built-in styles override some settings.
    # We need to explicitly override the style's font configuration.
    try:
        from mplfinance import _styles
        # Modify the base styles dictionary directly
        if hasattr(_styles, "_base_styles"):
            for style_name in _styles._base_styles:
                _styles._base_styles[style_name]["font.family"] = CHART_FONT
                _styles._base_styles[style_name]["font.unicode_minus"] = False
    except Exception:
        pass


setup_matplotlib_chinese_fonts()


@dataclass
class ThemeItem:
    """Market theme with keywords and sources."""

    name: str
    keywords: List[str]
    summary: str
    sources: List[str]
    validation_status: str = "unknown"  # confirmed/web_only/capital_only/weak
    capital_signal: str = ""  # Capital flow summary from Dragon Tiger List
    evidence: str = ""  # Combined evidence from web + capital


@dataclass
class AuditResult:
    """Audit result for a stock-theme pair."""

    ts_code: str
    name: str
    theme: str
    verdict: str
    rationale: str
    sources: List[str]


def run_search(query: str) -> str:
    """Execute web search using Zhipu AI."""
    if hasattr(zhipu_search, "invoke"):
        return zhipu_search.invoke(query)
    if callable(zhipu_search):
        return zhipu_search(query)
    return "❌ Search tool unavailable."


def parse_search_payload(raw: Any) -> Dict[str, Any]:
    """
    Parse search result payload into structured format.

    Args:
        raw: Raw search result (dict, JSON string, or other)

    Returns:
        Dict with summary, results list, and extracted URLs
    """
    if raw is None:
        return {"summary": "", "results": [], "urls": []}

    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw)
        payload = safe_json_loads(text)
        if not payload:
            urls = extract_urls(text)
            return {"summary": text, "results": [], "urls": urls}

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        results = []

    urls: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)

    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        summary = ""

    return {"summary": summary, "results": results, "urls": urls}


def qwen_match_themes(
    themes: List[ThemeItem], candidates: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """
    Match stocks to themes using Qwen LLM for semantic understanding.

    Args:
        themes: List of market themes to match against
        candidates: DataFrame of candidate stocks
        cache_path: Path to cache file for results

    Returns:
        DataFrame with additional 'matched_themes' column
    """
    if not USE_QWEN_THEME_MATCH or not themes or candidates.empty or not qwen_chat:
        return candidates

    cache: Dict[str, Any] = {}
    try:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load theme match cache: {e}")
        cache = {}

    theme_list = [
        {"name": t.name, "keywords": t.keywords, "summary": t.summary}
        for t in themes
        if t.name
    ]
    if not theme_list:
        return candidates

    def row_fingerprint(row: pd.Series) -> str:
        """Create fingerprint for caching theme matches."""
        from utils import create_row_fingerprint

        return create_row_fingerprint(
            row.to_dict(),
            ["name", "industry", "main_business", "business_scope", "introduction"],
        )

    def get_text(row: pd.Series) -> str:
        """Get text representation of stock for theme matching."""
        return " | ".join(
            [
                f"name={row.get('name','')}",
                f"code={row.get('ts_code','')}",
                f"industry={row.get('industry','')}",
                f"main_business={str(row.get('main_business',''))[:200]}",
                f"business_scope={str(row.get('business_scope',''))[:200]}",
                f"intro={str(row.get('introduction',''))[:200]}",
            ]
        )

    updated = candidates.copy()
    matched_map: Dict[str, List[str]] = {}

    batch = []
    batch_keys = []
    for _, row in updated.iterrows():
        key = row_fingerprint(row)
        if key in cache:
            val = cache.get(key)
            if isinstance(val, dict) and isinstance(val.get("matched"), list):
                matched_map[row["ts_code"]] = [
                    str(x) for x in val["matched"] if str(x) in [t["name"] for t in theme_list]
                ]
            continue
        batch.append({"ts_code": row["ts_code"], "text": get_text(row)})
        batch_keys.append(key)

    def flush(batch_items: List[dict], batch_fp: List[str]) -> None:
        """Flush batch of stocks to LLM for theme matching."""
        if not batch_items:
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "你是A股题材归属分类器。给定题材白名单和股票业务简介，判断该股票是否属于白名单题材。"
                    "只允许从白名单里选择0-2个题材；不确定就返回空数组。输出严格JSON："
                    '{"matches":{"ts_code":["theme1","theme2"]},"notes":{"ts_code":"reason"}}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"themes": theme_list, "stocks": batch_items},
                    ensure_ascii=False,
                ),
            },
        ]

        content = qwen_chat(messages) if qwen_chat else None
        parsed = safe_json_loads(content or "")
        matches = parsed.get("matches", {}) if isinstance(parsed, dict) else {}
        notes = parsed.get("notes", {}) if isinstance(parsed, dict) else {}

        for fp, item in zip(batch_fp, batch_items):
            ts_code = item.get("ts_code")
            picked = matches.get(ts_code, []) if isinstance(matches, dict) else []
            if not isinstance(picked, list):
                picked = []
            picked = [str(x) for x in picked if str(x) in [t["name"] for t in theme_list]]
            matched_map[ts_code] = picked
            cache[fp] = {
                "matched": picked,
                "note": notes.get(ts_code) if isinstance(notes, dict) else ""
            }

    # Process in batches
    chunk = []
    chunk_keys = []
    for item, fp in zip(batch, batch_keys):
        chunk.append(item)
        chunk_keys.append(fp)
        if len(chunk) >= 8:
            flush(chunk, chunk_keys)
            chunk = []
            chunk_keys = []
    flush(chunk, chunk_keys)

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, IOError) as e:
        logger.warning(f"Failed to save theme match cache: {e}")

    updated["matched_themes"] = updated["ts_code"].map(lambda c: matched_map.get(c, []))
    return updated


# Constants for theme expansion and filtering
PRIMARY_SOURCE_DOMAINS = (
    "cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
)

THEME_SYNONYMS: Dict[str, List[str]] = {
    "芯片": ["半导体", "集成电路", "国产替代", "IC", "EDA", "光刻", "存储"],
    "半导体": ["芯片", "集成电路", "国产替代", "IC", "EDA", "光刻", "存储"],
    "医药": ["医疗", "创新药", "生物制药", "疫苗", "CRO", "医药生物", "生物", "诊断", "IVD", "体外诊断"],
    "医疗": ["医药", "创新药", "生物制药", "疫苗", "CRO", "医药生物", "生物", "诊断", "IVD", "体外诊断"],
    "生物": ["医药", "医疗", "创新药", "生物制药", "疫苗", "CRO", "医药生物", "诊断", "IVD"],
    "航天": ["军工", "航空", "商业航天", "军工电子", "航材", "卫星", "航空航天", "航天电子", "中国卫通"],
    "航空": ["军工", "航天", "商业航天", "军工电子", "航材", "卫星", "航空航天"],
    "商业航天": ["军工", "航天", "航空", "军工电子", "航材", "卫星", "航空航天", "航天电子"],
    "军工": ["航空", "航天", "军工电子", "航材", "兵装", "武器", "商业航天", "卫星"],
    "AI": ["人工智能", "算力", "大模型", "数据中心", "服务器", "光模块", "CPO", "AI应用"],
    "科技": ["AI", "人工智能", "算力", "大模型", "芯片", "半导体", "科技成长"],
    "新能源": ["光伏", "风电", "锂电", "储能", "新能源车", "充电桩", "电池"],
    "消费": ["零售", "食品饮料", "白酒", "家电", "旅游", "酒店"],
    "券商": ["证券", "非银金融", "保险"],
}


def expand_theme_keywords(theme: ThemeItem) -> List[str]:
    """
    Expand theme keywords with synonyms.

    Args:
        theme: Theme item with name and keywords

    Returns:
        Expanded list of keywords including synonyms
    """
    keywords = set([theme.name] + list(theme.keywords or []))
    expanded = set()
    for kw in list(keywords):
        expanded.add(kw)
        # Check if any THEME_SYNONYMS key is in this keyword
        for key, syns in THEME_SYNONYMS.items():
            if key and key in kw:
                for s in syns:
                    expanded.add(s)
        # Check if this keyword is a THEME_SYNONYMS key
        if kw in THEME_SYNONYMS:
            for s in THEME_SYNONYMS[kw]:
                expanded.add(s)
        # Check if this keyword matches any synonym value (reverse lookup)
        for syns in THEME_SYNONYMS.values():
            if kw in syns:
                # Add all related terms from this synonym group
                for s in syns:
                    expanded.add(s)
    return [k for k in expanded if isinstance(k, str) and k.strip()]


def is_relevant_search_hit(name: str, symbol: str, hit: dict) -> bool:
    title = hit.get("title") if isinstance(hit.get("title"), str) else ""
    snippet = hit.get("snippet") if isinstance(hit.get("snippet"), str) else ""
    url = hit.get("url") if isinstance(hit.get("url"), str) else ""
    text = f"{title}\n{snippet}\n{url}"
    name = name.strip()
    symbol = symbol.strip()
    if name and name in text:
        return True
    if symbol and symbol in text:
        return True
    return False


def local_brief_for_audit(row: pd.Series) -> str:
    parts = [
        f"ts_code={row.get('ts_code','')}",
        f"name={row.get('name','')}",
        f"industry={row.get('industry','')}",
        f"market_cap={row.get('market_cap', '')}",
        f"consolidation_score={row.get('consolidation_score','')}",
        f"volume_boost={row.get('volume_boost','')}",
        f"ma_spread={row.get('ma_spread','')}",
        f"ma_spread_std={row.get('ma_spread_std','')}",
        f"main_business={str(row.get('main_business',''))[:200]}",
        f"business_scope={str(row.get('business_scope',''))[:200]}",
    ]
    return " | ".join(parts)


def run_python(code: str, context: Dict) -> str:
    """
    Execute Python code in a restricted environment.

    Note: This function executes arbitrary code. Use with caution.

    Args:
        code: Python code to execute
        context: Variables to inject into execution context

    Returns:
        Output from code execution or error message
    """
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Restrict imports to safe modules only."""
        allowed = {
            "pandas",
            "numpy",
            "duckdb",
            "math",
            "datetime",
            "re",
            "json",
        }
        if name in allowed:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"import not allowed: {name}")

    safe_builtins = {
        "__import__": safe_import,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "any": any,
        "all": all,
        "print": print,
        "str": str,
        "int": int,
        "float": float,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
    }

    local_ctx = dict(context)
    local_ctx["context"] = dict(context)
    local_ctx.setdefault("pd", pd)
    local_ctx.setdefault("np", np)
    local_ctx.setdefault("duckdb", duckdb)
    local_ctx["__builtins__"] = safe_builtins

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, local_ctx, local_ctx)
    except Exception as exc:
        logger.debug(f"Python execution error: {exc}")
        return f"python_error: {exc}"

    output = stdout.getvalue().strip()
    result = local_ctx.get("result")
    if result is not None:
        return f"{output}\nresult: {result}".strip()
    return output or "ok"


def run_duckdb_sql(sql: str, context: Dict[str, pd.DataFrame]) -> str:
    """
    Execute DuckDB SQL query on registered DataFrames.

    Args:
        sql: SQL query to execute
        context: Dictionary of DataFrame names to DataFrames

    Returns:
        Query results as markdown table or error message
    """
    con = duckdb.connect()
    for name, df in context.items():
        con.register(name, df)

    try:
        df = con.execute(sql).df()
        return df.head(20).to_markdown(index=False)
    except Exception as exc:
        logger.warning(f"DuckDB query failed: {exc}")
        return f"duckdb_error: {exc}"


def init_llm() -> ChatZhipuAI:
    return ChatZhipuAI(model="glm-4-flash", temperature=0.2, timeout=300)


def deepseek_merge_themes(
    web_themes: List[ThemeItem],
    capital_themes: Dict[str, Dict],
    llm_provider: LLMProvider,
) -> List[ThemeItem]:
    """
    Use DeepSeek to merge web search and Dragon Tiger List data.

    This is the key multi-source fusion function that combines:
    - Web Search: News sentiment, policy catalysts, market narratives
    - Dragon Tiger List: Real capital flows, hot money preferences, institutional activity

    Args:
        web_themes: Themes from web search (phase 1a)
        capital_themes: Themes from Dragon Tiger List analysis
        llm_provider: LLM provider for DeepSeek

    Returns:
        List of merged ThemeItem with validation_status and capital_signal
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    logger.info("Starting multi-source theme fusion with DeepSeek")

    # Format capital themes for LLM
    capital_summary = []
    for theme_name, stats in capital_themes.items():
        capital_summary.append(
            f"""
**{theme_name}**
- 上榜次数: {stats['hit_count']}次
- 累计净买入: {stats['net_buy'] / 1e8:.2f}亿元
- 热门股票: {', '.join(stats['hot_stocks'][:5])}
- 资金结构: 北上{stats['institution_mix']['north']*100:.0f}% + 机构{stats['institution_mix']['inst']*100:.0f}% + 游资{stats['institution_mix']['hot_money']*100:.0f}%
- 趋势: {stats['trend']}
"""
        )

    # Format web themes for LLM
    web_summary = []
    for theme in web_themes:
        web_summary.append(
            f"""
**{theme.name}**
- 关键词: {', '.join(theme.keywords)}
- 摘要: {theme.summary}
- 来源: {len(theme.sources)}个相关网页
"""
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是资深A股策略分析师，擅长多源数据融合。请综合Web Search热点和龙虎榜资金流向，判断真正的市场主线。",
            ),
            (
                "user",
                """## 数据源1: Web Search热点（新闻情绪、政策催化）
{web_summary}

## 数据源2: 龙虎榜资金流向（真实行为、游资偏好）
{capital_summary}

## 融合原则
1. **confirmed (主线题材)**: Web+龙虎榜都确认，重点布局
2. **web_only (观察中)**: Web热点但无资金验证，需等待资金入场
3. **capital_only (潜在机会)**: 有资金但无热点，深入挖掘背后的逻辑
4. **weak (不关注)**: 两者都弱，不关注

## 输出要求
输出JSON格式，包含融合后的主题列表：
```json
{{
  "themes": [
    {{
      "name": "主题名称",
      "validation_status": "confirmed|web_only|capital_only|weak",
      "keywords": ["关键词1", "关键词2"],
      "summary": "整合后的证据（Web + 龙虎榜数据综合）",
      "capital_signal": "资金信号总结（上榜次数、净买入、资金结构、趋势）",
      "sources": ["url1", "url2"]
    }}
  ]
}}
```

注意：
- validation_status只选择4个值之一
- summary要综合Web和龙虎榜双方面信息
- capital_signal重点描述资金行为和趋势
- 只保留最重要的3-5个主题
""",
            ),
        ]
    )

    chain = prompt | llm_provider.get_llm("deepseek", temperature=0.2) | StrOutputParser()
    result = chain.invoke({
        "web_summary": "\n".join(web_summary),
        "capital_summary": "\n".join(capital_summary) if capital_summary else "暂无龙虎榜数据",
    })

    logger.debug(f"DeepSeek fusion raw result: {truncate(result, 2000)}")

    # Parse result
    data = safe_json_loads(result)
    logger.debug(f"DeepSeek fusion parsed data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
    logger.debug(f"DeepSeek fusion themes count: {len(data.get('themes', [])) if isinstance(data, dict) else 0}")

    merged_themes = []

    # Map to keep sources from original web themes
    sources_map = {t.name: t.sources for t in web_themes}

    for item in data.get("themes", []):
        name = item.get("name", "").strip()
        if not name:
            continue

        # Use sources from original web theme if available
        sources = item.get("sources", [])
        if not sources and name in sources_map:
            sources = sources_map[name]

        merged_themes.append(
            ThemeItem(
                name=name,
                keywords=[kw.strip() for kw in item.get("keywords", []) if kw.strip()],
                summary=item.get("summary", "").strip(),
                sources=sources,
                validation_status=item.get("validation_status", "unknown"),
                capital_signal=item.get("capital_signal", ""),
                evidence=item.get("summary", ""),
            )
        )

    logger.info(f"Multi-source fusion complete: {len(merged_themes)} themes")
    for theme in merged_themes:
        logger.info(f"  - {theme.name}: {theme.validation_status} | {theme.capital_signal[:50]}...")

    # Fallback: if no themes returned, use web themes with default validation
    if not merged_themes:
        logger.warning("DeepSeek fusion returned 0 themes, falling back to web themes")
        for theme in web_themes:
            theme.validation_status = "web_only"
            theme.capital_signal = "暂无龙虎榜数据验证"
            theme.evidence = theme.summary
        return web_themes

    return merged_themes


def phase1_market_intel(llm: ChatZhipuAI) -> List[ThemeItem]:
    """
    Phase 1: Extract market themes from web search.

    Args:
        llm: Language model for processing search results

    Returns:
        List of market themes with keywords and sources
    """
    logger.info("Starting Phase 1: Market Intelligence")

    current_year_month = datetime.now().strftime("%Y年%m月")
    queries = [
        f"A股 {current_year_month} 核心题材 最新热点",
        f"龙虎榜 {current_year_month} 机构游资 重点板块 最新动向",
        f"A股 {current_year_month} 涨停复盘 市场热点",
    ]

    raw_results = []
    all_urls = []
    for query in queries:
        logger.debug(f"Searching: {query}")
        raw = run_search(query)
        parsed = parse_search_payload(raw)
        urls = parsed.get("urls", [])
        raw_results.append(
            {
                "query": query,
                "summary": parsed.get("summary", ""),
                "results": parsed.get("results", []),
                "urls": urls,
            }
        )
        for url in urls if isinstance(urls, list) else []:
            if url not in all_urls:
                all_urls.append(url)

    logger.debug(f"Collected {len(all_urls)} unique URLs from searches")

    # Use DeepSeek for theme extraction (to avoid Zhipu rate limits)
    llm_provider = get_llm_provider()
    deepseek_llm = llm_provider.get_llm("deepseek", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是A股游资策略师，遵循\"重势重质\"原则，提炼当前市场3-5个核心主线并给出关键词。",
            ),
            (
                "user",
                "基于搜索结果，输出JSON："
                '{{"themes":[{{"name":"","keywords":["",""],"summary":"","sources":["url1"]}}],'
                '"market_summary":""}}\n\n搜索结果:\n{results}',
            ),
        ]
    )

    chain = prompt | deepseek_llm | StrOutputParser()
    result = chain.invoke({"results": json.dumps(raw_results, ensure_ascii=False)})
    data = safe_json_loads(result)

    themes = []
    for item in data.get("themes", [])[:5]:
        sources = item.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        sources = [s for s in sources if isinstance(s, str)]
        sources = [s for s in sources if s in all_urls]
        if not sources:
            sources = all_urls[:3]
        themes.append(
            ThemeItem(
                name=item.get("name", "").strip(),
                keywords=[kw.strip() for kw in item.get("keywords", []) if kw.strip()],
                summary=item.get("summary", "").strip(),
                sources=sources,
            )
        )

    def is_actionable_theme(theme: ThemeItem) -> bool:
        """Check if theme is actionable (not generic noise)."""
        if not theme.name:
            return False
        bad_tokens = ["股票", "辨识度", "传统经济", "蓝筹", "龙虎榜", "游资", "机构", "复盘", "涨停"]
        if any(tok in theme.name for tok in bad_tokens):
            return False
        if len(theme.name) > 12:
            return False
        return True

    filtered = [t for t in themes if is_actionable_theme(t)]
    web_themes = filtered or themes
    logger.info(f"Web search extracted {len(web_themes)} market themes")

    # Phase 1b: Load Dragon Tiger List data for capital flow analysis
    logger.info("Loading Dragon Tiger List data for multi-source fusion...")
    dtl = DragonTigerList()
    capital_themes = dtl.identify_hot_themes(days=30)
    logger.info(f"Dragon Tiger List identified {len(capital_themes)} themes")

    # Phase 1c: Multi-source fusion with DeepSeek
    logger.info("Merging web search and Dragon Tiger List themes...")
    llm_provider = get_llm_provider()
    merged_themes = deepseek_merge_themes(web_themes, capital_themes, llm_provider)

    return merged_themes


def phase2_quant_filter(themes: List[ThemeItem]) -> pd.DataFrame:
    """
    Phase 2: Screen stocks using quantitative technical criteria.

    Args:
        themes: List of market themes for matching (should have validation_status)

    Returns:
        DataFrame of filtered candidate stocks
    """
    logger.info("Starting Phase 2: Quantitative Filtering")

    # Filter to only confirmed themes (web + capital both confirmed)
    confirmed_themes = [t for t in themes if t.validation_status == "confirmed"]
    web_only_themes = [t for t in themes if t.validation_status == "web_only"]

    # If no confirmed themes, fall back to web_only
    if not confirmed_themes and web_only_themes:
        logger.warning("No confirmed themes, falling back to web_only themes")
        confirmed_themes = web_only_themes

    if not confirmed_themes:
        logger.warning("No confirmed or web_only themes, using all themes")
        confirmed_themes = themes

    logger.info(f"Phase 2 filtering with {len(confirmed_themes)} confirmed themes: {[t.name for t in confirmed_themes]}")

    # Get stocks to exclude (recently on Dragon Tiger List)
    dtl = DragonTigerList()
    exclude_stocks = set(dtl.get_recent_toplist_stocks(days=60))
    logger.info(f"Excluding {len(exclude_stocks)} stocks recently on Dragon Tiger List")

    screen_df = screen_all_stocks()
    if screen_df is None or screen_df.empty:
        logger.warning("No screening results from screen_all_stocks()")
        return pd.DataFrame()

    # Exclude stocks recently on toplist
    if exclude_stocks and "ts_code" in screen_df.columns:
        before_count = len(screen_df)
        screen_df = screen_df[~screen_df["ts_code"].isin(exclude_stocks)]
        logger.info(f"Excluded {before_count - len(screen_df)} stocks recently on Dragon Tiger List")

    con = duckdb.connect()
    con.register("screen", screen_df)
    filtered = con.execute(
        """
        SELECT *
        FROM screen
        WHERE consolidation_score >= 70
          AND ma_spread <= 0.15
          AND ma_spread_std <= 0.03
          AND volume_boost >= 1.2
          AND volume_boost <= 3.0
        ORDER BY composite_score DESC
        """
    ).df()

    logger.info(f"Filtered to {len(filtered)} candidates by technical criteria")

    # Optional: use Qwen (small LLM) to match themes semantically
    filtered = qwen_match_themes(confirmed_themes, filtered, cache_path=Path(".cache/qwen_theme_match.json"))
    if "matched_themes" in filtered.columns:
        filtered = filtered.rename(columns={"matched_themes": "matched_themes_llm"})

    theme_keywords = []
    for theme in confirmed_themes:
        expanded = expand_theme_keywords(theme)
        if theme.name and expanded:
            theme_keywords.append((theme.name, expanded))

    def match_row(row: pd.Series) -> Dict[str, List[str]]:
        """Match row to themes by keyword search."""
        text = " ".join(
            [
                str(row.get("name", "")),
                str(row.get("industry", "")),
                str(row.get("main_business", "")),
                str(row.get("business_scope", "")),
                str(row.get("introduction", "")),
            ]
        )
        matches = []
        for theme_name, keywords in theme_keywords:
            if any(kw in text for kw in keywords):
                matches.append(theme_name)
        return {"matched_themes": matches}

    if theme_keywords:
        filtered["matched_themes_kw"] = filtered.apply(
            lambda row: match_row(row)["matched_themes"], axis=1
        )
    else:
        filtered["matched_themes_kw"] = [[] for _ in range(len(filtered))]

    def merge_matches(row: pd.Series) -> List[str]:
        """Merge LLM and keyword matches."""
        merged = []
        for key in ("matched_themes_llm", "matched_themes_kw"):
            val = row.get(key, [])
            if not isinstance(val, list):
                continue
            for item in val:
                item = str(item).strip()
                if item and item not in merged:
                    merged.append(item)
        return merged

    filtered["matched_themes"] = filtered.apply(merge_matches, axis=1)

    has_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
    if has_match == 0:
        logger.warning(
            "No candidates match current themes, returning top 15 by technical score (off_theme=True)"
        )
        filtered["off_theme"] = True
        return filtered.head(15)

    filtered = filtered[filtered["matched_themes"].apply(bool)]
    filtered["off_theme"] = False
    logger.info(f"Phase 2 complete: {len(filtered.head(15))} candidates with theme matches")
    return filtered.head(15)


def apply_audit_filter(
    candidates: pd.DataFrame, audits: List[AuditResult]
) -> tuple[pd.DataFrame, List[AuditResult]]:
    """
    Filter candidates based on audit results.

    Args:
        candidates: DataFrame of candidate stocks
        audits: List of audit results

    Returns:
        Tuple of (filtered candidates, filtered audits)
    """
    verdict_rank = {"fail": 2, "warn": 1, "pass": 0}
    worst = {}
    for audit in audits:
        rank = verdict_rank.get(audit.verdict, 1)
        prev = worst.get(audit.ts_code, -1)
        if rank > prev:
            worst[audit.ts_code] = rank

    def verdict_label(rank: int) -> str:
        """Convert rank to verdict label."""
        for key, val in verdict_rank.items():
            if val == rank:
                return key
        return "warn"

    candidates = candidates.copy()
    candidates["audit_verdict"] = candidates["ts_code"].map(
        lambda code: verdict_label(worst.get(code, 1))
    )

    if "audit_verdict" in candidates.columns:
        counts = candidates["audit_verdict"].value_counts().to_dict()
        logger.info(f"Audit verdict distribution: {counts}")

    filtered = candidates[candidates["audit_verdict"] != "fail"]
    allowed_codes = set(filtered["ts_code"])
    filtered_audits = [audit for audit in audits if audit.ts_code in allowed_codes]

    logger.info(f"Applied audit filter: {len(filtered)} candidates passed")
    return filtered, filtered_audits


def phase3_deep_audit(
    llm: ChatZhipuAI, candidates: pd.DataFrame, trace_path: Optional[Path] = None
) -> List[AuditResult]:
    top = candidates.head(15)
    audit_results: List[AuditResult] = []
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是A股审计级尽调员（红蓝军对抗），必须严守一票否决：立案调查/未结重大诉讼/巨额减持/伪概念（无订单无客户）= fail。",
            ),
            (
                "user",
                "给定搜索摘要，输出JSON："
                '{{"verdict":"pass|warn|fail","rationale":"","sources":["url1"]}}\n\n'
                "股票:{name} 题材:{theme}\n搜索结果:\n{results}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    for _, row in top.iterrows():
        name = row["name"]
        symbol = stock_symbol(row["ts_code"])
        themes = row.get("matched_themes", []) or ["未明确题材"]
        for theme in themes:
            logger.debug(f"Audit start: stock={name} theme={theme}")
            trace_append(trace_path, "audit_start", {"ts_code": row["ts_code"], "name": name, "theme": theme})
            merged = {}
            evidence_snippets = [f"[local]\n{local_brief_for_audit(row)}"]
            used_queries = set()
            verdict = "warn"
            rationale = ""
            sources = []

            hard_fail_patterns = [
                re.compile(r"(被|遭|因|涉嫌).{0,12}(立案|立案调查)"),
                re.compile(r"(重大诉讼|未决诉讼|仲裁|诉讼事项)"),
                re.compile(r"(拟|计划).{0,12}减持|减持计划"),
                re.compile(r"(终止上市|退市风险警示|暂停上市|强制退市)"),
            ]
            severe_regulatory_patterns = [
                re.compile(r"(行政处罚|处罚决定书|纪律处分|公开谴责|市场禁入)"),
            ]
            minor_regulatory_patterns = [
                re.compile(r"(监管函|问询函|关注函|责令改正|监管措施决定书)"),
            ]
            positive_terms = [
                "订单",
                "中标",
                "客户",
                "签约",
                "签署",
                "签订",
                "合同",
                "协议",
                "合作",
                "供货",
                "落地",
                "框架协议",
            ]
            executed_passes = 0
            for pass_id in range(1, 4):
                if pass_id == 1:
                    theme_hint = "" if theme in {"未明确题材", "未匹配主线"} else f" {theme}"
                    queries = [
                        f"{name}{theme_hint} 实锤 订单 客户 概念",
                        f"site:cninfo.com.cn {symbol} {name} 重大合同",
                        f"site:cninfo.com.cn {symbol} {name} 监管函 问询函 处罚",
                    ]
                else:
                    plan = deepseek_plan_queries(
                        name=name,
                        theme=theme,
                        evidence="\n".join(evidence_snippets[-4:])[-2000:],
                        pass_id=pass_id,
                    )
                    if plan is not None:
                        logger.debug(f"Audit plan pass={pass_id}: {truncate(pretty_print(plan), 1200)}")
                        trace_append(trace_path, "audit_plan", {"ts_code": row["ts_code"], "name": name, "theme": theme, "pass_id": pass_id, "plan": plan})
                    if plan and plan.get("stop"):
                        logger.debug(f"Audit plan stop pass={pass_id} reason={plan.get('reason','')}")
                        break
                    queries = plan.get("queries") if plan else None
                    if not queries:
                        queries = [
                            f"site:cninfo.com.cn {symbol} {name} 投资者关系 活动记录表",
                            f"site:cninfo.com.cn {symbol} {name} 中标 公告",
                            f"site:cninfo.com.cn {symbol} {name} 立案 调查",
                            f"site:cninfo.com.cn {symbol} {name} 重大诉讼 仲裁",
                            f"site:cninfo.com.cn {symbol} {name} 减持 计划",
                        ]

                for query in queries:
                    if query in used_queries:
                        continue
                    used_queries.add(query)
                    logger.debug(f"Audit search pass={pass_id} q={query}")
                    raw = run_search(query)
                    parsed = parse_search_payload(raw)
                    urls = parsed.get("urls", [])
                    search_results = parsed.get("results", [])
                    summary = parsed.get("summary", "")
                    items_text = ""
                    if isinstance(search_results, list) and search_results:
                        relevant_hits = [
                            hit
                            for hit in search_results
                            if isinstance(hit, dict) and is_relevant_search_hit(name, symbol, hit)
                        ]
                        if relevant_hits:
                            search_results = relevant_hits
                        else:
                            # Don't let strict filtering zero-out the evidence; keep top hits for planning.
                            search_results = [hit for hit in search_results if isinstance(hit, dict)]
                        parts = []
                        for item in search_results[:5]:
                            if not isinstance(item, dict):
                                continue
                            title = item.get("title") if isinstance(item.get("title"), str) else ""
                            url = item.get("url") if isinstance(item.get("url"), str) else ""
                            snippet = item.get("snippet") if isinstance(item.get("snippet"), str) else ""
                            date = item.get("date") if isinstance(item.get("date"), str) else ""
                            date_part = f" ({date})" if date else ""
                            parts.append(f"- {title}{date_part}\n  {url}\n  {snippet}".strip())
                        items_text = "\n".join(parts)
                    raw_clean = "\n".join([str(summary or "").strip(), items_text]).strip()
                    if not raw_clean:
                        url_preview = ", ".join(urls[:3]) if isinstance(urls, list) else ""
                        raw_clean = f"未找到可用摘要/结果（可能被过滤）。query={query} urls={url_preview}"
                    merged[f"pass{pass_id}_{len(merged)+1}"] = {
                        "query": query,
                        "raw": raw_clean,
                        "urls": urls,
                        "results": search_results if isinstance(search_results, list) else [],
                    }
                    trace_append(trace_path, "audit_search", {"ts_code": row["ts_code"], "name": name, "theme": theme, "pass_id": pass_id, "query": query, "urls": urls})
                    trace_append(
                        trace_path,
                        "audit_search_snippet",
                        {
                            "ts_code": row["ts_code"],
                            "name": name,
                            "theme": theme,
                            "pass_id": pass_id,
                            "query": query,
                            "raw_snippet": truncate(raw_clean, 1200),
                        },
                    )
                    evidence_snippets.append(raw_clean[:800])

                merged_text = "\n".join(
                    [str(item.get("raw", "")) for item in merged.values()]
                )
                flat_urls = []
                for item in merged.values():
                    for url in item.get("urls", []):
                        if url not in flat_urls:
                            flat_urls.append(url)

                recent_severe_reg = False
                recent_minor_reg = False
                for item in merged.values():
                    hits = item.get("results", [])
                    if not isinstance(hits, list):
                        continue
                    for hit in hits:
                        if not isinstance(hit, dict):
                            continue
                        dt = parse_result_date(hit.get("date"))
                        if not is_recent(dt, REGULATORY_MAX_AGE_DAYS):
                            continue
                        hay = " ".join(
                            [
                                str(hit.get("title", "")),
                                str(hit.get("snippet", "")),
                            ]
                        )
                        if any(p.search(hay) for p in severe_regulatory_patterns):
                            recent_severe_reg = True
                        if any(p.search(hay) for p in minor_regulatory_patterns):
                            recent_minor_reg = True

                # Hard veto: investigation / delisting / major litigation / huge reduction (not time-scoped here)
                if flat_urls and any(p.search(merged_text) for p in hard_fail_patterns):
                    verdict = "fail"
                    rationale = (
                        "触发一票否决关键词（立案/重大诉讼/巨额减持/退市风险等），直接剔除。"
                    )
                    sources = flat_urls[:5]
                    logger.debug("Audit hard fail: matched patterns")
                    trace_append(trace_path, "audit_hard_fail", {"ts_code": row["ts_code"], "name": name, "theme": theme, "sources": sources})
                    break

                # Regulatory letters/penalties: must be recent to count as veto evidence
                if recent_severe_reg:
                    verdict = "fail"
                    rationale = f"检索到近{REGULATORY_MAX_AGE_DAYS}天内的行政处罚/纪律处分等严重监管事件，按审计口径剔除。"
                    sources = flat_urls[:5]
                    trace_append(trace_path, "audit_hard_fail", {"ts_code": row["ts_code"], "name": name, "theme": theme, "sources": sources})
                    break

                output = chain.invoke(
                    {
                        "name": name,
                        "theme": theme,
                        "results": json.dumps(merged, ensure_ascii=False),
                    }
                )
                data = safe_json_loads(output)
                verdict = normalize_verdict(data.get("verdict", verdict))
                rationale = str(data.get("rationale", rationale) or "").strip()
                sources = data.get("sources", sources)
                logger.debug(f"Audit LLM verdict={verdict} rationale={truncate(rationale, 400)}")
                trace_append(trace_path, "audit_llm", {"ts_code": row["ts_code"], "name": name, "theme": theme, "verdict": verdict, "rationale": rationale, "sources": sources})
                if not sources or any(src.strip().lower() == "url1" for src in sources):
                    flat_urls = []
                    for item in merged.values():
                        for url in item.get("urls", []):
                            if url not in flat_urls:
                                flat_urls.append(url)
                    if flat_urls:
                        sources = flat_urls[:5]
                if verdict == "pass":
                    break

                executed_passes = pass_id

            final_text = "\n".join([str(item.get("raw", "")) for item in merged.values()])
            flat_urls = []
            for item in merged.values():
                for url in item.get("urls", []):
                    if url not in flat_urls:
                        flat_urls.append(url)
            primary_urls = [u for u in flat_urls if any(dom in u for dom in PRIMARY_SOURCE_DOMAINS)]
            has_positive = any(term in final_text for term in positive_terms)

            if verdict != "fail" and not has_positive:
                if not flat_urls:
                    verdict = "warn"
                    rationale = (rationale + "；检索未返回可核验URL，无法完成审计式验真，暂按存疑处理。").strip("；")
                elif executed_passes >= 2 or len(used_queries) >= 4:
                    verdict = "fail"
                    rationale = (rationale + "；多轮检索仍未找到明确订单/客户/中标等硬证据，按伪概念一票否决。").strip("；")
                else:
                    verdict = "warn"
                    rationale = (rationale + "；当前检索未找到明确订单/客户/中标等硬证据，暂按存疑处理。").strip("；")
            if verdict != "fail" and recent_minor_reg:
                rationale = (rationale + f"；检索到近{REGULATORY_MAX_AGE_DAYS}天内监管函/问询函等事项，需额外关注。").strip("；")
            if verdict == "pass" and not primary_urls:
                verdict = "warn"
                rationale = (rationale + "；缺少交易所/巨潮等一手来源链接，按审计口径降级。").strip("；")

            if not sources or any(str(src).strip().lower() == "url1" for src in sources):
                sources = (primary_urls or flat_urls)[:5]

            audit_results.append(
                AuditResult(
                    ts_code=row["ts_code"],
                    name=name,
                    theme=theme,
                    verdict=verdict,
                    rationale=rationale,
                    sources=sources,
                )
            )
            logger.debug(f"Audit done: stock={name} theme={theme} verdict={verdict}")
            trace_append(trace_path, "audit_done", {"ts_code": row["ts_code"], "name": name, "theme": theme, "verdict": verdict})

    return audit_results


def load_price_data(ts_code: str) -> pd.DataFrame:
    parquet_path = DATA_ROOT / "stock_ticks" / f"{ts_code}.parquet"
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT trade_date, open, high, low, close, vol, turnover_rate
        FROM parquet_scan(?)
        ORDER BY trade_date
        """,
        [str(parquet_path)],
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    df = df.rename(columns={"vol": "volume"})
    return df


def detect_turnover_spikes(
    df: pd.DataFrame, window: int = 20, multiple: float = 1.5
) -> pd.Series:
    if "turnover_rate" not in df.columns:
        rolling = df["volume"].rolling(window=window).mean()
        spikes = df["volume"] > (rolling * multiple)
        return spikes.fillna(False)

    rolling = df["turnover_rate"].rolling(window=window).mean()
    spikes = df["turnover_rate"] > (rolling * multiple)
    return spikes.fillna(False)


def phase4_plot_charts(candidates: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate K-line charts with technical indicators.

    Args:
        candidates: DataFrame of candidate stocks

    Returns:
        Dictionary mapping stock codes to spike dates
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_notes = {}
    setup_matplotlib_chinese_fonts()

    # Configure high DPI for better quality
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"

    for _, row in candidates.head(8).iterrows():
        ts_code = row["ts_code"]
        df = load_price_data(ts_code)
        if len(df) > CHART_DAYS:
            df = df.tail(CHART_DAYS)
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        spikes = detect_turnover_spikes(df)
        spike_dates = df.index[spikes].strftime("%Y-%m-%d").tolist()
        add_plots = [
            mpf.make_addplot(df["ma20"], color="orange"),
            mpf.make_addplot(df["ma60"], color="blue"),
            mpf.make_addplot(df["ma120"], color="purple"),
        ]
        if spikes.any():
            add_plots.append(
                mpf.make_addplot(
                    df["high"].where(spikes),
                    type="scatter",
                    markersize=50,
                    marker="^",
                    color="red",
                )
            )

        chart_path = CHART_DIR / f"{ts_code}.png"
        mpf.plot(
            df,
            type="candle",
            volume=True,
            addplot=add_plots,
            style="yahoo",
            title=f"{row['name']} {ts_code}",
            savefig=dict(
                fname=str(chart_path),
                dpi=150,
                facecolor="white",
                edgecolor="none",
                bbox_inches="tight",
                pad_inches=0.1,
            ),
            warn_too_much_data=CHART_DAYS + 10,
            figsize=(16, 10),
        )
        chart_notes[ts_code] = spike_dates
        logger.debug(f"Generated chart: {chart_path}")
    return chart_notes


def compute_signals(candidates: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    signals: Dict[str, Dict[str, object]] = {}
    for _, row in candidates.head(15).iterrows():
        ts_code = row["ts_code"]
        name = row.get("name", "")
        df = load_price_data(ts_code)
        df_120 = df.tail(120)
        if df_120.empty:
            continue
        box_top = float(df_120["high"].max())
        box_bottom = float(df_120["low"].min())
        amplitude = (box_top - box_bottom) / (box_bottom + 1e-9)
        close = float(df_120["close"].iloc[-1])
        dist_to_top = (box_top - close) / (box_top + 1e-9)
        pos = (close - box_bottom) / ((box_top - box_bottom) + 1e-9)

        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        ma120 = df["close"].rolling(120).mean()
        ma_recent = pd.concat([ma20, ma60, ma120], axis=1).tail(20).dropna()
        if ma_recent.empty:
            ma_spread_mean = None
            ma_spread_std = None
        else:
            spread = (ma_recent.max(axis=1) - ma_recent.min(axis=1)) / (ma_recent.min(axis=1) + 1e-9)
            ma_spread_mean = float(spread.mean())
            ma_spread_std = float(spread.std(ddof=0)) if len(spread) > 1 else 0.0

        if "turnover_rate" in df_120.columns and df_120["turnover_rate"].notna().any():
            base_turn = float(df_120["turnover_rate"].mean())
            recent_turn = float(df["turnover_rate"].tail(10).mean())
            turn_mult = recent_turn / (base_turn + 1e-9)
        else:
            base_vol = float(df_120["volume"].mean())
            recent_vol = float(df["volume"].tail(10).mean())
            turn_mult = recent_vol / (base_vol + 1e-9)

        ignition = 1.2 <= turn_mult <= 3.0
        ready = ignition and dist_to_top <= 0.03 and close >= float(df["close"].rolling(20).mean().iloc[-1])

        signals[ts_code] = {
            "name": name,
            "box_top": box_top,
            "box_bottom": box_bottom,
            "amplitude_120": amplitude,
            "close": close,
            "dist_to_box_top": dist_to_top,
            "close_position": pos,
            "turnover_mult": turn_mult,
            "ma_spread_mean_20": ma_spread_mean,
            "ma_spread_std_20": ma_spread_std,
            "ignition": ignition,
            "ready_to_break": ready,
        }
    return signals


def phase5_report(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    audits: List[AuditResult],
    chart_notes: Dict[str, List[str]],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = REPORT_DIR / f"report_{timestamp}.md"

    audit_map = {}
    for audit in audits:
        audit_map.setdefault(audit.ts_code, []).append(audit)

    lines = [
        "# A股趋势跟踪研报",
        "",
        "## 【市场风向标】",
    ]

    for theme in themes:
        lines.extend(
            [
                f"- **{theme.name}**：{theme.summary}",
                f"  - 关键词：{', '.join(theme.keywords)}",
                f"  - 来源：{', '.join(theme.sources)}",
            ]
        )

    lines.extend(
        [
            "",
            "## 【核心金股】",
            "",
            "| 股票 | 所属主线 | 形态特征 | 推荐理由 |",
            "| --- | --- | --- | --- |",
        ]
    )

    for _, row in candidates.head(10).iterrows():
        themes_str = ", ".join(row.get("matched_themes", [])) or "待确认"
        shape = f"横盘分{row['consolidation_score']:.0f}, 量能{row['volume_boost']:.2f}"
        reason = f"市值{row['market_cap']/1e8:.1f}亿, 换手{row['avg_turnover']:.2f}"
        lines.append(f"| {row['name']}({row['ts_code']}) | {themes_str} | {shape} | {reason} |")

    lines.append("")
    lines.append("## 【深度图解】")
    for _, row in candidates.head(8).iterrows():
        ts_code = row["ts_code"]
        name = row["name"]
        spikes = chart_notes.get(ts_code, [])
        spike_note = ", ".join(spikes) if spikes else "未检测到明显量能异动"
        chart_path = Path("..") / CHART_DIR / f"{ts_code}.png"
        lines.extend(
            [
                f"### {name} {ts_code}",
                "",
                f"- 量能异动日：{spike_note}",
                f"![{name} {ts_code}]({chart_path.as_posix()})",
                "",
            ]
        )

        for audit in audit_map.get(ts_code, []):
            lines.extend(
                [
                    f"- 尽调结论({audit.theme})：{audit.verdict}",
                    f"  - 说明：{audit.rationale}",
                    f"  - 来源：{', '.join(audit.sources)}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 【风险提示】",
            "- 题材轮动快，注意情绪退潮风险。",
            "- 量能异动需配合市场主线验证。",
            "- 若出现监管函、立案调查等硬伤，直接剔除。",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def phase5_report_with_deepseek(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    audits: List[AuditResult],
    chart_notes: Dict[str, List[str]],
    signals: Dict[str, Dict[str, object]],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = REPORT_DIR / f"report_{timestamp}.md"
    trace_path = REPORT_DIR / f"deepseek_trace_{timestamp}.jsonl"

    audit_df = pd.DataFrame([audit.__dict__ for audit in audits])
    summary = {
        "themes": [
            {
                "name": t.name,
                "keywords": t.keywords,
                "summary": t.summary,
                "sources": t.sources,
                "validation_status": t.validation_status,
                "capital_signal": t.capital_signal,
                "evidence": t.evidence,
            }
            for t in themes
        ],
        "candidates": candidates.head(12).to_dict("records"),
        "audits": audit_df.to_dict("records"),
        "chart_notes": chart_notes,
        "signals": signals,
    }

    system_prompt = (
        "你是资深A股投研团队负责人，遵循\"重势、通过滤、待时机\"理念。"
        ""
        "## 报告结构："
        ""
        "### 【市场风向标】"
        "每个主题分析："
        "- 主题名称和验证状态（confirmed/web_only/capital_only/weak）"
        "- 主题逻辑（从web search获取的新闻情绪、政策催化）"
        "- **资金验证**<font color='purple'>（从龙虎榜获取的资金信号：上榜次数、净买入额、资金结构、趋势）</font>"
        "- 持续观察指标"
        ""
        "### 【核心金股】表格"
        "列：股票、所属主线、形态特征、推荐理由（投资逻辑）"
        ""
        "### 【深度图解】（每个标的必须包含）："
        "**【投资逻辑】**<font color='blue'>"
        "- 观察现象：量能异动、技术形态、题材契合"
        "- 分析意义：资金态度、趋势方向、突破可能"
        "- 验证方式：龙虎榜、财报、公告"
        "- 结论：交易机会评级（强烈推荐/推荐/谨慎）"
        "</font>"
        ""
        "**【技术分析】**横盘时长/波动率、量能信号、均线排列、箱体位置"
        ""
        "**【资金验证】**<font color='purple'>机构游资动向、估值水平、市值适合度</font>"
        ""
        "**【核心催化】**<font color='red'>政策/事件/市场催化</font>"
        ""
        "**【交易建议】**<font color='green'>买入时机/仓位/止盈止损/持仓周期</font>"
        ""
        "**【风险提示】**<font color='orange'>核心风险及应对</font>"
        ""
        "- 量能异动日：[列表]\\n"
        "![股票名称 代码](../charts/代码.png)\\n"
        "- 尽调结论：pass/fail（说明+来源）"
        ""
        "### 【风险提示】"
        "用<font color='orange'>橙色</font>标注核心风险"
        ""
        "## 要求："
        "- 输出JSON：{\"final_report\":\"# Markdown...\"}"
        "- 引用真实URL，不使用占位符"
        "- 突出\"待时机\"：箱体上沿+温和放量(1.2-3.0x)+均线粘合"
        "- **在【市场风向标】中展示资金验证信息**：对于confirmed主题，必须展示龙虎榜资金信号"
        ""
        "记住：展示思考过程，而非仅结论。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "请生成完整研报，包含市场风向标、核心金股、深度图解、风险提示。"
                "注意引用来源URL。上下文如下：\n"
                + json.dumps(summary, ensure_ascii=False)
            ),
        },
    ]
    logger.debug(f"Report trace: {trace_path}")
    trace_append(trace_path, "report_init", {"system": truncate(system_prompt, 2000)})
    trace_append(trace_path, "report_context", {"summary_keys": list(summary.keys())})

    report_md = None
    tool_context = {
        "candidates_df": candidates,
        "audits_df": audit_df,
        "signals": signals,
        "candidates_records": candidates.head(15).to_dict("records"),
    }

    for _ in range(5):
        logger.debug(f"Report round: {len(messages)} messages")
        trace_append(trace_path, "report_request", {"messages": [{"role": m.get("role"), "content": truncate(str(m.get("content","")), 1200)} for m in messages[-3:]]})
        content = deepseek_chat(messages) if deepseek_chat else None
        if not content:
            logger.debug("Report: no response from DeepSeek")
            trace_append(trace_path, "report_no_response", {})
            break
        logger.debug(f"Report raw response: {truncate(content, 1200)}")
        trace_append(trace_path, "report_raw_response", {"content": truncate(content, 12000)})
        parsed = safe_json_loads(content)
        tool = parsed.get("tool")
        if tool:
            if tool == "web_search":
                tool_input = parsed.get("input", "")
                logger.debug(f"Report tool: web_search input={truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "web_search", "input": tool_input})
                result = run_search(tool_input)
            elif tool == "duckdb":
                tool_input = parsed.get("input", "")
                logger.debug(f"Report tool: duckdb input={truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "duckdb", "input": tool_input})
                result = run_duckdb_sql(tool_input, tool_context)
            elif tool == "python":
                tool_input = parsed.get("input", "")
                logger.debug(f"Report tool: python input={truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "python", "input": tool_input})
                result = run_python(tool_input, tool_context)
            else:
                result = "unknown_tool"
            logger.debug(f"Report tool result: {truncate(result, 800)}")
            trace_append(trace_path, "report_tool_result", {"tool": tool, "result": truncate(result, 20000)})
            messages.append({"role": "user", "content": f"TOOL_RESULT:\n{result}"})
            continue
        report_md = parsed.get("final_report")
        if report_md:
            logger.debug("Report: final received")
            trace_append(trace_path, "report_final_received", {"length": len(report_md)})
            break
        if content.lstrip().startswith("#"):
            report_md = content
            logger.debug("Report: markdown fallback")
            trace_append(trace_path, "report_markdown_fallback", {"length": len(report_md)})
            break
        messages.append({"role": "user", "content": "请按JSON格式返回。"})
        trace_append(trace_path, "report_retry", {})

    if not report_md:
        return phase5_report(themes, candidates, audits, chart_notes)

    if "url1" in report_md:
        real_urls = []
        for theme in themes:
            for url in theme.sources:
                if url not in real_urls:
                    real_urls.append(url)
        if real_urls:
            report_md = report_md.replace("url1", real_urls[0])

    md_path.write_text(report_md, encoding="utf-8")
    return md_path


def postprocess_markdown(md_path: Path) -> Path:
    """
    Post-process markdown to fix formatting issues for better PDF output.

    Args:
        md_path: Path to original Markdown file

    Returns:
        Path to processed Markdown file
    """
    import re
    from urllib.parse import urlparse

    content = md_path.read_text(encoding="utf-8")
    processed_path = md_path.parent / f"{md_path.stem}_processed.md"

    lines = content.split("\n")
    processed_lines = []

    for line in lines:
        # Fix long comma-separated date lists - break them into multiple lines
        if "量能异动日：" in line and "," in line:
            match = re.search(r"(.*?)量能异动日：(.*?)(?:$|!)", line)
            if match:
                prefix = match.group(1)
                dates_str = match.group(2).strip()
                dates = [d.strip() for d in dates_str.split(",")]
                processed_lines.append(f"{prefix}量能异动日：")
                # Group dates into lines of 8
                for i in range(0, len(dates), 8):
                    processed_lines.append(", ".join(dates[i:i+8]))
                continue

        # Fix long source lines - break into multiple lines BEFORE URL processing
        if "- 来源：" in line:
            match = re.search(r"(.*?- 来源：)(.*?)(?:$)", line)
            if match:
                prefix = match.group(1)
                sources_str = match.group(2).strip()

                # Extract URLs from the sources
                url_pattern = r"https?://[^\s,]+"
                urls = re.findall(url_pattern, sources_str)

                if urls:
                    processed_lines.append(prefix)
                    # Create clickable links, 2 per line
                    for i in range(0, len(urls), 2):
                        link_parts = []
                        for url in urls[i:i+2]:
                            parsed = urlparse(url)
                            domain = parsed.netloc or parsed.path[:30]
                            link_parts.append(f"[{domain}]({url})")
                        processed_lines.append("  " + "  ".join(link_parts))
                    continue

        # Fix bare URLs in other lines - make them clickable
        url_pattern = r"(https?://[^\s\]\),]+)"
        def replace_url(match):
            url = match.group(1)
            # Skip if already in markdown link format
            if f"]({url})" in line or f"](<{url}>)" in line:
                return url
            # Extract domain for link text
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path[:30]
            return f"[{domain}]({url})"

        line = re.sub(url_pattern, replace_url, line)

        processed_lines.append(line)

    processed_content = "\n".join(processed_lines)

    # Convert HTML font color tags to LaTeX color commands for PDF
    # Pattern: <font color='red'>text</font> or <font color="red">text</font>
    def convert_font_color(match):
        color_name = match.group(1).strip('"\'')
        text = match.group(2)
        # Map HTML color names to LaTeX colors
        color_map = {
            'red': 'highlightred',
            'green': 'highlightgreen',
            'blue': 'highlightblue',
            'orange': 'highlightorange',
            'purple': 'highlightpurple',
        }
        latex_color = color_map.get(color_name.lower(), color_name)
        return f'\\textcolor{{{latex_color}}}{{{text}}}'
    # Use a more robust regex that handles nested structures
    processed_content = re.sub(
        r"<font\s+color=['\"]([^'\"]+)['\"]>([^<]+)</font>",
        convert_font_color,
        processed_content,
        flags=re.IGNORECASE
    )

    # Add spacing improvements
    processed_content = re.sub(r"\n{3,}", "\n\n", processed_content)  # Fix excessive blank lines

    processed_path.write_text(processed_content, encoding="utf-8")
    return processed_path


def build_pdf(md_path: Path) -> Optional[Path]:
    """
    Build PDF from Markdown using pandoc with improved formatting.

    Args:
        md_path: Path to Markdown file

    Returns:
        Path to generated PDF, or None if generation failed
    """
    # Post-process markdown for better formatting
    processed_md = postprocess_markdown(md_path)

    pdf_path = md_path.with_suffix(".pdf")

    # Create LaTeX header for better styling
    latex_header = r"""\usepackage{geometry}
\usepackage{hyperref}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{graphicx}

% Page geometry
\geometry{
    a4paper,
    left=25mm,
    right=25mm,
    top=25mm,
    bottom=25mm,
}

% Hyperlink setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue,
    pdftitle={A股趋势跟踪研报},
    pdfauthor={Trend Agent},
}

% Table styling
\renewcommand{\arraystretch}{1.3}

% Image styling - center all images
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother

% Center images and scale if needed
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\makeatletter
\g@addto@macro\@floatboxreset\centering
\makeatother

% Define colors for highlights
\definecolor{highlightred}{RGB}{220, 50, 50}
\definecolor{highlightgreen}{RGB}{50, 160, 50}
\definecolor{highlightblue}{RGB}{50, 100, 220}
\definecolor{highlightorange}{RGB}{220, 120, 20}
\definecolor{highlightpurple}{RGB}{140, 50, 180}
"""

    # Write header to temp file
    header_path = md_path.parent / "header.tex"
    header_path.write_text(latex_header, encoding="utf-8")

    build_sh = md_path.parent / "build.sh"
    build_sh.write_text(
        f'#!/usr/bin/env bash\ncd "{md_path.parent}" && pandoc "{processed_md.name}" -o "{pdf_path.name}" --pdf-engine=xelatex -H header.tex -V CJKmainfont="{CHART_FONT}" --toc --number-sections\n',
        encoding="utf-8",
    )
    build_sh.chmod(0o755)

    try:
        # Run pandoc with improved options for better formatting
        subprocess.run(
            [
                "pandoc",
                processed_md.name,
                "-o",
                pdf_path.name,
                "--pdf-engine=xelatex",
                "-H", "header.tex",
                "-V", f"CJKmainfont={CHART_FONT}",
                "--toc",
                "--number-sections",
                "--wrap=none",  # Don't wrap lines automatically
                "--columns=80",  # Set line length
            ],
            cwd=str(md_path.parent),
            check=True,
            capture_output=True,
        )
        logger.info(f"PDF generated: {pdf_path}")
        return pdf_path
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"Pandoc PDF generation failed: {exc}")
        if hasattr(exc, "stderr") and exc.stderr:
            logger.debug(f"Pandoc stderr: {exc.stderr.decode('utf-8', errors='ignore')}")
        return None


def main() -> None:
    """Main entry point for the trend agent pipeline."""
    # Setup logging
    log_level = logging.DEBUG if DEBUG_DEEPSEEK else logging.INFO
    setup_logging(level=log_level, log_file=REPORT_DIR / "trend_agent.log")

    logger.info("=" * 60)
    logger.info("Starting Trend Agent Pipeline")
    logger.info("=" * 60)

    llm = init_llm()

    # Phase 1
    logger.info("Phase 1: Market Intelligence...")
    themes = phase1_market_intel(llm)
    if not themes:
        logger.warning("No themes generated, check search tool or LLM output")

    # Phase 2
    candidates = phase2_quant_filter(themes)
    if candidates.empty:
        logger.error("Phase 2: No candidates after filtering")
        return

    # Phase 3
    logger.info("Phase 3: Deep Research (Audit)...")
    audit_trace = REPORT_DIR / f"audit_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger.debug(f"Audit trace: {audit_trace}")
    audits = phase3_deep_audit(llm, candidates, trace_path=audit_trace)
    candidates, audits = apply_audit_filter(candidates, audits)
    if candidates.empty:
        logger.warning("No candidates passed audit filter")
        return

    # Phase 4
    logger.info("Phase 4: Visualization...")
    chart_notes = phase4_plot_charts(candidates)
    signals = compute_signals(candidates)

    # Phase 5
    logger.info("Phase 5: Report Generation...")
    md_path = phase5_report_with_deepseek(themes, candidates, audits, chart_notes, signals)
    pdf_path = build_pdf(md_path)

    logger.info("=" * 60)
    logger.info(f"Report generated: {md_path}")
    if pdf_path:
        logger.info(f"PDF generated: {pdf_path}")
    else:
        logger.info("PDF not generated (pandoc may not be available)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
