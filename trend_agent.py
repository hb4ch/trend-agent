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
import hashlib
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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deep_researcher import deepseek_chat, deepseek_plan_queries, zhipu_search
from screen_growth_stocks import screen_all_stocks
from utils import (
    DragonTigerList,
    EPSILON,
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
from llm_provider import get_llm_provider, LLMProvider, invoke_llm_messages

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
REGULATORY_MAX_AGE_DAYS = int(os.environ.get("REGULATORY_MAX_AGE_DAYS", "730"))
THEME_CACHE_SCHEMA_VERSION = "3"


@dataclass
class StrategyConfig:
    """Runtime strategy configuration for alpha-sensitive controls."""

    holding_horizon: str = "swing_2_8w"
    toplist_exclusion_mode: str = "penalty"  # penalty|exclude_crowded|none
    toplist_penalty_weight: float = 0.25
    toplist_lookback_days: int = 60
    toplist_crowded_min_hits: int = 4
    hard_fail_require_recency: bool = True
    hard_fail_max_age_days: int = REGULATORY_MAX_AGE_DAYS
    hard_fail_reduce_materiality_threshold: float = 0.03
    theme_cache_version: str = THEME_CACHE_SCHEMA_VERSION
    max_names_per_theme: int = 4
    max_names_per_industry: int = 4

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        """Build configuration from environment variables."""
        return cls(
            holding_horizon=os.environ.get("HOLDING_HORIZON", "swing_2_8w"),
            toplist_exclusion_mode=os.environ.get("TOPLIST_EXCLUSION_MODE", "penalty"),
            toplist_penalty_weight=float(os.environ.get("TOPLIST_PENALTY_WEIGHT", "0.25")),
            toplist_lookback_days=int(os.environ.get("TOPLIST_LOOKBACK_DAYS", "60")),
            toplist_crowded_min_hits=int(os.environ.get("TOPLIST_CROWDED_MIN_HITS", "4")),
            hard_fail_require_recency=os.environ.get("HARD_FAIL_REQUIRE_RECENCY", "1").strip() in {"1", "true", "True", "YES", "yes"},
            hard_fail_max_age_days=int(os.environ.get("HARD_FAIL_MAX_AGE_DAYS", str(REGULATORY_MAX_AGE_DAYS))),
            hard_fail_reduce_materiality_threshold=float(os.environ.get("HARD_FAIL_REDUCE_MATERIALITY_THRESHOLD", "0.03")),
            theme_cache_version=os.environ.get("THEME_CACHE_VERSION", THEME_CACHE_SCHEMA_VERSION),
            max_names_per_theme=int(os.environ.get("MAX_NAMES_PER_THEME", "4")),
            max_names_per_industry=int(os.environ.get("MAX_NAMES_PER_INDUSTRY", "4")),
        )



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
class PositiveFinding:
    """Positive finding from opportunity discovery research."""

    category: str  # "contract", "customer", "policy", "technology", "expansion"
    description: str
    evidence: str
    confidence: float  # 0.0-1.0
    source_url: str
    date: Optional[str] = None


@dataclass
class GrowthCatalyst:
    """Growth catalyst identified during research."""

    catalyst_type: str  # "policy", "tech_breakthrough", "market_expansion", "competitive_moat"
    description: str
    timeframe: str  # "near_term", "medium_term", "long_term"
    confidence: float


@dataclass
class AuditResult:
    """Audit result for a stock-theme pair."""

    ts_code: str
    name: str
    theme: str
    verdict: str
    rationale: str
    sources: List[str]
    # New fields for opportunity discovery
    positive_findings: List[PositiveFinding] = None
    growth_catalysts: List[GrowthCatalyst] = None
    confidence_score: float = 0.5
    research_depth: str = "standard"
    capital_signal_summary: str = ""

    def __post_init__(self):
        if self.positive_findings is None:
            self.positive_findings = []
        if self.growth_catalysts is None:
            self.growth_catalysts = []


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
    themes: List[ThemeItem], candidates: pd.DataFrame, cache_path: Path, cache_version: str = THEME_CACHE_SCHEMA_VERSION
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
    if not themes or candidates.empty:
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
    theme_fingerprint_text = json.dumps(
        sorted(
            [{"name": t["name"], "keywords": sorted(t.get("keywords", [])), "summary": t.get("summary", "")} for t in theme_list],
            key=lambda x: x["name"],
        ),
        ensure_ascii=False,
    )
    theme_set_hash = hashlib.sha256(theme_fingerprint_text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def row_fingerprint(row: pd.Series) -> str:
        """Create fingerprint for caching theme matches."""
        from utils import create_row_fingerprint

        return create_row_fingerprint(
            row.to_dict(),
            ["name", "industry", "main_business", "business_scope", "introduction"],
        )

    def get_text(row: pd.Series) -> str:
        """Get text representation of stock for theme matching."""
        main_business = str(row.get("main_business", "") or "")
        introduction = str(row.get("introduction", "") or "")
        business_scope = str(row.get("business_scope", "") or "")
        # `business_scope` is often registration boilerplate with broad terms.
        # Keep only a short snippet so it cannot dominate classification.
        business_scope = re.sub(r"一般项目[:：]", "", business_scope)[:180]
        return " | ".join(
            [
                f"name={row.get('name','')}",
                f"code={row.get('ts_code','')}",
                f"industry={row.get('industry','')}",
                f"main_business={main_business}",
                f"business_scope={business_scope}",
                f"intro={introduction[:500]}",
            ]
        )

    def _theme_tokens(theme_name: str, keywords: List[str]) -> set[str]:
        toks: set[str] = set()
        for part in [theme_name, *(keywords or [])]:
            cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(part))
            for tok in cleaned.split():
                if len(tok) >= 2 and tok not in {"题材", "板块", "产业", "行业", "应用", "前沿", "高端", "制造"}:
                    toks.add(tok)
        return toks

    theme_token_map: Dict[str, set[str]] = {
        t["name"]: _theme_tokens(t["name"], t.get("keywords", [])) for t in theme_list
    }

    updated = candidates.copy()
    matched_map: Dict[str, List[str]] = {}
    row_by_code: Dict[str, pd.Series] = {str(r["ts_code"]): r for _, r in updated.iterrows()}

    def validate_match(ts_code: str, picked: List[str]) -> List[str]:
        """
        Deterministic sanity guard:
        - require >=1 hit in main_business, OR
        - require >=2 hits in (main_business + introduction)
        This blocks matches that rely only on generic business_scope phrases.
        """
        row = row_by_code.get(str(ts_code))
        if row is None:
            return picked
        main_business = str(row.get("main_business", "") or "")
        combined = f"{main_business} {str(row.get('introduction','') or '')}"
        kept: List[str] = []
        for theme_name in picked:
            toks = theme_token_map.get(theme_name, set())
            if not toks:
                continue
            main_hits = sum(1 for tok in toks if tok in main_business)
            combined_hits = sum(1 for tok in toks if tok in combined)
            if main_hits >= 1 or combined_hits >= 2:
                kept.append(theme_name)
        return kept

    batch = []
    batch_keys = []
    for _, row in updated.iterrows():
        key = f"{cache_version}:{theme_set_hash}:{row_fingerprint(row)}"
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
                    "你是A股题材归属分类器，采用多层次关系分析。\n\n"
                    "**匹配层次（按优先级）：**\n"
                    "1. **直接匹配**：公司业务直接属于题材\n"
                    "2. **间接关联**：\n"
                    "   - 产业链上下游（供应商/客户关系）\n"
                    "   - 技术生态关联（技术兼容、生态位）\n"
                    "   - 市场传导逻辑（需求传导、政策联动）\n"
                    "3. **概念延伸**：符合题材叙事逻辑或市场预期的标的\n\n"
                    "**要求：**\n"
                    "- 从白名单中选择0-2个最相关题材\n"
                    "- 仅当主营业务/产品/客户链条有实质关联时才可匹配\n"
                    "- 不允许仅凭经营范围模板词（如通信设备制造、AI硬件销售等）直接匹配\n"
                    "- 若证据主要来自经营范围而非主营业务，必须返回空数组\n"
                    "- 完全无关才返回空数组\n"
                    "- 输出严格JSON：{\"matches\":{\"ts_code\":[\"theme1\",\"theme2\"]},\"notes\":{\"ts_code\":\"reason\"}}\n"
                    "- notes中说明关联逻辑（直接/间接/概念延伸）"
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

        content = invoke_llm_messages("qwen", messages, temperature=0.1)
        parsed = safe_json_loads(content or "")
        matches = parsed.get("matches", {}) if isinstance(parsed, dict) else {}
        notes = parsed.get("notes", {}) if isinstance(parsed, dict) else {}

        for fp, item in zip(batch_fp, batch_items):
            ts_code = item.get("ts_code")
            picked = matches.get(ts_code, []) if isinstance(matches, dict) else []
            if not isinstance(picked, list):
                picked = []
            picked = [str(x) for x in picked if str(x) in [t["name"] for t in theme_list]]
            picked = validate_match(str(ts_code), picked)
            matched_map[ts_code] = picked
            cache[fp] = {
                "schema_version": cache_version,
                "theme_set_hash": theme_set_hash,
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


def heuristic_match_themes(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    existing_col: str = "matched_themes",
) -> pd.DataFrame:
    """
    Heuristically assign themes based on keyword and theme-name token overlap.

    Used as a fallback when LLM matching returns sparse or zero coverage.
    """
    if candidates.empty or not themes:
        return candidates
    out = candidates.copy()

    def normalized_tokens(text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text or ""))
        return [tok for tok in cleaned.split() if len(tok) >= 2]

    theme_features = []
    for t in themes:
        if not t.name:
            continue
        tokens = set(normalized_tokens(t.name))
        for kw in (t.keywords or []):
            tokens.update(normalized_tokens(kw))
        tokens = {tok for tok in tokens if tok not in {"主题", "板块", "产业", "行业", "概念"}}
        theme_features.append((t.name, tokens))

    if not theme_features:
        return out

    def text_for_row(row: pd.Series) -> str:
        return " ".join(
            [
                str(row.get("name", "")),
                str(row.get("industry", "")),
                str(row.get("main_business", "")),
                str(row.get("business_scope", "")),
                str(row.get("introduction", "")),
            ]
        )

    matched_values = []
    for _, row in out.iterrows():
        base = row.get(existing_col, [])
        base = base if isinstance(base, list) else []
        if base:
            matched_values.append(base)
            continue
        text = text_for_row(row)
        picked = []
        for theme_name, toks in theme_features:
            if not toks:
                continue
            if any(tok and tok in text for tok in toks):
                picked.append(theme_name)
            if len(picked) >= 2:
                break
        matched_values.append(picked)
    out[existing_col] = matched_values
    return out


# Constants for regulatory audit - Expanded source domains for opportunity discovery
# Tier 1: Official Disclosures (highest credibility)
PRIMARY_SOURCE_DOMAINS = (
    "cninfo.com.cn",    # 巨潮资讯网 (official announcements)
    "sse.com.cn",       # 上交所
    "szse.cn",          # 深交所
)

# Tier 2: Financial News & Analysis (high credibility)
SECONDARY_SOURCE_DOMAINS = (
    "eastmoney.com",    # 东方财富
    "10jqka.com.cn",    # 同花顺
    "cls.cn",           # 财联社 (fast news)
    "yicai.com",        # 第一财经
    "caixin.com",       # 财新网
    "wallstreetcn.com", # 华尔街见闻
    "sina.com.cn",      # 新浪财经
    "gelonghui.com",    # 格隆汇
    "xueqiu.com",       # 雪球
)

# Tier 3: Government & Policy (for policy catalysts)
POLICY_SOURCE_DOMAINS = (
    "gov.cn",           # 政府门户
    "ndrc.gov.cn",      # 发改委
    "miit.gov.cn",      # 工信部
    "most.gov.cn",      # 科技部
)

# Tier 4: Company Background (for due diligence)
BACKGROUND_SOURCE_DOMAINS = (
    "tianyancha.com",   # 天眼查
    "qichacha.com",     # 企查查
)

# Combined for broader opportunity searches
ALL_SOURCE_DOMAINS = PRIMARY_SOURCE_DOMAINS + SECONDARY_SOURCE_DOMAINS + POLICY_SOURCE_DOMAINS + BACKGROUND_SOURCE_DOMAINS


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


def parse_reduce_ratio(text: str) -> Optional[float]:
    """Parse reduce percentage from text (e.g. 3.5%)."""
    if not text:
        return None
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        try:
            return float(m) / 100.0
        except ValueError:
            continue
    return None


def is_material_reduce_event(text: str, threshold: float) -> bool:
    """
    Determine if a减持 event is materially large.

    Treat clear liquidation language as material even without explicit ratio.
    """
    if not text:
        return False
    ratio = parse_reduce_ratio(text)
    if ratio is not None and ratio >= threshold:
        return True
    liquidation_terms = ("清仓", "全部减持", "大比例减持", "集中竞价减持")
    return any(term in text for term in liquidation_terms)


def detect_hard_fail_reason(
    hit: dict,
    name: str,
    symbol: str,
    require_recency: bool,
    max_age_days: int,
    reduce_threshold: float,
) -> Optional[str]:
    """Classify a search hit into a hard-fail reason, if any."""
    if not isinstance(hit, dict):
        return None
    if not is_relevant_search_hit(name, symbol, hit):
        return None

    if require_recency:
        dt = parse_result_date(hit.get("date"))
        if not is_recent(dt, max_age_days):
            return None

    title = str(hit.get("title", ""))
    snippet = str(hit.get("snippet", ""))
    text = f"{title} {snippet}"

    if re.search(r"(被|遭|因|涉嫌).{0,12}(立案|立案调查)", text):
        return "recent_investigation"
    if re.search(r"(重大诉讼|未决诉讼|仲裁|诉讼事项)", text):
        return "major_litigation"
    if re.search(r"(终止上市|退市风险警示|暂停上市|强制退市)", text):
        return "delisting_risk"
    if re.search(r"(拟|计划).{0,12}减持|减持计划", text) and is_material_reduce_event(text, reduce_threshold):
        return "material_reduction"
    return None


def extract_toplist_hit_counts(dtl: DragonTigerList, days: int) -> Dict[str, int]:
    """Build per-stock toplist hit counts in a lookback window."""
    df = dtl.load_recent_toplist(days=days)
    if df is None or df.empty or "ts_code" not in df.columns:
        return {}
    counts = df["ts_code"].value_counts()
    return {str(code): int(cnt) for code, cnt in counts.items()}


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
        logger.info(f"Python execution result: {result}")
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


def init_llm():
    """
    Initialize and return the DeepSeek LLM for audit tasks.

    Returns:
        LangChain BaseChatModel configured for DeepSeek
    """
    return get_llm_provider().get_llm("deepseek", temperature=0.2)


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


def phase1_market_intel(llm: BaseChatModel) -> List[ThemeItem]:
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


def phase2_quant_filter(themes: List[ThemeItem], config: Optional[StrategyConfig] = None) -> pd.DataFrame:
    """
    Phase 2: Screen stocks using quantitative technical criteria with progressive relaxation.

    Implements progressive relaxation when no theme matches are found:
    1. Strict tier - highest quality criteria
    2. Relaxed tier - moderate criteria
    3. Loose tier - most lenient criteria
    4. OFF_THEME_FALLBACK - best technical stocks (clearly marked)

    Args:
        themes: List of market themes for matching (should have validation_status)

    Returns:
        DataFrame of filtered candidate stocks
    """
    logger.info("Starting Phase 2: Quantitative Filtering with Progressive Relaxation")
    config = config or StrategyConfig.from_env()

    # Filter to only confirmed themes (web + capital both confirmed)
    confirmed_themes = [t for t in themes if t.validation_status == "confirmed"]
    web_only_themes = [t for t in themes if t.validation_status == "web_only"]
    capital_only_themes = [t for t in themes if t.validation_status == "capital_only"]
    weak_themes = [t for t in themes if t.validation_status == "weak"]

    # If no confirmed themes, fall back to web_only
    if not confirmed_themes and web_only_themes:
        logger.warning("No confirmed themes, falling back to web_only themes")
        confirmed_themes = web_only_themes

    if not confirmed_themes:
        logger.warning("No confirmed or web_only themes, using all themes")
        confirmed_themes = themes

    logger.info(f"Phase 2 filtering with {len(confirmed_themes)} primary themes: {[t.name for t in confirmed_themes]}")
    expanded_themes = confirmed_themes + [t for t in (web_only_themes + capital_only_themes) if t.name not in {x.name for x in confirmed_themes}]
    broad_themes = expanded_themes + [t for t in weak_themes if t.name not in {x.name for x in expanded_themes}]

    # Build toplist recency features for overcrowding control
    dtl = DragonTigerList()
    toplist_hit_counts = extract_toplist_hit_counts(dtl, days=config.toplist_lookback_days)
    logger.info(f"Loaded toplist hit counts for {len(toplist_hit_counts)} stocks")

    screen_df = screen_all_stocks()
    if screen_df is None or screen_df.empty:
        logger.warning("No screening results from screen_all_stocks()")
        return pd.DataFrame()

    if "ts_code" in screen_df.columns:
        screen_df = screen_df.copy()
        screen_df["toplist_hit_count"] = screen_df["ts_code"].map(lambda x: toplist_hit_counts.get(str(x), 0))
        screen_df["toplist_recency_score"] = screen_df["toplist_hit_count"].map(lambda x: min(1.0, float(x) / 5.0))

        if config.toplist_exclusion_mode == "exclude_crowded":
            before_count = len(screen_df)
            screen_df = screen_df[screen_df["toplist_hit_count"] < config.toplist_crowded_min_hits]
            logger.info(f"Excluded {before_count - len(screen_df)} ultra-crowded toplist stocks")
        elif config.toplist_exclusion_mode == "none":
            screen_df["toplist_recency_score"] = 0.0

    con = duckdb.connect()
    con.register("screen", screen_df)
    theme_strength_map = {}
    for theme in themes:
        base = 0.35
        if theme.validation_status == "confirmed":
            base = 1.0
        elif theme.validation_status == "web_only":
            base = 0.75
        elif theme.validation_status == "capital_only":
            base = 0.7
        elif theme.validation_status == "weak":
            base = 0.4
        if theme.capital_signal:
            base = min(1.1, base + 0.05)
        theme_strength_map[theme.name] = base

    # Define progressive relaxation tiers
    filter_tiers = [
        {
            "name": "Strict",
            "consolidation": 70,
            "ma_spread": 0.15,
            "ma_spread_std": 0.03,
            "volume_min": 1.2,
            "volume_max": 3.0,
        },
        {
            "name": "Relaxed",
            "consolidation": 60,
            "ma_spread": 0.20,
            "ma_spread_std": 0.05,
            "volume_min": 1.0,
            "volume_max": 4.0,
        },
        {
            "name": "Loose",
            "consolidation": 50,
            "ma_spread": 0.30,
            "ma_spread_std": 0.08,
            "volume_min": 0.8,
            "volume_max": 5.0,
        },
    ]

    # Try each tier until we find theme matches
    for tier in filter_tiers:
        filtered = con.execute(f"""
            SELECT *
            FROM screen
            WHERE consolidation_score >= {tier['consolidation']}
              AND ma_spread <= {tier['ma_spread']}
              AND ma_spread_std <= {tier['ma_spread_std']}
              AND volume_boost >= {tier['volume_min']}
              AND volume_boost <= {tier['volume_max']}
            ORDER BY composite_score DESC
        """).df()

        logger.info(f"Tier '{tier['name']}': {len(filtered)} candidates passed technical filter")

        # Use staged theme matching: primary themes first, then broader pool if needed.
        filtered = qwen_match_themes(
            confirmed_themes,
            filtered,
            cache_path=Path(".cache/qwen_theme_match.json"),
            cache_version=config.theme_cache_version,
        )
        has_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
        if has_match == 0 and expanded_themes:
            filtered = qwen_match_themes(
                expanded_themes,
                filtered,
                cache_path=Path(".cache/qwen_theme_match.json"),
                cache_version=f"{config.theme_cache_version}_expanded",
            )
            has_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
        if has_match == 0 and broad_themes:
            filtered = heuristic_match_themes(broad_themes, filtered, existing_col="matched_themes")

        has_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
        logger.info(f"Tier '{tier['name']}': {has_match} candidates matched themes")

        if has_match > 0:
            # Found theme matches at this tier
            filtered = filtered[filtered["matched_themes"].apply(bool)]
            filtered["theme_strength_score"] = filtered["matched_themes"].map(
                lambda arr: max([theme_strength_map.get(str(t), 0.5) for t in arr], default=0.0)
            )
            filtered["alpha_rank_score"] = (
                filtered["composite_score"]
                + filtered["theme_strength_score"] * 15.0
                - filtered["toplist_recency_score"] * (config.toplist_penalty_weight * 10.0)
            )
            filtered["off_theme"] = False
            filtered["filter_tier"] = tier['name']
            filtered = filtered.sort_values("alpha_rank_score", ascending=False)
            logger.info(f"Phase 2 complete: Using tier '{tier['name']}', {len(filtered.head(15))} candidates with theme matches")
            return filtered.head(15)

    # Final fallback - clearly mark as off-theme
    logger.warning("No theme matches after all tiers, returning best technical stocks (OFF-THEME FALLBACK)")
    # Get the best candidates from the loosest tier
    filtered = con.execute(f"""
        SELECT *
        FROM screen
        WHERE consolidation_score >= 50
        ORDER BY composite_score DESC
    """).df()
    filtered = qwen_match_themes(
        confirmed_themes,
        filtered,
        cache_path=Path(".cache/qwen_theme_match.json"),
        cache_version=config.theme_cache_version,
    )
    has_fallback_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
    if has_fallback_match == 0 and expanded_themes:
        filtered = qwen_match_themes(
            expanded_themes,
            filtered,
            cache_path=Path(".cache/qwen_theme_match.json"),
            cache_version=f"{config.theme_cache_version}_expanded",
        )
        has_fallback_match = filtered["matched_themes"].apply(bool).sum() if "matched_themes" in filtered.columns else 0
    if has_fallback_match == 0 and broad_themes:
        filtered = heuristic_match_themes(broad_themes, filtered, existing_col="matched_themes")
    filtered["theme_strength_score"] = filtered["matched_themes"].map(
        lambda arr: max([theme_strength_map.get(str(t), 0.5) for t in arr], default=0.0)
    )
    filtered["alpha_rank_score"] = (
        filtered["composite_score"]
        + filtered["theme_strength_score"] * 15.0
        - filtered["toplist_recency_score"] * (config.toplist_penalty_weight * 10.0)
    )
    filtered["off_theme"] = filtered["matched_themes"].map(lambda arr: len(arr) == 0)
    filtered["filter_tier"] = "OFF_THEME_FALLBACK"
    filtered = filtered.sort_values("alpha_rank_score", ascending=False)
    logger.info(f"Phase 2 complete: OFF_THEME_FALLBACK mode, returning top 15 technical candidates")
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


def clamp01(value: float) -> float:
    """Clamp numeric value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def apply_diversification_constraints(
    ranked: pd.DataFrame,
    max_per_theme: int,
    max_per_industry: int,
    target_n: int = 15,
) -> pd.DataFrame:
    """Select top candidates while limiting concentration by theme/industry."""
    selected_rows = []
    theme_count: Dict[str, int] = {}
    industry_count: Dict[str, int] = {}

    for _, row in ranked.iterrows():
        industry = str(row.get("industry", "") or "Unknown")
        matched = row.get("matched_themes", [])
        themes = matched if isinstance(matched, list) and matched else ["OFF_THEME"]
        primary_theme = str(themes[0])
        if theme_count.get(primary_theme, 0) >= max_per_theme:
            continue
        if industry_count.get(industry, 0) >= max_per_industry:
            continue
        selected_rows.append(row)
        theme_count[primary_theme] = theme_count.get(primary_theme, 0) + 1
        industry_count[industry] = industry_count.get(industry, 0) + 1
        if len(selected_rows) >= target_n:
            break

    if len(selected_rows) < target_n:
        selected_codes = {str(r["ts_code"]) for r in selected_rows}
        for _, row in ranked.iterrows():
            if str(row.get("ts_code", "")) in selected_codes:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= target_n:
                break

    return pd.DataFrame(selected_rows) if selected_rows else ranked.head(target_n)


def rank_candidates_for_alpha(
    candidates: pd.DataFrame,
    audits: List[AuditResult],
    signals: Dict[str, Dict[str, object]],
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """
    Build alpha-oriented ranking fields and return diversified top candidates.
    """
    if candidates is None or candidates.empty:
        return candidates
    config = config or StrategyConfig.from_env()
    ranked = candidates.copy()

    from deep_researcher import get_source_tier_weight

    audit_by_code: Dict[str, List[AuditResult]] = {}
    for audit in audits:
        audit_by_code.setdefault(audit.ts_code, []).append(audit)

    verdict_risk = {"pass": 0.1, "warn": 0.45, "fail": 0.95}

    def audit_features(ts_code: str) -> Dict[str, float]:
        items = audit_by_code.get(ts_code, [])
        if not items:
            return {
                "audit_risk_score": 0.5,
                "positive_finding_count": 0.0,
                "source_quality_score": 0.0,
                "catalyst_diversity": 0.0,
            }
        worst = max(verdict_risk.get(item.verdict, 0.45) for item in items)
        findings = []
        catalyst_types = set()
        source_scores = []
        for item in items:
            findings.extend(item.positive_findings or [])
            for c in (item.growth_catalysts or []):
                catalyst_types.add(c.catalyst_type)
            for src in (item.sources or []):
                source_scores.append(get_source_tier_weight(src))
        source_quality = float(np.mean(source_scores)) if source_scores else 0.0
        return {
            "audit_risk_score": worst,
            "positive_finding_count": float(len(findings)),
            "source_quality_score": source_quality,
            "catalyst_diversity": float(len(catalyst_types)),
        }

    alpha_rows = []
    for _, row in ranked.iterrows():
        ts_code = str(row["ts_code"])
        sig = signals.get(ts_code, {})
        af = audit_features(ts_code)
        breakout_window_ok = 1.0 if sig.get("breakout_window_ok") else 0.0
        already_breakout = 1.0 if sig.get("already_breakout") else 0.0
        extended_breakout = 1.0 if sig.get("extended_breakout") else 0.0
        turnover_mult = float(sig.get("turnover_mult", row.get("volume_boost", 1.0)) or 1.0)
        volume_quality = clamp01(1.0 - abs(turnover_mult - 1.8) / 1.8)
        ma_spread = float(row.get("ma_spread", 0.3) or 0.3)
        ma_comp = clamp01(1.0 - ma_spread / 0.30)
        theme_strength = clamp01(float(row.get("theme_strength_score", 0.0)))
        source_quality = clamp01(af["source_quality_score"])
        finding_score = clamp01(af["positive_finding_count"] / 4.0)
        catalyst_score = clamp01(af["catalyst_diversity"] / 3.0)
        audit_safe = 1.0 - clamp01(af["audit_risk_score"])
        toplist_recency = clamp01(float(row.get("toplist_recency_score", 0.0) or 0.0))
        blowoff_penalty = clamp01(max(0.0, turnover_mult - 3.0) / 2.0)
        breakout_penalty = 0.20 if extended_breakout > 0 else (0.08 if already_breakout > 0 else 0.0)
        overcrowding_penalty = clamp01(0.65 * toplist_recency + 0.35 * blowoff_penalty)

        score_01 = (
            0.22 * breakout_window_ok +
            0.17 * volume_quality +
            0.16 * ma_comp +
            0.15 * theme_strength +
            0.10 * finding_score +
            0.07 * catalyst_score +
            0.06 * source_quality +
            0.07 * audit_safe
            - 0.12 * overcrowding_penalty
            - breakout_penalty
        )
        score_01 = clamp01(score_01)
        alpha_rows.append(
            {
                "ts_code": ts_code,
                "audit_risk_score": af["audit_risk_score"],
                "positive_finding_count": af["positive_finding_count"],
                "source_quality_score": source_quality,
                "catalyst_diversity": af["catalyst_diversity"],
                "alpha_rank_score": score_01 * 100.0,
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    ranked = ranked.merge(alpha_df, on="ts_code", how="left", suffixes=("", "_new"))
    if "alpha_rank_score_new" in ranked.columns:
        ranked["alpha_rank_score"] = ranked["alpha_rank_score_new"]
        ranked = ranked.drop(columns=["alpha_rank_score_new"])

    ranked = ranked.sort_values("alpha_rank_score", ascending=False)
    ranked = apply_diversification_constraints(
        ranked,
        max_per_theme=config.max_names_per_theme,
        max_per_industry=config.max_names_per_industry,
        target_n=15,
    )
    return ranked.reset_index(drop=True)


def phase3_deep_audit(
    llm: BaseChatModel,
    candidates: pd.DataFrame,
    trace_path: Optional[Path] = None,
    themes: Optional[List[ThemeItem]] = None,
    config: Optional[StrategyConfig] = None,
) -> List[AuditResult]:
    """
    Phase 3: Deep Research with Opportunity-First, Then Adversarial Audit.

    Two-pass research strategy:
    1. **Opportunity Discovery Pass**: Find positive catalysts (contracts, tech, policy, expansion)
    2. **Adversarial Veto Pass**: Due diligence to check for hard fails

    Args:
        llm: Language model for processing
        candidates: DataFrame of candidate stocks
        trace_path: Optional path for audit trace
        themes: Optional list of themes with capital signals

    Returns:
        List of AuditResult with both positive findings and veto status
    """
    from deep_researcher import (
        generate_opportunity_queries,
        extract_positive_findings,
        get_source_tier_weight,
    )

    config = config or StrategyConfig.from_env()
    top = candidates.head(15)
    audit_results: List[AuditResult] = []

    # Build theme capital signal map
    theme_capital_map = {}
    if themes:
        for t in themes:
            if t.name and t.capital_signal:
                theme_capital_map[t.name] = t.capital_signal

    audit_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是A股尽调员，评估股票风险。\n"
                "**一票否决（fail）条件（必须有明确证据）：**\n"
                "- 正在被立案调查（不是历史已结案）\n"
                "- 正在进行的重大诉讼（不是历史诉讼）\n"
                "- 近期公告的大规模减持计划\n"
                "- 退市风险警示（*ST）\n\n"
                "**存疑（warn）条件：**\n"
                "- 监管函、问询函（但已回复）\n"
                "- 业绩不确定性\n"
                "- 证据不足无法验证\n\n"
                "**通过（pass）条件：**\n"
                "- 无明显风险，有正面业务证据\n\n"
                "注意：缺少证据≠fail，应判warn。只有明确的重大风险才判fail。",
            ),
            (
                "user",
                "给定搜索摘要，输出JSON："
                '{{"verdict":"pass|warn|fail","rationale":"","sources":["url1"]}}\n\n'
                "股票:{name} 题材:{theme}\n搜索结果:\n{results}",
            ),
        ]
    )
    audit_chain = audit_prompt | llm | StrOutputParser()

    for _, row in top.iterrows():
        name = row["name"]
        symbol = stock_symbol(row["ts_code"])
        matched_themes = row.get("matched_themes", []) or ["未明确题材"]

        for theme in matched_themes:
            logger.debug(f"Research start: stock={name} theme={theme}")
            trace_append(trace_path, "research_start", {"ts_code": row["ts_code"], "name": name, "theme": theme})

            # Get capital signal for this theme
            capital_signal = theme_capital_map.get(theme, "")

            # ============ PASS 1: Opportunity Discovery ============
            logger.debug(f"Opportunity discovery pass: stock={name}")
            trace_append(trace_path, "opportunity_pass_start", {"ts_code": row["ts_code"], "name": name, "theme": theme})

            positive_findings: List[PositiveFinding] = []
            growth_catalysts: List[GrowthCatalyst] = []
            opportunity_urls = []
            opportunity_evidence = []

            # Generate opportunity queries (without site: restrictions for broader coverage)
            opportunity_queries = generate_opportunity_queries(name, theme)

            for q_info in opportunity_queries:
                query = q_info["query"]
                category = q_info["category"]
                logger.debug(f"Opportunity search: category={category} q={query}")

                raw = run_search(query)
                parsed = parse_search_payload(raw)
                urls = parsed.get("urls", [])
                search_results = parsed.get("results", [])

                trace_append(trace_path, "opportunity_search", {
                    "ts_code": row["ts_code"],
                    "name": name,
                    "theme": theme,
                    "category": category,
                    "query": query,
                    "urls": urls,
                })

                # Extract positive findings
                findings = extract_positive_findings(search_results, name, category)
                for f in findings:
                    positive_findings.append(PositiveFinding(
                        category=f["category"],
                        description=f["description"],
                        evidence=f["evidence"],
                        confidence=f["confidence"],
                        source_url=f["source_url"],
                        date=f.get("date"),
                    ))
                    if f["source_url"] not in opportunity_urls:
                        opportunity_urls.append(f["source_url"])

                # Collect evidence snippets for catalyst synthesis
                for result in search_results[:3]:
                    if isinstance(result, dict):
                        snippet = result.get("snippet", "")
                        if snippet and name in snippet:
                            opportunity_evidence.append(f"[{category}] {snippet[:200]}")

            # Synthesize growth catalysts from findings
            if positive_findings:
                catalyst_categories = {}
                for f in positive_findings:
                    if f.category not in catalyst_categories:
                        catalyst_categories[f.category] = []
                    catalyst_categories[f.category].append(f)

                for cat, findings_list in catalyst_categories.items():
                    if len(findings_list) >= 1:
                        avg_confidence = sum(f.confidence for f in findings_list) / len(findings_list)
                        # Map category to catalyst type
                        catalyst_type_map = {
                            "policy_driver": "policy",
                            "tech_breakthrough": "tech_breakthrough",
                            "market_expansion": "market_expansion",
                            "competitive_moat": "competitive_moat",
                            "contract_evidence": "contract_evidence",
                        }
                        catalyst_type = catalyst_type_map.get(cat, cat)

                        # Determine timeframe based on evidence
                        timeframe = "medium_term"  # Default
                        if any("近期" in f.description or "已" in f.description for f in findings_list):
                            timeframe = "near_term"
                        elif any("规划" in f.description or "计划" in f.description for f in findings_list):
                            timeframe = "long_term"

                        growth_catalysts.append(GrowthCatalyst(
                            catalyst_type=catalyst_type,
                            description=findings_list[0].description,
                            timeframe=timeframe,
                            confidence=avg_confidence,
                        ))

            logger.info(f"Opportunity discovery: {len(positive_findings)} findings, {len(growth_catalysts)} catalysts for {name}")
            trace_append(trace_path, "opportunity_pass_done", {
                "ts_code": row["ts_code"],
                "name": name,
                "theme": theme,
                "finding_count": len(positive_findings),
                "catalyst_count": len(growth_catalysts),
            })

            # ============ PASS 2: Adversarial Veto Audit ============
            logger.debug(f"Adversarial veto pass: stock={name}")
            trace_append(trace_path, "veto_pass_start", {"ts_code": row["ts_code"], "name": name, "theme": theme})

            merged = {}
            evidence_snippets = [f"[local]\n{local_brief_for_audit(row)}"]
            used_queries = set()
            verdict = "warn"
            rationale = ""
            sources = []

            severe_regulatory_patterns = [
                re.compile(r"(行政处罚|处罚决定书|纪律处分|公开谴责|市场禁入)"),
            ]
            minor_regulatory_patterns = [
                re.compile(r"(监管函|问询函|关注函|责令改正|监管措施决定书)"),
            ]
            positive_terms = [
                "订单", "中标", "客户", "签约", "签署", "签订",
                "合同", "协议", "合作", "供货", "落地", "框架协议",
            ]

            executed_passes = 0
            for pass_id in range(1, 4):
                if pass_id == 1:
                    # First pass: balanced check - positive verification + focused risk check
                    queries = [
                        # Positive verification (2 queries)
                        f"site:cninfo.com.cn {symbol} {name} 重大合同 中标 签约",
                        f"site:cninfo.com.cn {symbol} {name} 投资者关系",
                        # Focused risk check (2 queries - only critical risks)
                        f"site:cninfo.com.cn {symbol} {name} 立案调查",
                        f"site:cninfo.com.cn {symbol} {name} 退市 风险警示",
                    ]
                else:
                    plan = deepseek_plan_queries(
                        name=name,
                        theme=theme,
                        evidence="\n".join(evidence_snippets[-4:])[-2000:],
                        pass_id=pass_id,
                    )
                    if plan is not None:
                        logger.debug(f"Veto plan pass={pass_id}: {truncate(pretty_print(plan), 1200)}")
                        trace_append(trace_path, "veto_plan", {"ts_code": row["ts_code"], "name": name, "theme": theme, "pass_id": pass_id, "plan": plan})
                    if plan and plan.get("stop"):
                        logger.debug(f"Veto plan stop pass={pass_id} reason={plan.get('reason','')}")
                        executed_passes = pass_id
                        break
                    queries = plan.get("queries") if plan else None
                    if not queries:
                        queries = [
                            f"site:cninfo.com.cn {symbol} {name} 投资者关系 活动记录表",
                            f"site:cninfo.com.cn {symbol} {name} 中标 公告",
                        ]

                for query in queries:
                    if query in used_queries:
                        continue
                    used_queries.add(query)
                    logger.debug(f"Veto search pass={pass_id} q={query}")
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
                        raw_clean = f"未找到可用摘要/结果。query={query} urls={url_preview}"
                    merged[f"pass{pass_id}_{len(merged)+1}"] = {
                        "query": query,
                        "raw": raw_clean,
                        "urls": urls,
                        "results": search_results if isinstance(search_results, list) else [],
                    }
                    trace_append(trace_path, "veto_search", {"ts_code": row["ts_code"], "name": name, "theme": theme, "pass_id": pass_id, "query": query, "urls": urls})
                    evidence_snippets.append(raw_clean[:800])

                merged_text = "\n".join([str(item.get("raw", "")) for item in merged.values()])
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
                        if not is_recent(dt, config.hard_fail_max_age_days):
                            continue
                        hay = " ".join([str(hit.get("title", "")), str(hit.get("snippet", ""))])
                        if any(p.search(hay) for p in severe_regulatory_patterns):
                            recent_severe_reg = True
                        if any(p.search(hay) for p in minor_regulatory_patterns):
                            recent_minor_reg = True

                # Hard veto checks - only with relevant and (optionally) recent evidence
                hard_fail_reason = None
                hard_fail_sources: List[str] = []
                for item in merged.values():
                    hits = item.get("results", [])
                    if not isinstance(hits, list):
                        continue
                    for hit in hits:
                        reason = detect_hard_fail_reason(
                            hit=hit,
                            name=name,
                            symbol=symbol,
                            require_recency=config.hard_fail_require_recency,
                            max_age_days=config.hard_fail_max_age_days,
                            reduce_threshold=config.hard_fail_reduce_materiality_threshold,
                        )
                        if reason:
                            hard_fail_reason = reason
                            url = str(hit.get("url", "")).strip()
                            if url:
                                hard_fail_sources.append(url)
                    if hard_fail_reason:
                        break

                if hard_fail_reason:
                    verdict = "fail"
                    reason_map = {
                        "recent_investigation": "近期开启立案调查，按审计口径直接剔除。",
                        "major_litigation": "近期重大诉讼/仲裁风险明确，按审计口径直接剔除。",
                        "delisting_risk": "近期出现退市风险信号，按审计口径直接剔除。",
                        "material_reduction": "近期大比例减持计划明确，按审计口径直接剔除。",
                    }
                    rationale = reason_map.get(hard_fail_reason, "触发一票否决条件，直接剔除。")
                    dedup_urls = []
                    for u in hard_fail_sources + flat_urls:
                        if u and u not in dedup_urls:
                            dedup_urls.append(u)
                    sources = dedup_urls[:5]
                    logger.debug(f"Veto hard fail: reason={hard_fail_reason}")
                    trace_append(trace_path, "veto_hard_fail", {"ts_code": row["ts_code"], "name": name, "theme": theme, "reason": hard_fail_reason, "sources": sources})
                    break

                if recent_severe_reg:
                    verdict = "fail"
                    rationale = f"检索到近{config.hard_fail_max_age_days}天内的行政处罚/纪律处分等严重监管事件，按审计口径剔除。"
                    sources = flat_urls[:5]
                    trace_append(trace_path, "veto_hard_fail", {"ts_code": row["ts_code"], "name": name, "theme": theme, "sources": sources})
                    break

                output = audit_chain.invoke({
                    "name": name,
                    "theme": theme,
                    "results": json.dumps(merged, ensure_ascii=False),
                })
                data = safe_json_loads(output)
                verdict = normalize_verdict(data.get("verdict", verdict))
                rationale = str(data.get("rationale", rationale) or "").strip()
                sources = data.get("sources", sources)
                logger.debug(f"Veto LLM verdict={verdict} rationale={truncate(rationale, 400)}")
                trace_append(trace_path, "veto_llm", {"ts_code": row["ts_code"], "name": name, "theme": theme, "verdict": verdict, "rationale": rationale, "sources": sources})

                if not sources or any(src.strip().lower() == "url1" for src in sources):
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

            # Check if we found positive evidence from opportunity pass
            # Lower threshold: any finding or evidence snippet counts
            has_positive_from_opportunity = len(positive_findings) >= 1 or len(opportunity_evidence) >= 2
            has_positive = any(term in final_text for term in positive_terms) or has_positive_from_opportunity

            # Override LLM verdict if we have opportunity findings (LLM may be too harsh)
            if verdict == "fail" and has_positive_from_opportunity:
                verdict = "warn"
                rationale = (rationale + "；LLM判定失败但发现正面催化信息，降级为存疑。").strip("；")

            if verdict != "fail" and not has_positive:
                if not flat_urls and not opportunity_urls:
                    verdict = "warn"
                    rationale = (rationale + "；检索未返回可核验URL，无法完成审计式验真，暂按存疑处理。").strip("；")
                elif executed_passes >= 2 or len(used_queries) >= 4:
                    # More lenient: only fail if we have zero opportunity evidence
                    if len(positive_findings) >= 1 or len(opportunity_evidence) >= 1:
                        verdict = "warn"
                        rationale = (rationale + "；官方源证据不足，但发现部分正面信息，暂按存疑处理。").strip("；")
                    else:
                        verdict = "warn"  # Changed from fail - let research phase be more lenient
                        rationale = (rationale + "；未找到明确硬证据，暂按存疑处理（需进一步验证）。").strip("；")
                else:
                    verdict = "warn"
                    rationale = (rationale + "；当前检索未找到明确订单/客户/中标等硬证据，暂按存疑处理。").strip("；")

            if verdict != "fail" and recent_minor_reg:
                rationale = (rationale + f"；检索到近{config.hard_fail_max_age_days}天内监管函/问询函等事项，需额外关注。").strip("；")
            if verdict == "pass" and not primary_urls:
                verdict = "warn"
                rationale = (rationale + "；缺少交易所/巨潮等一手来源链接，按审计口径降级。").strip("；")

            # Combine sources from both passes
            all_sources = list(set((primary_urls or flat_urls)[:3] + opportunity_urls[:2]))
            if not all_sources or any(str(src).strip().lower() == "url1" for src in all_sources):
                all_sources = (primary_urls or flat_urls or opportunity_urls)[:5]

            # Calculate confidence score based on findings
            confidence = 0.5  # Base
            if positive_findings:
                avg_finding_confidence = sum(f.confidence for f in positive_findings) / len(positive_findings)
                confidence = 0.3 + avg_finding_confidence * 0.5
            if verdict == "pass":
                confidence = min(1.0, confidence + 0.2)
            elif verdict == "fail":
                confidence = max(0.0, confidence - 0.3)

            # Add opportunity findings to rationale if verdict is pass/warn
            if verdict != "fail" and positive_findings:
                finding_summary = "；".join([f"{f.category}:{f.description[:30]}" for f in positive_findings[:3]])
                rationale = f"发现正面催化：{finding_summary}。{rationale}"

            audit_results.append(
                AuditResult(
                    ts_code=row["ts_code"],
                    name=name,
                    theme=theme,
                    verdict=verdict,
                    rationale=rationale,
                    sources=all_sources,
                    positive_findings=positive_findings,
                    growth_catalysts=growth_catalysts,
                    confidence_score=round(confidence, 2),
                    research_depth="deep" if len(positive_findings) >= 3 else "standard",
                    capital_signal_summary=capital_signal,
                )
            )
            logger.debug(f"Research done: stock={name} theme={theme} verdict={verdict} findings={len(positive_findings)}")
            trace_append(trace_path, "research_done", {
                "ts_code": row["ts_code"],
                "name": name,
                "theme": theme,
                "verdict": verdict,
                "finding_count": len(positive_findings),
                "catalyst_count": len(growth_catalysts),
                "confidence": confidence,
            })

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
    Generate K-line charts with technical indicators using Plotly.

    Args:
        candidates: DataFrame of candidate stocks

    Returns:
        Dictionary mapping stock codes to spike dates
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_notes = {}

    for _, row in candidates.head(8).iterrows():
        ts_code = row["ts_code"]
        df = load_price_data(ts_code)
        if df is None or df.empty:
            logger.warning(f"No price data available for {ts_code}, skipping chart")
            continue
        if len(df) > CHART_DAYS:
            df = df.tail(CHART_DAYS)
        # Skip if not enough data for meaningful chart
        if len(df) < 60:
            logger.warning(f"Insufficient data for {ts_code}: {len(df)} rows, need at least 60")
            continue

        # Create subplot layout: candlestick on top (70%), volume on bottom (30%)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{row['name']} ({ts_code})", "成交量"),
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="K线",
                increasing_line_color="#ef5350",  # Red for up (Chinese convention)
                decreasing_line_color="#26a69a",  # Green for down
                increasing_fillcolor="#ef5350",
                decreasing_fillcolor="#26a69a",
            ),
            row=1, col=1,
        )

        # Moving averages
        if len(df) >= 60:
            df["ma60"] = df["close"].rolling(60).mean()
            if df["ma60"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["ma60"],
                        mode="lines", name="MA60",
                        line=dict(color="orange", width=1.5),
                    ),
                    row=1, col=1,
                )
        if len(df) >= 120:
            df["ma120"] = df["close"].rolling(120).mean()
            if df["ma120"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["ma120"],
                        mode="lines", name="MA120",
                        line=dict(color="dodgerblue", width=1.5),
                    ),
                    row=1, col=1,
                )
        if len(df) >= 250:
            df["ma250"] = df["close"].rolling(250).mean()
            if df["ma250"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["ma250"],
                        mode="lines", name="MA250",
                        line=dict(color="purple", width=1.5),
                    ),
                    row=1, col=1,
                )

        # Turnover spike markers
        spikes = detect_turnover_spikes(df)
        spike_dates = df.index[spikes].strftime("%Y-%m-%d").tolist()
        if spikes.any():
            spike_df = df[spikes]
            fig.add_trace(
                go.Scatter(
                    x=spike_df.index,
                    y=spike_df["high"] * 1.02,  # Slightly above high
                    mode="markers",
                    name="放量信号",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                    ),
                    hovertemplate="%{x}<br>放量异动<extra></extra>",
                ),
                row=1, col=1,
            )

        # Volume bars with color based on price direction
        vol_col = "volume" if "volume" in df.columns else "vol"
        colors = ["#ef5350" if c >= o else "#26a69a"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[vol_col],
                name="成交量",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )

        # Layout configuration
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis_rangeslider_visible=False,
            margin=dict(l=60, r=40, t=80, b=40),
            height=800,
            width=1600,
            font=dict(family="Noto Sans CJK SC, Source Han Sans CN, SimHei, sans-serif"),
        )

        # Hide weekend gaps
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ],
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
        )

        # Save chart
        chart_path = CHART_DIR / f"{ts_code}.png"
        fig.write_image(str(chart_path), scale=2)
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
        box_top_prev = float(df_120["high"].iloc[:-1].max()) if len(df_120) > 1 else box_top
        amplitude = (box_top - box_bottom) / (box_bottom + EPSILON)
        close = float(df_120["close"].iloc[-1])
        dist_to_top = (box_top_prev - close) / (box_top_prev + EPSILON)
        pos = (close - box_bottom) / ((box_top - box_bottom) + EPSILON)

        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        ma120 = df["close"].rolling(120).mean()
        ma_recent = pd.concat([ma20, ma60, ma120], axis=1).tail(20).dropna()
        if ma_recent.empty:
            ma_spread_mean = None
            ma_spread_std = None
        else:
            spread = (ma_recent.max(axis=1) - ma_recent.min(axis=1)) / (ma_recent.min(axis=1) + EPSILON)
            ma_spread_mean = float(spread.mean())
            ma_spread_std = float(spread.std(ddof=0)) if len(spread) > 1 else 0.0

        if "turnover_rate" in df_120.columns and df_120["turnover_rate"].notna().any():
            base_turn = float(df_120["turnover_rate"].mean())
            recent_turn = float(df["turnover_rate"].tail(10).mean())
            turn_mult = recent_turn / (base_turn + EPSILON)
        else:
            base_vol = float(df_120["volume"].mean())
            recent_vol = float(df["volume"].tail(10).mean())
            turn_mult = recent_vol / (base_vol + EPSILON)

        ignition = 1.2 <= turn_mult <= 3.0
        already_breakout = close > box_top_prev
        extended_breakout = close >= (box_top_prev * 1.03)
        breakout_window_ok = (not already_breakout) and (0.0 <= dist_to_top <= 0.03)
        ready = ignition and breakout_window_ok and close >= float(df["close"].rolling(20).mean().iloc[-1])

        signals[ts_code] = {
            "name": name,
            "box_top": box_top,
            "box_top_prev": box_top_prev,
            "box_bottom": box_bottom,
            "amplitude_120": amplitude,
            "close": close,
            "dist_to_box_top": dist_to_top,
            "close_position": pos,
            "turnover_mult": turn_mult,
            "ma_spread_mean_20": ma_spread_mean,
            "ma_spread_std_20": ma_spread_std,
            "ignition": ignition,
            "breakout_window_ok": breakout_window_ok,
            "already_breakout": already_breakout,
            "extended_breakout": extended_breakout,
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


def build_deterministic_core_table(candidates: pd.DataFrame, audits: List[AuditResult], top_n: int = 8) -> str:
    """Build deterministic core stock markdown table from ranked candidates."""
    if candidates is None or candidates.empty:
        return (
            "## 【核心金股】\n\n"
            "| 股票 | 所属主线 | 形态特征 | 置信度 | 推荐理由 |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
    audit_conf = {}
    for a in audits or []:
        audit_conf.setdefault(a.ts_code, []).append(float(a.confidence_score or 0.5))

    lines = [
        "## 【核心金股】",
        "",
        "| 股票 | 所属主线 | 形态特征 | 置信度 | 推荐理由 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in candidates.head(top_n).iterrows():
        ts_code = str(row.get("ts_code", ""))
        name = str(row.get("name", ts_code))
        matched = row.get("matched_themes", [])
        if not isinstance(matched, list):
            matched = []
        off_theme = bool(row.get("off_theme", not bool(matched)))
        if off_theme:
            theme_text = "**OFF-THEME（无题材匹配）**"
        else:
            theme_text = ", ".join(matched[:2]) if matched else "待确认"
        shape_parts = []
        if "consolidation_score" in row:
            shape_parts.append(f"横盘分{float(row.get('consolidation_score', 0.0)):.0f}")
        if "volume_boost" in row:
            shape_parts.append(f"量能{float(row.get('volume_boost', 0.0)):.2f}")
        if "filter_tier" in row:
            shape_parts.append(f"层级{row.get('filter_tier')}")
        shape = "，".join(shape_parts) if shape_parts else "技术形态待补充"
        conf_list = audit_conf.get(ts_code, [])
        confidence = float(np.mean(conf_list)) if conf_list else 0.5
        reason_parts = []
        if off_theme:
            reason_parts.append("该股票未匹配当前热点题材，仅基于技术形态入选")
        if "alpha_rank_score" in row:
            reason_parts.append(f"alpha评分{float(row.get('alpha_rank_score', 0.0)):.1f}")
        if "toplist_recency_score" in row:
            reason_parts.append(f"拥挤度{float(row.get('toplist_recency_score', 0.0)):.2f}")
        reason = "；".join(reason_parts) if reason_parts else "综合评分靠前"
        lines.append(f"| {name}({ts_code}) | {theme_text} | {shape} | {confidence:.2f} | {reason} |")
    return "\n".join(lines)


def upsert_core_table_in_report(report_md: str, table_section_md: str) -> str:
    """Replace existing core table section or insert one if missing."""
    if not report_md:
        return table_section_md
    pattern = r"##\s*【核心金股】[\s\S]*?(?=\n##\s*【|\Z)"
    if re.search(pattern, report_md):
        return re.sub(pattern, table_section_md + "\n\n", report_md, count=1)
    insert_after = re.search(r"##\s*【市场风向标】[\s\S]*?(?=\n##\s*【|\Z)", report_md)
    if insert_after:
        idx = insert_after.end()
        return report_md[:idx] + "\n\n" + table_section_md + "\n\n" + report_md[idx:]
    return table_section_md + "\n\n" + report_md


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

    # Convert audits to dict, handling dataclass fields properly
    audit_records = []
    for audit in audits:
        record = {
            "ts_code": audit.ts_code,
            "name": audit.name,
            "theme": audit.theme,
            "verdict": audit.verdict,
            "rationale": audit.rationale,
            "sources": audit.sources,
            "confidence_score": audit.confidence_score,
            "research_depth": audit.research_depth,
            "capital_signal_summary": audit.capital_signal_summary,
            "positive_findings": [
                {
                    "category": f.category,
                    "description": f.description,
                    "evidence": f.evidence[:200],
                    "confidence": f.confidence,
                    "source_url": f.source_url,
                    "date": f.date,
                }
                for f in (audit.positive_findings or [])
            ],
            "growth_catalysts": [
                {
                    "catalyst_type": c.catalyst_type,
                    "description": c.description,
                    "timeframe": c.timeframe,
                    "confidence": c.confidence,
                }
                for c in (audit.growth_catalysts or [])
            ],
        }
        audit_records.append(record)
    audit_df = pd.DataFrame(audit_records)

    # Ensure off_theme and filter_tier columns exist in candidates
    candidates_with_flag = candidates.head(15).copy()
    if "off_theme" not in candidates_with_flag.columns:
        candidates_with_flag["off_theme"] = False
    if "filter_tier" not in candidates_with_flag.columns:
        candidates_with_flag["filter_tier"] = "Unknown"

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
        "candidates": candidates_with_flag.to_dict("records"),
        "audits": audit_records,  # Use the enhanced audit records with findings
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
        "列：股票、所属主线、形态特征、置信度、推荐理由"
        ""
        "**置信度说明（confidence_score）：**\n"
        "- 0.8-1.0: 高置信度，多源验证充分\n"
        "- 0.6-0.8: 中等置信度，部分验证\n"
        "- 0.4-0.6: 低置信度，需要更多验证\n\n"
        ""
        "**重要：如果股票的off_theme字段为True，必须：**\n"
        "- 在\"所属主线\"列开头标注 **⚠️ OFF-THEME（无题材匹配）**\n"
        "- 在推荐理由中说明：\"该股票未匹配当前热点题材，仅基于技术形态入选\"\n"
        "- 在报告末尾添加【风险提示】说明OFF-THEME股票风险\n\n"
        ""
        "### 【深度图解】（每个标的必须包含）："
        ""
        "**【投资逻辑】**<font color='blue'>"
        "- 观察现象：量能异动、技术形态、题材契合"
        "- 分析意义：资金态度、趋势方向、突破可能"
        "- 验证方式：龙虎榜、财报、公告"
        "- 结论：交易机会评级（强烈推荐/推荐/谨慎）"
        "</font>"
        ""
        "**【正面催化发现】**<font color='green'>（从positive_findings提取）\n"
        "- 列出发现的正面信息（订单、客户、政策、技术突破、产能扩张等）\n"
        "- 每个发现标注类别和置信度\n"
        "- 引用来源URL\n"
        "</font>"
        ""
        "**【增长催化剂】**<font color='red'>（从growth_catalysts提取）\n"
        "- 催化剂类型：policy/tech_breakthrough/market_expansion/competitive_moat\n"
        "- 时间框架：near_term/medium_term/long_term\n"
        "- 置信度评估\n"
        "</font>"
        ""
        "**【技术分析】**横盘时长/波动率、量能信号、均线排列、箱体位置"
        "- ignition信号：是否处于温和放量阶段(1.2-3.0x)"
        "- ready_to_break信号：是否接近箱体突破"
        ""
        "**【资金验证】**<font color='purple'>（从capital_signal_summary提取）\n"
        "- 龙虎榜资金信号\n"
        "- 机构游资动向\n"
        "- 估值水平、市值适合度\n"
        "</font>"
        ""
        "**【交易建议】**<font color='green'>买入时机/仓位/止盈止损/持仓周期</font>"
        ""
        "**【风险提示】**<font color='orange'>核心风险及应对</font>"
        ""
        "- 量能异动日：[列表]\\n"
        "![股票名称 代码](../charts/代码.png)\\n"
        "- 尽调结论：pass/warn/fail（说明+来源）"
        "- 研究深度：standard/deep"
        ""
        "### 【风险提示】"
        "用<font color='orange'>橙色</font>标注核心风险"
        ""
        "**如果有OFF-THEME股票，必须额外添加风险说明：**\n"
        "<font color='orange'>\n"
        "**OFF-THEME股票风险提示：**\n"
        "- 该股票未匹配当前市场热点题材，仅基于技术形态入选\n"
        "- 缺乏题材催化，上涨动力可能不足\n"
        "- 建议谨慎对待，优先关注有题材匹配的标的\n"
        "</font>\n\n"
        ""
        "## 要求："
        "- 输出JSON：{\"final_report\":\"# Markdown...\"}"
        "- 引用真实URL，不使用占位符"
        "- 突出\"待时机\"：箱体上沿+温和放量(1.2-3.0x)+均线粘合"
        "- **在【市场风向标】中展示资金验证信息**：对于confirmed主题，必须展示龙虎榜资金信号"
        "- **在【深度图解】中展示正面催化发现**：从positive_findings和growth_catalysts中提取关键信息"
        "- **展示置信度分数**：在核心金股表格中显示confidence_score"
        ""
        "记住：先展示发现的机会（正面催化），再展示风险（审计结果）。"
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

    core_table_md = build_deterministic_core_table(candidates_with_flag, audits, top_n=min(10, len(candidates_with_flag)))
    report_md = upsert_core_table_in_report(report_md, core_table_md)
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
        # Avoid pandoc YAML misparse for bare '---' separators in body text.
        # Keep visual separator semantics using markdown horizontal rule '***'.
        if line.strip() == "---":
            processed_lines.append("***")
            continue

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

    # Convert HTML font color tags to pandoc raw LaTeX for PDF rendering
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
        # Escape LaTeX special characters in text (except backslash which is handled by raw attribute)
        # Only escape characters that would break the \textcolor{} command
        def escape_latex_text(s: str) -> str:
            # Escape braces first to avoid interfering with other replacements
            s = s.replace('{', '\\{').replace('}', '\\}')
            # Escape other LaTeX special characters
            replacements = [
                ('#', '\\#'),
                ('$', '\\$'),
                ('%', '\\%'),
                ('&', '\\&'),
                ('_', '\\_'),
                ('^', '\\^{}'),
                ('~', '\\textasciitilde{}'),
            ]
            for orig, repl in replacements:
                s = s.replace(orig, repl)
            return s
        text = escape_latex_text(text)
        # For multiline content, use fenced code block raw LaTeX
        # For single-line, use inline raw attribute syntax
        if '\n' in text:
            # Use {\color{name}...} instead of \textcolor for multiline content
            # because \textcolor cannot contain paragraph breaks
            return f'\n\n```{{=latex}}\n{{\\color{{{latex_color}}} {text}}}\n```\n\n'
        else:
            # Inline raw attribute syntax: `\textcolor{color}{text}`{=latex}
            return f'`\\textcolor{{{latex_color}}}{{{text}}}`{{=latex}}'
    # Use a more robust regex that handles nested structures
    processed_content = re.sub(
        r"<font\s+color=['\"]([^'\"]+)['\"]>(.*?)</font>",
        convert_font_color,
        processed_content,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Add spacing improvements
    processed_content = re.sub(r"\n{3,}", "\n\n", processed_content)  # Fix excessive blank lines

    # Fix relative image paths - convert ../charts/ to absolute paths for pandoc
    charts_abs_path = (md_path.parent.parent / "charts").resolve()
    processed_content = re.sub(
        r"!\[([^\]]*)\]\(\.\./charts/([^)]+)\)",
        lambda m: f"![{m.group(1)}]({charts_abs_path / m.group(2)})",
        processed_content
    )

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
        f'#!/usr/bin/env bash\ncd "{md_path.parent}" && pandoc --from=markdown+raw_attribute "{processed_md.name}" -o "{pdf_path.name}" --pdf-engine=xelatex -H header.tex -V CJKmainfont="{CHART_FONT}" --toc --number-sections\n',
        encoding="utf-8",
    )
    build_sh.chmod(0o755)

    try:
        # Run pandoc with improved options for better formatting
        subprocess.run(
            [
                "pandoc",
                "--from=markdown+raw_attribute",
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
    config = StrategyConfig.from_env()
    logger.info(f"Strategy config: horizon={config.holding_horizon}, toplist_mode={config.toplist_exclusion_mode}")

    llm = init_llm()

    # Phase 1
    logger.info("Phase 1: Market Intelligence...")
    themes = phase1_market_intel(llm)
    if not themes:
        logger.warning("No themes generated, check search tool or LLM output")

    # Phase 2
    candidates = phase2_quant_filter(themes, config=config)
    if candidates.empty:
        logger.error("Phase 2: No candidates after filtering")
        return

    # Phase 3
    logger.info("Phase 3: Deep Research (Opportunity Discovery + Adversarial Audit)...")
    audit_trace = REPORT_DIR / f"audit_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger.debug(f"Audit trace: {audit_trace}")
    audits = phase3_deep_audit(llm, candidates, trace_path=audit_trace, themes=themes, config=config)
    candidates, audits = apply_audit_filter(candidates, audits)
    if candidates.empty:
        logger.warning("No candidates passed audit filter")
        return

    # Alpha ranking before visualization/reporting
    signals = compute_signals(candidates)
    candidates = rank_candidates_for_alpha(candidates, audits, signals, config=config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates_export = REPORT_DIR / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    candidates.to_csv(candidates_export, index=False, encoding="utf-8-sig")
    logger.info(f"Exported ranked candidates: {candidates_export}")

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
