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
import math
import os
import random
import re
import time
import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deep_researcher import deepseek_chat, deepseek_plan_queries, zhipu_search
from screen_growth_stocks import (
    screen_all_stocks,
    compute_atr,
    compute_adx,
    compute_bollinger_width,
    compute_obv,
    compute_rsi,
    compute_ema,
    classify_valuation_label,
)
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
    theme_match_policy: str = "conservative"  # conservative|balanced|aggressive
    max_names_per_theme: int = 4
    max_names_per_industry: int = 4
    gemma_batch_size: int = 4
    gemma_rate_limit_max_retries: int = 6
    gemma_rate_limit_base_delay_sec: float = 1.0
    gemma_rate_limit_max_delay_sec: float = 20.0
    gemma_request_interval_sec: float = 0.35
    audit_llm_timeout_max_retries: int = 5
    audit_llm_timeout_base_delay_sec: float = 5.0
    audit_llm_timeout_max_delay_sec: float = 90.0
    valuation_mode: str = "blend"
    valuation_outlier_percentile: float = 0.97
    valuation_weight_screen: float = 0.12
    valuation_weight_alpha: float = 0.10
    valuation_allow_premium: bool = True

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        """Build configuration from environment variables."""
        theme_match_policy = os.environ.get("THEME_MATCH_POLICY", "conservative").strip().lower()
        if theme_match_policy not in {"conservative", "balanced", "aggressive"}:
            logger.warning(f"Invalid THEME_MATCH_POLICY='{theme_match_policy}', fallback to conservative")
            theme_match_policy = "conservative"
        return cls(
            holding_horizon=os.environ.get("HOLDING_HORIZON", "swing_2_8w"),
            toplist_exclusion_mode=os.environ.get("TOPLIST_EXCLUSION_MODE", "penalty"),
            toplist_penalty_weight=float(os.environ.get("TOPLIST_PENALTY_WEIGHT", "0.25")),
            toplist_lookback_days=int(os.environ.get("TOPLIST_LOOKBACK_DAYS", "60")),
            toplist_crowded_min_hits=int(os.environ.get("TOPLIST_CROWDED_MIN_HITS", "4")),
            hard_fail_require_recency=os.environ.get("HARD_FAIL_REQUIRE_RECENCY", "1").strip() in {"1", "true", "True", "YES", "yes"},
            hard_fail_max_age_days=int(os.environ.get("HARD_FAIL_MAX_AGE_DAYS", str(REGULATORY_MAX_AGE_DAYS))),
            hard_fail_reduce_materiality_threshold=float(os.environ.get("HARD_FAIL_REDUCE_MATERIALITY_THRESHOLD", "0.03")),
            theme_match_policy=theme_match_policy,
            max_names_per_theme=int(os.environ.get("MAX_NAMES_PER_THEME", "4")),
            max_names_per_industry=int(os.environ.get("MAX_NAMES_PER_INDUSTRY", "4")),
            gemma_batch_size=max(1, int(os.environ.get("GEMMA_BATCH_SIZE", "4"))),
            gemma_rate_limit_max_retries=max(0, int(os.environ.get("GEMMA_RATE_LIMIT_MAX_RETRIES", "6"))),
            gemma_rate_limit_base_delay_sec=max(0.0, float(os.environ.get("GEMMA_RATE_LIMIT_BASE_DELAY_SEC", "1.0"))),
            gemma_rate_limit_max_delay_sec=max(0.0, float(os.environ.get("GEMMA_RATE_LIMIT_MAX_DELAY_SEC", "20.0"))),
            gemma_request_interval_sec=max(0.0, float(os.environ.get("GEMMA_REQUEST_INTERVAL_SEC", "0.35"))),
            audit_llm_timeout_max_retries=max(0, int(os.environ.get("AUDIT_LLM_TIMEOUT_MAX_RETRIES", "5"))),
            audit_llm_timeout_base_delay_sec=max(0.0, float(os.environ.get("AUDIT_LLM_TIMEOUT_BASE_DELAY_SEC", "5.0"))),
            audit_llm_timeout_max_delay_sec=max(0.0, float(os.environ.get("AUDIT_LLM_TIMEOUT_MAX_DELAY_SEC", "90.0"))),
            valuation_mode=os.environ.get("VALUATION_MODE", "blend").strip().lower() or "blend",
            valuation_outlier_percentile=max(0.5, min(0.999, float(os.environ.get("VALUATION_OUTLIER_PERCENTILE", "0.97")))),
            valuation_weight_screen=max(0.0, float(os.environ.get("VALUATION_WEIGHT_SCREEN", "0.12"))),
            valuation_weight_alpha=max(0.0, float(os.environ.get("VALUATION_WEIGHT_ALPHA", "0.10"))),
            valuation_allow_premium=os.environ.get("VALUATION_ALLOW_PREMIUM", "1").strip() in {"1", "true", "True", "YES", "yes"},
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


@dataclass
class ChartArtifact:
    """Chart artifacts used by report renderers."""

    ts_code: str
    spike_dates: List[str]
    plotly_html: str
    png_rel_path: Optional[str] = None


@dataclass
class ReportThemeOverview:
    """Structured market overview item for rendering."""

    name: str
    validation_status: str
    logic: List[str]
    capital_validation: List[str]
    watch_items: List[str]
    source_urls: List[str]


@dataclass
class ReportStockSection:
    """Structured per-stock report section."""

    ts_code: str
    name: str
    matched_themes: List[str]
    recommendation: str
    recommendation_label: str
    research_depth: str
    summary: str
    investment_logic: List[str]
    positive_findings: List[PositiveFinding]
    growth_catalysts: List[GrowthCatalyst]
    technical_analysis: List[str]
    capital_validation: List[str]
    trade_plan: List[str]
    risks: List[str]
    source_urls: List[str]
    chart: Optional[ChartArtifact] = None
    audit_summaries: Optional[List[AuditResult]] = None

    def __post_init__(self):
        if self.audit_summaries is None:
            self.audit_summaries = []


@dataclass
class ReportModel:
    """Structured report model rendered into HTML and debug markdown."""

    title: str
    generated_at: str
    theme_overviews: List[ReportThemeOverview]
    core_table_rows: List[Dict[str, str]]
    theme_table_rows: List[Dict[str, str]]
    stock_sections: List[ReportStockSection]
    risks: List[str]


@dataclass
class ReportArtifacts:
    """Paths to report artifacts produced by Phase 5."""

    html_path: Path
    markdown_debug_path: Path
    trace_path: Path


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


def gemma_match_themes(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    relaxed_validation: bool = False,
    named_stock_codes: Optional[set] = None,
) -> pd.DataFrame:
    """
    Match stocks to themes using Gemma for semantic understanding.

    Args:
        themes: List of market themes to match against
        candidates: DataFrame of candidate stocks
        relaxed_validation: Use relaxed keyword validation rules
        named_stock_codes: Stock codes explicitly named in theme evidence (bypass validation)

    Returns:
        DataFrame with additional 'matched_themes' column
    """
    if not themes or candidates.empty:
        return candidates
    config = config or StrategyConfig.from_env()

    theme_list = [
        {"name": t.name, "keywords": t.keywords, "summary": t.summary}
        for t in themes
        if t.name
    ]
    if not theme_list:
        return candidates

    def get_text(row: pd.Series) -> str:
        """Get text representation of stock for theme matching."""
        main_business = str(row.get("main_business", "") or "")
        introduction = str(row.get("introduction", "") or "")
        business_scope = str(row.get("business_scope", "") or "")
        business_scope = re.sub(r"一般项目[:：]", "", business_scope)[:180]
        return " | ".join(
            [
                f"name={row.get('name','')}",
                f"code={row.get('ts_code','')}",
                f"industry={row.get('industry','')}",
                f"main_business={main_business}",
                f"business_scope（工商登记模板，非实际主营）={business_scope}",
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
    total_batches = 0
    rate_limited_batches = 0
    exhausted_batches = 0
    effective_retries_used = 0

    def is_rate_limit_error(err: Exception) -> bool:
        status_code = getattr(err, "status_code", None)
        response = getattr(err, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code == 429:
            return True
        name = type(err).__name__.lower()
        if "ratelimit" in name or "rate_limit" in name:
            return True
        message = str(err).lower()
        return "429" in message and "rate" in message and "limit" in message

    def retry_after_seconds(err: Exception) -> Optional[float]:
        response = getattr(err, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return None

    def validate_match(ts_code: str, picked: List[str]) -> List[str]:
        """
        Deterministic sanity guard:
        - require >=1 hit in main_business, OR
        - require >=2 hits in (main_business + introduction)
        This blocks matches that rely only on generic business_scope phrases.

        When relaxed_validation is True (outer scope):
        - Accept >=1 hit in (main_business + introduction + business_scope)
        - OR: stock was explicitly named in theme evidence (bypass keyword check)
        """
        row = row_by_code.get(str(ts_code))
        if row is None:
            return picked
        if relaxed_validation and named_stock_codes and str(ts_code) in named_stock_codes:
            return picked
        main_business = str(row.get("main_business", "") or "")
        introduction = str(row.get("introduction", "") or "")
        combined = f"{main_business} {introduction}"
        kept: List[str] = []
        for theme_name in picked:
            toks = theme_token_map.get(theme_name, set())
            if not toks:
                continue
            main_hits = sum(1 for tok in toks if tok in main_business)
            combined_hits = sum(1 for tok in toks if tok in combined)
            if relaxed_validation:
                business_scope = str(row.get("business_scope", "") or "")
                full_text = f"{combined} {business_scope}"
                full_hits = sum(1 for tok in toks if tok in full_text)
                if main_hits >= 1 or combined_hits >= 2 or full_hits >= 3:
                    kept.append(theme_name)
            else:
                if main_hits >= 1 or combined_hits >= 2:
                    kept.append(theme_name)
        if kept:
            logger.debug(f"validate_match {ts_code}: kept={kept} main_hits={main_hits} combined_hits={combined_hits}"
                         + (f" full_hits={full_hits}" if relaxed_validation else ""))
        return kept

    # Build all batch items (no cache — always send everything to LLM)
    batch = [
        {"ts_code": row["ts_code"], "text": get_text(row)}
        for _, row in updated.iterrows()
    ]

    _flush_lock = __import__("threading").Lock()

    def flush(batch_items: List[dict]) -> None:
        """Flush batch of stocks to LLM for theme matching."""
        nonlocal total_batches, rate_limited_batches, exhausted_batches, effective_retries_used
        if not batch_items:
            return
        with _flush_lock:
            total_batches += 1

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
                    "- business_scope为工商登记经营范围模板词，不反映实际业务，不可作为匹配依据\n"
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

        content = None
        max_attempts = config.gemma_rate_limit_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                content = invoke_llm_messages("gemma", messages, temperature=0.1)
                break
            except Exception as err:
                if not is_rate_limit_error(err):
                    raise
                with _flush_lock:
                    rate_limited_batches += 1
                    effective_retries_used += 1
                retries_used = attempt - 1
                if attempt >= max_attempts:
                    with _flush_lock:
                        exhausted_batches += 1
                    logger.warning(
                        "Gemma theme matching exhausted after rate limits: batch_size=%s retries=%s",
                        len(batch_items),
                        retries_used,
                    )
                    with _flush_lock:
                        for item in batch_items:
                            matched_map[item["ts_code"]] = []
                    return
                backoff_cap = min(
                    config.gemma_rate_limit_max_delay_sec,
                    config.gemma_rate_limit_base_delay_sec * (2 ** retries_used),
                )
                jitter = random.uniform(0.0, max(0.1, 0.2 * backoff_cap))
                computed_wait = min(config.gemma_rate_limit_max_delay_sec, backoff_cap + jitter)
                wait_seconds = retry_after_seconds(err) or computed_wait
                logger.warning(
                    "Gemma theme matching rate-limited (attempt=%s/%s, batch_size=%s); sleeping %.2fs",
                    attempt,
                    max_attempts,
                    len(batch_items),
                    wait_seconds,
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            finally:
                if config.gemma_request_interval_sec > 0:
                    time.sleep(config.gemma_request_interval_sec)

        if content is None:
            return
        parsed = safe_json_loads(content or "")
        raw_matches = parsed.get("matches", {}) if isinstance(parsed, dict) else {}
        raw_notes = parsed.get("notes", {}) if isinstance(parsed, dict) else {}

        # Build normalized name lookup: collapse whitespace so "AI 算力" matches "AI算力"
        _norm = lambda s: re.sub(r"\s+", "", s)
        norm_to_original = {_norm(t["name"]): t["name"] for t in theme_list}
        theme_name_set = {t["name"] for t in theme_list}
        batch_codes = {item["ts_code"] for item in batch_items}

        # Detect malformed response: some local models may return
        # {"matches": {"ts_code": ["600584.SH", ...]}} instead of
        # {"matches": {"600584.SH": ["theme1"], ...}}
        matches = {}
        if isinstance(raw_matches, dict):
            # Check if keys are stock codes or something else
            keys_are_codes = any(k in batch_codes for k in raw_matches.keys())
            if keys_are_codes:
                matches = raw_matches
            else:
                # Malformed: try to reconstruct from notes (which usually has correct format)
                # Also check if values are stock codes (inverted format)
                for key, val in raw_matches.items():
                    if isinstance(val, list) and val and all(str(v) in batch_codes for v in val):
                        # Inverted format: key is theme name, values are stock codes
                        theme_canonical = norm_to_original.get(_norm(str(key)), str(key))
                        if theme_canonical in theme_name_set:
                            for code in val:
                                matches.setdefault(str(code), []).append(theme_canonical)
                    elif isinstance(val, list) and val and any(str(v) in theme_name_set or _norm(str(v)) in norm_to_original for v in val):
                        # Normal format but key might be a stock code
                        matches[key] = val

                # Fallback: extract from notes if matches still empty
                if not matches and isinstance(raw_notes, dict):
                    for code in batch_codes:
                        note = raw_notes.get(code, "")
                        if isinstance(note, str):
                            for t_name in theme_name_set:
                                if t_name in note or _norm(t_name) in _norm(note):
                                    matches.setdefault(code, []).append(t_name)

                if matches:
                    logger.info(f"Recovered {len(matches)} matches from malformed Gemma response")

        for item in batch_items:
            ts_code = item.get("ts_code")
            picked = matches.get(ts_code, []) if isinstance(matches, dict) else []
            if not isinstance(picked, list):
                picked = []
            # Fuzzy-match theme names: normalize whitespace before comparing to whitelist
            resolved = []
            for x in picked:
                x_str = str(x)
                if x_str in theme_name_set:
                    resolved.append(x_str)
                else:
                    canonical = norm_to_original.get(_norm(x_str))
                    if canonical:
                        resolved.append(canonical)
            picked = validate_match(str(ts_code), resolved)
            with _flush_lock:
                matched_map[ts_code] = picked

    # Process in batches — up to 8 concurrent requests to local vLLM
    import concurrent.futures
    chunks = [batch[i:i + config.gemma_batch_size] for i in range(0, len(batch), config.gemma_batch_size)]
    max_workers = min(8, len(chunks)) if chunks else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(flush, chunks))
    if total_batches > 0:
        logger.info(
            "Gemma match stats: total_batches=%s rate_limited_batches=%s exhausted_batches=%s effective_retries_used=%s",
            total_batches,
            rate_limited_batches,
            exhausted_batches,
            effective_retries_used,
        )

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
        # Skip if resolved or favorable outcome
        if not re.search(r"(胜诉|判决获支持|已结案|已和解|已撤诉|已了结|调解结案)", text):
            # Require directional context — company is defendant/subject
            if re.search(r"(被诉|被告|被仲裁|遭.{0,6}诉讼|因.{0,6}诉讼|涉诉|涉及.{0,6}诉讼|面临.{0,6}诉讼)", text):
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
            "itertools",
            "statistics",
            "collections",
        }
        if name in allowed:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"import not allowed: {name}")

    def _normalize_python_obj(obj: Any, limit: int = 8) -> str:
        if isinstance(obj, pd.DataFrame):
            head = obj.head(limit)
            try:
                return head.to_markdown(index=False)
            except Exception:
                return head.to_string(index=False)
        if isinstance(obj, pd.Series):
            head = obj.head(limit)
            try:
                return head.to_frame(name="value").to_markdown()
            except Exception:
                return head.to_string()
        if isinstance(obj, (dict, list, tuple, set)):
            serializable = list(obj) if isinstance(obj, set) else obj
            try:
                return truncate(json.dumps(serializable, ensure_ascii=False, default=str, indent=2), 2000)
            except Exception:
                return truncate(repr(obj), 2000)
        return truncate(repr(obj), 2000)

    def to_df(obj: Any) -> pd.DataFrame:
        """Normalize common Python objects into a DataFrame."""
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, pd.Series):
            return obj.to_frame().T
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        if isinstance(obj, list):
            if not obj:
                return pd.DataFrame()
            if all(isinstance(item, dict) for item in obj):
                return pd.DataFrame(obj)
            return pd.DataFrame({"value": obj})
        return pd.DataFrame([{"value": obj}])

    def current_stock_df() -> pd.DataFrame:
        """Build a one-row DataFrame combining the stock profile and current signal row."""
        merged = {}
        if isinstance(context.get("stock_profile"), dict):
            merged.update(context["stock_profile"])
        if isinstance(context.get("signal_row"), dict):
            merged.update(context["signal_row"])
        return pd.DataFrame([merged]) if merged else pd.DataFrame()

    def audit_summary() -> Dict[str, Any]:
        """Flatten current-stock audits into a compact summary dict."""
        rows = context.get("audit_rows") or []
        if not isinstance(rows, list):
            rows = []
        verdicts: Dict[str, str] = {}
        findings: List[Dict[str, Any]] = []
        risks: List[str] = []
        sources: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            theme = str(row.get("theme", "")).strip()
            verdict = str(row.get("verdict", "")).strip()
            rationale = str(row.get("rationale", "")).strip()
            if theme:
                verdicts[theme] = verdict
            if rationale and verdict in {"warn", "fail"}:
                risks.append(f"{theme}: {rationale}" if theme else rationale)
            for finding in row.get("positive_findings") or []:
                if isinstance(finding, dict):
                    findings.append(
                        {
                            "theme": theme,
                            "category": finding.get("category"),
                            "description": finding.get("description"),
                            "confidence": finding.get("confidence"),
                            "source_url": finding.get("source_url"),
                        }
                    )
            for url in row.get("sources") or []:
                if isinstance(url, str) and url and url not in sources:
                    sources.append(url)
        return {
            "verdicts": verdicts,
            "positive_findings": findings[:12],
            "risks": risks[:12],
            "sources": sources[:20],
        }

    def recent_prices(ts_code: Optional[str] = None, days: int = 60) -> pd.DataFrame:
        """Load recent local parquet prices for a stock, defaulting to the current stock."""
        resolved_code = ts_code or context.get("stock_profile", {}).get("ts_code") or context.get("stock_data", {}).get("ts_code")
        if not resolved_code:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover_rate", "amount"])
        parquet_path = DATA_ROOT / "stock_ticks" / f"{resolved_code}.parquet"
        if not parquet_path.exists():
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover_rate", "amount"])
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover_rate", "amount"])
        rename_map = {"trade_date": "date", "vol": "volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "date" in df.columns:
            try:
                date_series = pd.to_datetime(df["date"])
                cutoff = date_series.max() - pd.Timedelta(days=max(1, int(days)))
                df = df.loc[date_series >= cutoff].copy()
            except Exception:
                pass
        columns = [col for col in ["ts_code", "date", "open", "high", "low", "close", "volume", "turnover_rate", "amount"] if col in df.columns]
        if columns:
            df = df[columns]
        if "date" in df.columns:
            try:
                df = df.sort_values("date", ascending=False)
            except Exception:
                pass
        return df.head(240)

    def show(obj: Any, limit: int = 8) -> str:
        """Pretty-print an object to stdout and return the rendered preview."""
        rendered = _normalize_python_obj(obj, limit=limit)
        print(rendered)
        return rendered

    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "next": next,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "len": len,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "ValueError": ValueError,
        "zip": zip,
        "any": any,
        "all": all,
    }

    local_ctx = dict(context)
    local_ctx["context"] = dict(context)
    local_ctx.setdefault("stock_profile", dict(context.get("stock_profile") or context.get("stock_data") or {}))
    local_ctx.setdefault("signal_row", dict(context.get("signal_row") or context.get("signals") or {}))
    local_ctx.setdefault("audit_rows", list(context.get("audit_rows") or context.get("audits") or []))
    local_ctx.setdefault("chart_notes", list(context.get("chart_notes") or []))
    local_ctx.setdefault("stock_data", local_ctx["stock_profile"])
    local_ctx.setdefault("signals", local_ctx["signal_row"])
    local_ctx.setdefault("audits", local_ctx["audit_rows"])
    local_ctx.setdefault("pd", pd)
    local_ctx.setdefault("np", np)
    local_ctx.setdefault("duckdb", duckdb)
    local_ctx.setdefault("json", json)
    local_ctx.setdefault("math", math)
    local_ctx.setdefault("datetime", datetime)
    local_ctx.setdefault("re", re)
    local_ctx.setdefault("show", show)
    local_ctx.setdefault("to_df", to_df)
    local_ctx.setdefault("current_stock_df", current_stock_df)
    local_ctx.setdefault("recent_prices", recent_prices)
    local_ctx.setdefault("audit_summary", audit_summary)
    local_ctx["__builtins__"] = safe_builtins

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, local_ctx, local_ctx)
    except Exception as exc:
        logger.warning(f"Python execution error: {exc}")
        available_vars = ["stock_profile", "signal_row", "audit_rows", "chart_notes", "candidates_df", "pd", "np", "duckdb"]
        utilities = ["show", "to_df", "current_stock_df", "recent_prices", "audit_summary"]
        extra_hint = ""
        if isinstance(exc, KeyError):
            sr_keys = sorted(local_ctx.get("signal_row", {}).keys())
            sp_keys = sorted(local_ctx.get("stock_profile", {}).keys())
            extra_hint = f" | signal_row_keys={sr_keys} | stock_profile_keys={sp_keys}"
        return (
            f"python_error[{type(exc).__name__}]: {exc} | "
            f"available_vars={', '.join(available_vars)} | "
            f"utilities={', '.join(utilities)}{extra_hint}"
        )

    output = stdout.getvalue().strip()
    result = local_ctx.get("result")
    if result is not None:
        rendered = _normalize_python_obj(result)
        logger.info(f"Python execution result: {truncate(rendered, 500)}")
        parts = []
        if output:
            parts.append(f"stdout:\n{output}")
        parts.append(f"result_type: {type(result).__name__}")
        parts.append(f"result_preview:\n{rendered}")
        return "\n".join(parts).strip()
    return output or "ok"


DUCKDB_REPO_TABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "stock_basic": {
        "path_parts": ("stock_basic", "stock_basic.parquet"),
        "columns": ["ts_code", "symbol", "name", "area", "industry", "list_date", "market", "exchange"],
        "sql": "CREATE VIEW stock_basic AS SELECT * FROM parquet_scan(?)",
    },
    "stock_company": {
        "path_parts": ("stock_company", "stock_company.parquet"),
        "columns": [
            "ts_code", "chairman", "manager", "secretary", "reg_capital", "setup_date", "province",
            "city", "introduction", "website", "employees", "main_business", "business_scope",
        ],
        "sql": "CREATE VIEW stock_company AS SELECT * FROM parquet_scan(?)",
    },
    "stock_ticks": {
        "path_parts": ("stock_ticks", "*.parquet"),
        "columns": [
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg",
            "vol", "amount", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
            "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv",
        ],
        "sql": "CREATE VIEW stock_ticks AS SELECT * FROM parquet_scan(?)",
    },
    "stock_basic_daily": {
        "path_parts": ("stock_ticks", "*.parquet"),
        "columns": ["ts_code", "date", "open", "high", "low", "close", "volume", "turnover_rate", "amount", "pe", "pb", "total_mv", "circ_mv"],
        "sql": (
            "CREATE VIEW stock_basic_daily AS "
            "SELECT ts_code, trade_date AS date, open, high, low, close, vol AS volume, "
            "turnover_rate, amount, pe, pb, total_mv, circ_mv "
            "FROM parquet_scan(?)"
        ),
    },
    "top_list": {
        "path_parts": ("top_list", "*.parquet"),
        "columns": [
            "trade_date", "ts_code", "name", "close", "pct_change", "turnover_rate", "amount",
            "l_sell", "l_buy", "l_amount", "net_amount", "net_rate", "amount_rate", "float_values", "reason",
        ],
        "sql": "CREATE VIEW top_list AS SELECT * FROM parquet_scan(?)",
    },
    "top_inst": {
        "path_parts": ("top_inst", "*.parquet"),
        "columns": ["trade_date", "ts_code", "exalter", "buy", "buy_rate", "sell", "sell_rate", "net_buy", "side", "reason"],
        "sql": "CREATE VIEW top_inst AS SELECT * FROM parquet_scan(?)",
    },
}


def _duckdb_repo_table_path(spec: Dict[str, Any]) -> Path:
    return DATA_ROOT.joinpath(*spec["path_parts"])


def _context_table_columns(value: Any) -> List[str]:
    if isinstance(value, pd.DataFrame):
        return [str(col) for col in value.columns]
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        seen: List[str] = []
        for item in value[:5]:
            for key in item.keys():
                skey = str(key)
                if skey not in seen:
                    seen.append(skey)
        return seen
    if isinstance(value, pd.Series):
        return [str(key) for key in value.index]
    return []


def _coerce_context_relation(value: Any) -> Optional[pd.DataFrame]:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, dict):
        return pd.DataFrame([value])
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return pd.DataFrame(value)
    return None


def _is_read_only_duckdb_sql(sql: str) -> bool:
    stripped = (sql or "").strip()
    if not stripped:
        return False
    if ";" in stripped.rstrip(";"):
        return False
    forbidden = re.compile(
        r"\b(insert|update|delete|merge|create|replace|alter|drop|copy|attach|detach|truncate|grant|revoke|call|export|import|use|set|install|load)\b",
        flags=re.IGNORECASE,
    )
    if forbidden.search(stripped):
        return False
    return bool(re.match(r"^\s*(select|with|show|describe|desc|explain|pragma)\b", stripped, flags=re.IGNORECASE))


def _extract_referenced_table_names(sql: str, table_names: List[str]) -> List[str]:
    referenced: List[str] = []
    for name in sorted(table_names, key=len, reverse=True):
        if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", sql, flags=re.IGNORECASE):
            referenced.append(name)
    return referenced


def _extract_sql_ts_code(sql: str) -> Optional[str]:
    match = re.search(r"\bts_code\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _infer_context_ts_code(context: Dict[str, Any]) -> Optional[str]:
    for key in ("stock_profile", "stock_data", "signal_row", "signals"):
        value = context.get(key)
        if isinstance(value, dict):
            ts_code = str(value.get("ts_code", "")).strip()
            if ts_code:
                return ts_code
    return None


def _resolve_repo_table_path(spec: Dict[str, Any], table_name: str, sql: str, context: Dict[str, Any]) -> Path:
    path = _duckdb_repo_table_path(spec)
    if table_name not in {"stock_ticks", "stock_basic_daily"}:
        return path
    ts_code = _extract_sql_ts_code(sql) or _infer_context_ts_code(context)
    if not ts_code:
        return path
    specific_path = DATA_ROOT / "stock_ticks" / f"{ts_code}.parquet"
    if specific_path.exists():
        return specific_path
    return path


def _register_repo_duckdb_tables(con: duckdb.DuckDBPyConnection, sql: str, context: Dict[str, Any]) -> List[str]:
    available: List[str] = []
    referenced = _extract_referenced_table_names(sql, list(DUCKDB_REPO_TABLE_SPECS.keys()))
    for table_name in referenced:
        spec = DUCKDB_REPO_TABLE_SPECS[table_name]
        path = _resolve_repo_table_path(spec, table_name, sql, context)
        if "*" in path.as_posix():
            if not list(path.parent.glob(path.name)):
                continue
        elif not path.exists():
            continue
        try:
            path_sql = path.as_posix().replace("'", "''")
            con.execute(spec["sql"].replace("?", f"'{path_sql}'"))
            available.append(table_name)
        except Exception as exc:
            logger.warning(f"Failed to register DuckDB repo table {table_name}: {exc}")
    return available


def _build_duckdb_schema_prompt(context: Dict[str, Any]) -> str:
    lines = [
        "DuckDB tool is read-only.",
        "Allowed SQL: SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, PRAGMA.",
        "Do not use INSERT/UPDATE/DELETE/CREATE/ALTER/DROP/COPY or multiple statements.",
        "DuckDB is mainly for raw row retrieval; do transformations in Python when possible.",
        "Available DuckDB tables and columns:",
    ]
    for name, value in context.items():
        columns = _context_table_columns(value)
        if columns:
            lines.append(f"- {name}({', '.join(columns)})")
    for name, spec in DUCKDB_REPO_TABLE_SPECS.items():
        path = _duckdb_repo_table_path(spec)
        exists = bool(list(path.parent.glob(path.name))) if "*" in path.as_posix() else path.exists()
        if exists:
            lines.append(f"- {name}({', '.join(spec['columns'])})")
    lines.append("Use stock_basic_daily for daily price queries with columns date/open/high/low/close/volume/turnover_rate.")
    lines.append("Use stock_ticks if you need raw trade_date/vol fields.")
    lines.append("stock_profile and signal_row are one-row current-stock context tables; query them directly without filtering by name.")
    return "\n".join(lines)


def _context_columns_map(context: Dict[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for name, value in context.items():
        columns = _context_table_columns(value)
        if columns:
            result[name] = columns
    return result


def _table_columns_for_error(context: Dict[str, Any], registered_names: List[str]) -> Dict[str, List[str]]:
    columns = _context_columns_map(context)
    for name in registered_names:
        if name in DUCKDB_REPO_TABLE_SPECS:
            columns[name] = list(DUCKDB_REPO_TABLE_SPECS[name]["columns"])
    return columns


def _suggest_duckdb_query(table_name: str) -> Optional[str]:
    if table_name == "stock_basic_daily":
        return "SELECT date, close, volume, turnover_rate FROM stock_basic_daily WHERE ts_code = '000001.SZ' ORDER BY date DESC LIMIT 20"
    if table_name == "stock_profile":
        return "SELECT * FROM stock_profile"
    if table_name == "signal_row":
        return "SELECT ts_code, close, dist_to_box_top, turnover_mult FROM signal_row"
    if table_name == "audit_rows":
        return "SELECT theme, verdict, confidence_score FROM audit_rows"
    return None


def _format_duckdb_error(exc: Exception, sql: str, context: Dict[str, Any], registered_names: List[str]) -> str:
    parts = [f"duckdb_error: {exc}"]
    if registered_names:
        parts.append(f"available_tables={', '.join(sorted(registered_names))}")
    referenced = _extract_referenced_table_names(sql, list(_context_columns_map(context).keys()) + list(DUCKDB_REPO_TABLE_SPECS.keys()))
    if referenced:
        table_name = referenced[0]
        table_columns = _table_columns_for_error(context, registered_names)
        parts.append(f"referenced_table={table_name}")
        if table_name in table_columns:
            parts.append(f"valid_columns={', '.join(table_columns[table_name])}")
        if table_name in {"signal_row", "stock_profile"}:
            parts.append("hint=this is a single current-stock row; query it directly instead of filtering by name or ts_code")
        suggestion = _suggest_duckdb_query(table_name)
        if suggestion:
            parts.append(f"suggestion={suggestion}")
    return " | ".join(parts)


def run_duckdb_sql(sql: str, context: Dict[str, Any]) -> str:
    """
    Execute DuckDB SQL query on registered DataFrames.

    Args:
        sql: SQL query to execute
        context: Dictionary of SQL-registerable objects plus auxiliary values

    Returns:
        Query results as markdown table or error message
    """
    stripped_sql = (sql or "").strip()
    if not _is_read_only_duckdb_sql(stripped_sql):
        return "duckdb_error: only read-only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN/PRAGMA queries are allowed"

    con = duckdb.connect()
    registered_names: List[str] = []
    for name, value in context.items():
        relation = _coerce_context_relation(value)
        if relation is not None:
            con.register(name, relation)
            registered_names.append(name)
    registered_names.extend(_register_repo_duckdb_tables(con, stripped_sql, context))

    try:
        df = con.execute(stripped_sql).df()
        head = df.head(20)
        try:
            return head.to_markdown(index=False)
        except Exception:
            return head.to_string(index=False)
    except Exception as exc:
        logger.warning(f"DuckDB query failed: {exc}")
        return _format_duckdb_error(exc, stripped_sql, context, registered_names)


def _execute_agent_tool(tool: str, tool_input: str, tool_context: Dict[str, Any]) -> str:
    """Execute an LLM-requested tool and always return a string result."""
    try:
        if tool == "web_search":
            return run_search(tool_input)
        if tool == "duckdb":
            return run_duckdb_sql(tool_input, tool_context)
        if tool == "python":
            return run_python(tool_input, tool_context)
        return f"tool_error: unknown_tool '{tool}'"
    except Exception as exc:
        logger.warning(f"Agentic tool execution failed: tool={tool} error={exc}")
        return f"tool_error: {tool} failed with {type(exc).__name__}: {exc}"


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


def extract_theme_named_stocks(
    themes: List[ThemeItem],
    screen_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Extract stock names explicitly mentioned in theme capital_signal and evidence fields.

    Returns:
        Dict mapping ts_code -> list of matched theme names
    """
    # Load stock_basic for name -> ts_code lookup
    basic_path = DATA_ROOT / "stock_basic"
    basic_files = list(basic_path.glob("*.parquet")) if basic_path.exists() else []
    if not basic_files:
        return {}
    basic_df = pd.concat([pd.read_parquet(f) for f in basic_files], ignore_index=True)
    if basic_df.empty or "name" not in basic_df.columns or "ts_code" not in basic_df.columns:
        return {}

    # Build name -> ts_code map (only for stocks in screen_df)
    screen_codes = set(screen_df["ts_code"].astype(str)) if "ts_code" in screen_df.columns else set()
    name_to_code: Dict[str, str] = {}
    for _, row in basic_df.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("ts_code", "")).strip()
        if name and code and len(name) >= 3 and code in screen_codes:
            name_to_code[name] = code

    result: Dict[str, List[str]] = {}
    for theme in themes:
        text = f"{theme.capital_signal or ''} {theme.evidence or ''}"
        if not text.strip():
            continue
        for stock_name, ts_code in name_to_code.items():
            if stock_name in text:
                result.setdefault(ts_code, [])
                if theme.name not in result[ts_code]:
                    result[ts_code].append(theme.name)

    return result


def phase2_quant_filter(themes: List[ThemeItem], config: Optional[StrategyConfig] = None) -> pd.DataFrame:
    """
    Phase 2: Dual-list stock selection.

    Always produces two complementary lists:
    1. Theme-Driven List (max 5): Stocks matching identified themes with relaxed filters
    2. Technical Alpha List (fills to 10 total): Best consolidation/squeeze candidates

    Args:
        themes: List of market themes for matching (should have validation_status)

    Returns:
        DataFrame of filtered candidate stocks with 'list_type' column
    """
    logger.info("Starting Phase 2: Dual-List Stock Selection")
    config = config or StrategyConfig.from_env()
    TOTAL_CAP = 10
    THEME_CAP = 5

    def normalize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "matched_themes" not in out.columns:
            out["matched_themes"] = [[] for _ in range(len(out))]
        out["matched_themes"] = out["matched_themes"].map(lambda x: x if isinstance(x, list) else [])
        out["off_theme"] = out["matched_themes"].map(lambda arr: len(arr) == 0)
        return out

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

    # Build toplist recency features for overcrowding control
    dtl = DragonTigerList()
    toplist_hit_counts = extract_toplist_hit_counts(dtl, days=config.toplist_lookback_days)
    logger.info(f"Loaded toplist hit counts for {len(toplist_hit_counts)} stocks")

    screen_df = screen_all_stocks()
    if screen_df is None or screen_df.empty:
        logger.warning("No screening results from screen_all_stocks()")
        return pd.DataFrame()
    screen_df = ensure_valuation_columns(screen_df)
    theme_order_col = (
        "momentum_score"
        if "momentum_score" in screen_df.columns
        else "composite_score"
        if "composite_score" in screen_df.columns
        else None
    )

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

    # ============ Step A: Build Theme-Driven List (max 5) ============
    theme_list: pd.DataFrame = pd.DataFrame()
    if themes:
        logger.info("Step A: Building theme-driven list...")

        # A1: Extract stocks explicitly named in theme evidence
        named_stocks = extract_theme_named_stocks(themes, screen_df)
        named_codes = set(named_stocks.keys())
        logger.info(f"Found {len(named_codes)} stocks named in theme evidence")

        # A2: Minimal filter from screen_df — only volume_boost >= 0.5
        if theme_order_col:
            con_a = duckdb.connect()
            try:
                con_a.register("screen", screen_df)
                theme_pool = con_a.execute(
                    f"""
                    SELECT *
                    FROM screen
                    WHERE volume_boost >= 0.5
                    ORDER BY {theme_order_col} DESC
                    LIMIT 100
                    """
                ).df()
            finally:
                con_a.close()
        else:
            logger.warning("Theme pool ranking columns missing; falling back to volume filter without explicit ordering")
            if "volume_boost" in screen_df.columns:
                theme_pool = screen_df[screen_df["volume_boost"] >= 0.5].head(100).copy()
            else:
                theme_pool = screen_df.head(100).copy()

        # Ensure named stocks are in the pool even if they didn't make top-100 momentum
        if named_codes:
            named_in_pool = set(theme_pool["ts_code"].astype(str)) if not theme_pool.empty else set()
            missing_named = named_codes - named_in_pool
            if missing_named:
                extra = screen_df[screen_df["ts_code"].astype(str).isin(missing_named)]
                if not extra.empty:
                    theme_pool = pd.concat([theme_pool, extra], ignore_index=True)
                    logger.info(f"Added {len(extra)} named stocks missing from top-100 momentum pool")

        logger.info(f"Theme pool: {len(theme_pool)} candidates after minimal filter")

        if not theme_pool.empty:
            # A3: Run gemma_match_themes with ALL themes (confirmed + web_only + capital_only)
            # Theme strength scoring already accounts for validation_status differences
            theme_pool = gemma_match_themes(
                expanded_themes,
                theme_pool,
                config=config,
                relaxed_validation=True,
                named_stock_codes=named_codes,
            )

            # A4: Merge named stocks (assign their themes from evidence) BEFORE normalize
            if "matched_themes" not in theme_pool.columns:
                theme_pool["matched_themes"] = [[] for _ in range(len(theme_pool))]
            pool_codes = set(theme_pool["ts_code"].astype(str))
            for ts_code, theme_names in named_stocks.items():
                if str(ts_code) not in pool_codes:
                    logger.warning(f"Named stock {ts_code} not found in theme pool ({len(pool_codes)} stocks)")
                    continue
                mask = theme_pool["ts_code"].astype(str) == str(ts_code)
                idx = theme_pool.index[mask]
                existing = theme_pool.at[idx[0], "matched_themes"]
                if not isinstance(existing, list):
                    existing = []
                merged_themes = list(dict.fromkeys(existing + theme_names))
                for i in idx:
                    theme_pool.at[i, "matched_themes"] = merged_themes
                logger.info(f"Named stock merge: {ts_code} -> {merged_themes}")

            theme_pool = normalize_match_columns(theme_pool)
            gemma_matches = theme_pool["matched_themes"].apply(bool).sum()
            logger.info(f"Theme pool after Gemma + named stock merge: {gemma_matches} stocks with theme matches")

            # A5: Filter to only matched stocks
            matched_mask = theme_pool["matched_themes"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            theme_matched = theme_pool[matched_mask].copy()

            if not theme_matched.empty:
                # A6: Score with momentum-weighted formula
                theme_matched["theme_strength_score"] = theme_matched["matched_themes"].map(
                    lambda arr: max([theme_strength_map.get(str(t), 0.5) for t in arr], default=0.0)
                )
                momentum_col = "momentum_score" if "momentum_score" in theme_matched.columns else "composite_score"
                volume_col = "volume_quality_score" if "volume_quality_score" in theme_matched.columns else "volume_boost"
                toplist_penalty = theme_matched.get("toplist_recency_score", pd.Series(0.0, index=theme_matched.index))
                valuation_quality = theme_matched.get("valuation_quality_score", pd.Series(50.0, index=theme_matched.index))
                valuation_stretch = theme_matched.get("valuation_stretch_score", pd.Series(50.0, index=theme_matched.index))
                valuation_penalty = valuation_stretch / 100.0
                theme_matched["alpha_rank_score"] = (
                    theme_matched[momentum_col] * 0.30
                    + theme_matched["theme_strength_score"] * 20.0
                    + theme_matched[volume_col] * 0.15
                    + theme_matched["composite_score"] * 0.10
                    + valuation_quality * 0.20
                    - toplist_penalty * (config.toplist_penalty_weight * 10.0)
                    - valuation_penalty * 8.0
                )
                theme_matched = theme_matched.sort_values("alpha_rank_score", ascending=False)

                # A7: Take top 5, tag as theme_driven
                theme_list = theme_matched.head(THEME_CAP).copy()
                theme_list["list_type"] = "theme_driven"
                theme_list["filter_tier"] = "theme_driven"
                logger.info(f"Theme-driven list: {len(theme_list)} stocks selected")
            else:
                logger.info("No theme matches found in broader pool")
    else:
        logger.info("No themes available, skipping theme-driven list")

    # ============ Step B: Build Technical Alpha List (fills to 10) ============
    tech_budget = TOTAL_CAP - len(theme_list)
    logger.info(f"Step B: Building technical alpha list (budget={tech_budget})...")

    con_b = duckdb.connect()
    con_b.register("screen", screen_df)
    tech_pool = con_b.execute("""
        SELECT *
        FROM screen
        WHERE consolidation_score >= 50
        ORDER BY composite_score DESC
    """).df()
    con_b.close()
    logger.info(f"Technical pool: {len(tech_pool)} candidates after consolidation filter")

    if not tech_pool.empty:
        # Exclude stocks already in theme list
        if not theme_list.empty and "ts_code" in theme_list.columns:
            theme_codes = set(theme_list["ts_code"].astype(str))
            tech_pool = tech_pool[~tech_pool["ts_code"].astype(str).isin(theme_codes)]

        tech_pool = normalize_match_columns(tech_pool)
        if theme_list.empty:
            heuristic_passes = {
                "conservative": 0,
                "balanced": 2,
                "aggressive": 4,
            }.get(config.theme_match_policy, 0)
            heuristic_themes = expanded_themes or confirmed_themes or themes
            for _ in range(heuristic_passes):
                tech_pool = heuristic_match_themes(heuristic_themes, tech_pool)
            tech_pool = normalize_match_columns(tech_pool)
        tech_pool["theme_strength_score"] = 0.0
        tech_pool["alpha_rank_score"] = tech_pool["composite_score"]

        tech_list = tech_pool.head(tech_budget).copy()
        tech_list["list_type"] = "technical"
        tech_list["filter_tier"] = "OFF_THEME_FALLBACK" if theme_list.empty else "technical"
    else:
        tech_list = pd.DataFrame()

    # ============ Step C: Combine ============
    parts = []
    if not theme_list.empty:
        parts.append(theme_list)
    if not tech_list.empty:
        parts.append(tech_list)

    if not parts:
        logger.warning("No candidates from either list")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    # Mark stocks appearing in both lists
    if not theme_list.empty and not tech_list.empty:
        theme_codes = set(theme_list["ts_code"].astype(str)) if not theme_list.empty else set()
        tech_codes = set(tech_list["ts_code"].astype(str)) if not tech_list.empty else set()
        both_codes = theme_codes & tech_codes
        if both_codes:
            combined.loc[combined["ts_code"].astype(str).isin(both_codes), "list_type"] = "both"

    combined = normalize_match_columns(combined)
    theme_count = len(combined[combined["list_type"].isin(["theme_driven", "both"])])
    tech_count = len(combined[combined["list_type"].isin(["technical", "both"])])
    logger.info(
        f"Phase 2 complete: {len(combined)} total candidates "
        f"(theme_driven={theme_count}, technical={tech_count})"
    )
    return combined


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


def is_timeout_error(err: Exception) -> bool:
    """Best-effort timeout detection across OpenAI/httpx/httpcore wrappers."""
    name = type(err).__name__.lower()
    if "timeout" in name:
        return True
    cause = getattr(err, "__cause__", None)
    if cause is not None and cause is not err and is_timeout_error(cause):
        return True
    message = str(err).lower()
    return "timed out" in message or "timeout" in message


def invoke_with_timeout_retries(
    operation: Callable[[], str],
    *,
    description: str,
    max_retries: int,
    base_delay_sec: float,
    max_delay_sec: float,
) -> str:
    """Retry timeout-prone LLM calls with wider exponential backoff."""
    max_attempts = max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as err:
            if not is_timeout_error(err) or attempt >= max_attempts:
                raise
            retries_used = attempt - 1
            backoff_cap = min(max_delay_sec, base_delay_sec * (2 ** retries_used))
            jitter = random.uniform(0.0, max(0.5, 0.25 * backoff_cap))
            wait_seconds = min(max_delay_sec, backoff_cap + jitter)
            logger.warning(
                "%s timed out (attempt=%s/%s); sleeping %.2fs before retry",
                description,
                attempt,
                max_attempts,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"{description} failed without producing a result")


def clamp01(value: float) -> float:
    """Clamp numeric value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def ensure_valuation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill neutral valuation fields when upstream mocks or legacy data omit them."""
    if df is None or df.empty:
        return df
    out = df.copy()
    defaults = {
        "valuation_quality_score": 50.0,
        "valuation_stretch_score": 50.0,
        "valuation_label": "估值待补充",
        "valuation_outlier": False,
        "pe_percentile_industry": np.nan,
        "pb_percentile_industry": np.nan,
        "ps_ttm_percentile_industry": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    if "valuation_label" in out.columns:
        missing_label = out["valuation_label"].isna() | (out["valuation_label"].astype(str).str.strip() == "")
        out.loc[missing_label, "valuation_label"] = out.loc[missing_label, "valuation_stretch_score"].map(classify_valuation_label)
    return out


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
    ranked = ensure_valuation_columns(candidates)

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
        valuation_quality = clamp01(float(row.get("valuation_quality_score", 50.0)) / 100.0)
        valuation_stretch = clamp01(float(row.get("valuation_stretch_score", 50.0)) / 100.0)
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
            config.valuation_weight_alpha * valuation_quality +
            0.10 * finding_score +
            0.07 * catalyst_score +
            0.06 * source_quality +
            0.07 * audit_safe
            - 0.12 * overcrowding_penalty
            - (0.10 if config.valuation_allow_premium else 0.18) * valuation_stretch
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
                "valuation_quality_score": valuation_quality * 100.0,
                "valuation_stretch_score": valuation_stretch * 100.0,
                "valuation_label": str(row.get("valuation_label", classify_valuation_label(valuation_stretch * 100.0))),
                "alpha_rank_score": score_01 * 100.0,
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    ranked = ranked.merge(alpha_df, on="ts_code", how="left", suffixes=("", "_new"))
    for col in ["valuation_quality_score", "valuation_stretch_score", "valuation_label"]:
        new_col = f"{col}_new"
        if new_col in ranked.columns:
            ranked[col] = ranked[new_col]
            ranked = ranked.drop(columns=[new_col])
    if "alpha_rank_score_new" in ranked.columns:
        ranked["alpha_rank_score"] = ranked["alpha_rank_score_new"]
        ranked = ranked.drop(columns=["alpha_rank_score_new"])

    # Sort within each list_type group, then concatenate (theme first, then technical)
    has_list_type = "list_type" in ranked.columns
    if has_list_type:
        theme_part = ranked[ranked["list_type"].isin(["theme_driven", "both"])].sort_values("alpha_rank_score", ascending=False)
        tech_part = ranked[ranked["list_type"] == "technical"].sort_values("alpha_rank_score", ascending=False)
        ranked = pd.concat([theme_part, tech_part], ignore_index=True)
        # Drop duplicates (stocks in both lists already have list_type="both")
        ranked = ranked.drop_duplicates(subset="ts_code", keep="first")
    else:
        ranked = ranked.sort_values("alpha_rank_score", ascending=False)

    ranked = apply_diversification_constraints(
        ranked,
        max_per_theme=config.max_names_per_theme,
        max_per_industry=config.max_names_per_industry,
        target_n=10,
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
        deepseek_plan_opportunity_queries,
        OPPORTUNITY_QUERY_TEMPLATES,
    )

    config = config or StrategyConfig.from_env()
    top = candidates
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

            # Adaptive follow-up: if findings are sparse, use AI to plan targeted queries
            if len(positive_findings) < 2:
                logger.debug(f"Sparse findings ({len(positive_findings)}) for {name}, attempting adaptive follow-up")
                found_categories = {f.category for f in positive_findings}
                all_categories = list(OPPORTUNITY_QUERY_TEMPLATES.keys())
                evidence_gaps = [c for c in all_categories if c not in found_categories]

                current_findings_summary = "; ".join(
                    f.description[:100] for f in positive_findings
                ) if positive_findings else "暂无发现"

                followup_plan = deepseek_plan_opportunity_queries(
                    name=name,
                    theme=theme,
                    current_findings=current_findings_summary,
                    evidence_gaps=evidence_gaps[:3],
                )

                if followup_plan and followup_plan.get("queries"):
                    followup_queries = followup_plan["queries"][:5]
                    logger.debug(f"Adaptive follow-up: {len(followup_queries)} queries for {name}")
                    trace_append(trace_path, "opportunity_followup", {
                        "ts_code": row["ts_code"], "name": name,
                        "queries": followup_queries,
                        "focus_areas": followup_plan.get("focus_areas", []),
                    })

                    for fq in followup_queries:
                        raw = run_search(fq)
                        parsed = parse_search_payload(raw)
                        search_results = parsed.get("results", [])

                        for cat in evidence_gaps:
                            findings = extract_positive_findings(search_results, name, cat)
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

                        for result in search_results[:3]:
                            if isinstance(result, dict):
                                snippet = result.get("snippet", "")
                                if snippet and name in snippet:
                                    opportunity_evidence.append(f"[followup] {snippet[:200]}")

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

                output = invoke_with_timeout_retries(
                    lambda: audit_chain.invoke({
                        "name": name,
                        "theme": theme,
                        "results": json.dumps(merged, ensure_ascii=False),
                    }),
                    description=f"Phase 3 audit LLM call for {name}/{theme}",
                    max_retries=config.audit_llm_timeout_max_retries,
                    base_delay_sec=config.audit_llm_timeout_base_delay_sec,
                    max_delay_sec=config.audit_llm_timeout_max_delay_sec,
                )
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
    if not parquet_path.exists():
        logger.warning(f"Price parquet not found for {ts_code}: {parquet_path}")
        return pd.DataFrame()

    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT trade_date, open, high, low, close, vol, turnover_rate
            FROM parquet_scan(?)
            ORDER BY trade_date
            """,
            [str(parquet_path)],
        ).df()
    except Exception as exc:
        logger.warning(f"Failed to load price data for {ts_code}: {exc}")
        return pd.DataFrame()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame()
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


def phase4_plot_charts(candidates: pd.DataFrame) -> Dict[str, ChartArtifact]:
    """
    Generate report-ready chart artifacts using Plotly.

    Returns:
        Dictionary mapping stock codes to chart artifacts.
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_artifacts: Dict[str, ChartArtifact] = {}

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

        # Bollinger Bands (20-day, 2 std) for squeeze visualization
        if len(df) >= 20:
            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std(ddof=0)
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            if bb_upper.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=bb_upper,
                        mode="lines", name="BB Upper",
                        line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot"),
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=bb_lower,
                        mode="lines", name="BB Lower",
                        line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot"),
                        fill="tonexty",
                        fillcolor="rgba(200,200,200,0.1)",
                        showlegend=False,
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

        chart_path = CHART_DIR / f"{ts_code}.png"
        png_rel_path: Optional[str] = None
        try:
            fig.write_image(str(chart_path), scale=2)
            png_rel_path = (Path("..") / CHART_DIR / f"{ts_code}.png").as_posix()
            logger.debug(f"Generated chart PNG: {chart_path}")
        except Exception as exc:
            logger.warning(f"PNG chart export failed for {ts_code}: {exc}")

        chart_artifacts[ts_code] = ChartArtifact(
            ts_code=ts_code,
            spike_dates=spike_dates,
            plotly_html=pio.to_html(
                fig,
                include_plotlyjs=False,
                full_html=False,
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "scrollZoom": True,
                },
                div_id=f"chart-{re.sub(r'[^a-zA-Z0-9]+', '-', ts_code)}",
            ),
            png_rel_path=png_rel_path,
        )

    return chart_artifacts


def compute_signals(candidates: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    signals: Dict[str, Dict[str, object]] = {}
    for _, row in candidates.iterrows():
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

        # New indicator signals
        atr_series = compute_atr(df)
        atr_now = float(atr_series.iloc[-1]) if atr_series.notna().any() else None
        atr_60d_ago = float(atr_series.iloc[-60]) if len(atr_series) >= 60 and atr_series.iloc[-60] == atr_series.iloc[-60] else None
        atr_squeeze = (atr_now / (atr_60d_ago + EPSILON)) < 0.7 if (atr_now is not None and atr_60d_ago is not None) else False

        bbw_series = compute_bollinger_width(df)
        bbw_now = float(bbw_series.iloc[-1]) if bbw_series.notna().any() else None
        bbw_120 = bbw_series.tail(120).dropna()
        bbw_squeeze = False
        if len(bbw_120) > 1 and bbw_now is not None:
            bbw_pct = (bbw_now - float(bbw_120.min())) / (float(bbw_120.max()) - float(bbw_120.min()) + EPSILON)
            bbw_squeeze = bbw_pct < 0.2

        adx_series, _, _ = compute_adx(df)
        adx_value = float(adx_series.iloc[-1]) if adx_series.notna().any() else 20.0
        adx_5d_ago = float(adx_series.iloc[-5]) if len(adx_series) >= 5 and adx_series.iloc[-5] == adx_series.iloc[-5] else adx_value
        adx_inflecting = adx_value < 25 and (adx_value - adx_5d_ago) > 0

        obv_series = compute_obv(df)
        obv_recent = obv_series.tail(20)
        obv_accumulating = False
        if len(obv_recent) >= 2:
            obv_slope = float(np.polyfit(range(len(obv_recent)), obv_recent.values, 1)[0])
            obv_accumulating = obv_slope > 0

        rsi_series = compute_rsi(df)
        rsi_val = float(rsi_series.iloc[-1]) if rsi_series.notna().any() else 50.0

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
            "atr_squeeze": atr_squeeze,
            "bbw_squeeze": bbw_squeeze,
            "adx_value": adx_value,
            "adx_inflecting": adx_inflecting,
            "obv_accumulating": obv_accumulating,
            "rsi": rsi_val,
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
            "| 股票 | 所属主线 | 估值 | 形态特征 | 推荐理由 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for _, row in candidates.head(10).iterrows():
        themes_str = ", ".join(row.get("matched_themes", [])) or "待确认"
        shape = f"横盘分{row['consolidation_score']:.0f}, 量能{row['volume_boost']:.2f}"
        reason = f"市值{row['market_cap']/1e8:.1f}亿, 换手{row['avg_turnover']:.2f}"
        valuation = str(row.get("valuation_label", "估值待补充"))
        lines.append(f"| {row['name']}({row['ts_code']}) | {themes_str} | {valuation} | {shape} | {reason} |")

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


def build_deterministic_theme_table(candidates: pd.DataFrame, audits: List[AuditResult], top_n: int = 5) -> str:
    """Build deterministic theme-driven stock markdown table."""
    rows = build_theme_table_rows(candidates, audits, top_n=top_n)
    if not rows:
        return ""
    return _build_markdown_table(
        "## 【核心金股 - 题材驱动精选】",
        ["股票", "匹配题材", "题材强度", "估值", "动量评分", "Alpha评分"],
        rows,
    )


def build_deterministic_core_table(candidates: pd.DataFrame, audits: List[AuditResult], top_n: int = 8) -> str:
    """Build deterministic core stock markdown table from ranked candidates (technical alpha)."""
    rows = build_core_table_rows(candidates, audits, top_n=top_n)
    if not rows:
        return (
            "## 【核心金股 - 技术形态精选】\n\n"
            "| 股票 | 所属主线 | 估值 | 形态特征 | 置信度 | 推荐理由 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
        )
    return _build_markdown_table(
        "## 【核心金股 - 技术形态精选】",
        ["股票", "所属主线", "估值", "形态特征", "置信度", "推荐理由"],
        rows,
    )


def upsert_core_table_in_report(report_md: str, table_sections_md: str) -> str:
    """Replace existing core table sections or insert them if missing."""
    if not report_md:
        return table_sections_md
    # Remove any existing core table sections (both old and new format)
    for pattern in [
        r"##\s*【核心金股 - 技术形态精选】[\s\S]*?(?=\n##\s*【|\Z)",
        r"##\s*【核心金股 - 题材驱动精选】[\s\S]*?(?=\n##\s*【|\Z)",
        r"##\s*【核心金股】[\s\S]*?(?=\n##\s*【|\Z)",
    ]:
        report_md = re.sub(pattern, "", report_md)
    insert_after = re.search(r"##\s*【市场风向标】[\s\S]*?(?=\n##\s*【|\Z)", report_md)
    if insert_after:
        idx = insert_after.end()
        return report_md[:idx] + "\n\n" + table_sections_md + "\n\n" + report_md[idx:]
    return table_sections_md + "\n\n" + report_md


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_bullet_text(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", str(text or "").strip())
    cleaned = re.sub(r"\burl\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _coerce_lines(value: Any) -> List[str]:
    if value is None:
        return []
    items: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = (
                    item.get("description")
                    or item.get("summary")
                    or item.get("title")
                    or item.get("text")
                    or item.get("name")
                )
                if text:
                    items.append(str(text))
            elif item is not None:
                items.append(str(item))
    elif isinstance(value, str):
        chunks = [seg for seg in re.split(r"\n+", value) if seg.strip()]
        items.extend(chunks or [value])
    else:
        items.append(str(value))
    normalized: List[str] = []
    for item in items:
        cleaned = _clean_bullet_text(item)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _normalize_source_urls(*values: Any) -> List[str]:
    urls: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                candidate = candidate.get("source_url") or candidate.get("url") or candidate.get("source")
            for url in extract_urls(str(candidate)):
                if url not in urls:
                    urls.append(url)
    return urls


def _build_markdown_table(title: str, headers: List[str], rows: List[Dict[str, str]]) -> str:
    lines = [title, "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def build_theme_table_rows(candidates: pd.DataFrame, audits: List[AuditResult], top_n: int = 5) -> List[Dict[str, str]]:
    """Build deterministic theme-driven stock table rows."""
    if candidates is None or candidates.empty or "list_type" not in candidates.columns:
        return []
    theme_stocks = candidates[candidates["list_type"].isin(["theme_driven", "both"])]
    if theme_stocks.empty:
        return []
    rows: List[Dict[str, str]] = []
    for _, row in theme_stocks.head(top_n).iterrows():
        ts_code = str(row.get("ts_code", ""))
        name = str(row.get("name", ts_code))
        matched = row.get("matched_themes", [])
        if not isinstance(matched, list):
            matched = []
        rows.append(
            {
                "股票": f"{name}({ts_code})",
                "匹配题材": ", ".join(matched[:2]) if matched else "待确认",
                "题材强度": f"{_coerce_float(row.get('theme_strength_score', 0.0)):.2f}",
                "估值": str(row.get("valuation_label", "估值待补充")),
                "动量评分": f"{_coerce_float(row.get('momentum_score', row.get('composite_score', 0.0))):.1f}",
                "Alpha评分": f"{_coerce_float(row.get('alpha_rank_score', 0.0)):.1f}",
            }
        )
    return rows


def build_core_table_rows(candidates: pd.DataFrame, audits: List[AuditResult], top_n: int = 8) -> List[Dict[str, str]]:
    """Build deterministic technical-alpha stock table rows."""
    if candidates is None:
        return []
    tech_stocks = candidates[candidates["list_type"].isin(["technical", "both"])] if "list_type" in candidates.columns else candidates
    if tech_stocks is None or tech_stocks.empty:
        return []
    audit_conf: Dict[str, List[float]] = {}
    for audit in audits or []:
        audit_conf.setdefault(audit.ts_code, []).append(float(audit.confidence_score or 0.5))
    rows: List[Dict[str, str]] = []
    for _, row in tech_stocks.head(top_n).iterrows():
        ts_code = str(row.get("ts_code", ""))
        name = str(row.get("name", ts_code))
        matched = row.get("matched_themes", [])
        if not isinstance(matched, list):
            matched = []
        off_theme = bool(row.get("off_theme", not bool(matched)))
        shape_parts = []
        if "consolidation_score" in row:
            shape_parts.append(f"横盘分{_coerce_float(row.get('consolidation_score', 0.0)):.0f}")
        if "volume_boost" in row:
            shape_parts.append(f"量能{_coerce_float(row.get('volume_boost', 0.0)):.2f}")
        reason_parts = []
        if "alpha_rank_score" in row:
            reason_parts.append(f"alpha评分{_coerce_float(row.get('alpha_rank_score', 0.0)):.1f}")
        if "toplist_recency_score" in row:
            reason_parts.append(f"拥挤度{_coerce_float(row.get('toplist_recency_score', 0.0)):.2f}")
        conf_list = audit_conf.get(ts_code, [])
        rows.append(
            {
                "股票": f"{name}({ts_code})",
                "所属主线": "技术形态入选" if off_theme else (", ".join(matched[:2]) if matched else "待确认"),
                "估值": str(row.get("valuation_label", "估值待补充")),
                "形态特征": "，".join(shape_parts) if shape_parts else "技术形态待补充",
                "置信度": f"{(float(np.mean(conf_list)) if conf_list else 0.5):.2f}",
                "推荐理由": "；".join(reason_parts) if reason_parts else "综合评分靠前",
            }
        )
    return rows


_MARKET_OVERVIEW_SYSTEM_PROMPT = (
    "你是资深A股投研团队负责人，遵循\"重势、通过滤、待时机\"理念。\n"
    "只返回严格JSON，不要Markdown，不要代码块。\n"
    "输出格式："
    "{\"themes\":[{\"name\":\"主题名称\",\"validation_status\":\"confirmed|web_only|capital_only|weak\","
    "\"logic\":[\"主题逻辑要点\"],\"capital_validation\":[\"资金验证要点\"],"
    "\"watch_items\":[\"持续观察点\"],\"source_urls\":[\"https://...\"]}]}\n"
    "要求：对confirmed主题必须写出龙虎榜/资金验证；引用真实URL；数组为空时返回空数组。"
)

_STOCK_SECTION_SYSTEM_PROMPT_TEMPLATE = (
    "你是资深A股投研分析师，遵循\"重势、通过滤、待时机\"理念。\n"
    "只返回严格JSON，不要Markdown，不要代码块。\n"
    "如需补充信息，可先返回工具调用："
    "{\"tool\":\"web_search|duckdb|python\",\"input\":\"...\"}\n"
    "优先顺序：1) 直接使用上下文；2) 用python做检查、衍生指标、证据汇总、数据变换；3) 缺少原始历史行时再用duckdb；4) 缺少外部信息时用web_search。\n"
    "Python工具保持开放式，不限制写法。可用变量：stock_profile(dict), signal_row(dict), audit_rows(list[dict]), chart_notes(list), candidates_df(DataFrame), pd, np, duckdb, json, math, datetime, re。\n"
    "Python中额外提供可选工具：show(obj, limit), to_df(obj), current_stock_df(), recent_prices(ts_code=None, days=60), audit_summary()。它们只是方便函数，你也可以自由写任意Python代码。\n"
    "DuckDB主要用于原始历史行检索。signal_row 和 stock_profile 都是单行当前股票上下文，不是完整股票表，不要按 name/ts_code 再过滤它们。\n"
    "signal_row的字段：__SIGNAL_ROW_KEYS__\n"
    "stock_profile的字段：__STOCK_PROFILE_KEYS__\n"
    "示例Python调用1：{\"tool\":\"python\",\"input\":\"summary = audit_summary(); show(summary); result = summary\"}\n"
    "示例Python调用2：{\"tool\":\"python\",\"input\":\"df = recent_prices(days=60); show(df[['date','close','volume']].head(10)); result = df[['date','close','volume']].head(10)\"}\n"
    "示例DuckDB调用：{\"tool\":\"duckdb\",\"input\":\"SELECT date, close, volume, turnover_rate FROM stock_basic_daily WHERE ts_code = '000001.SZ' ORDER BY date DESC LIMIT 20\"}\n"
    "__DUCKDB_SCHEMA__\n"
    "最终输出格式："
    "{\"stock\":{\"ts_code\":\"000001.SZ\",\"name\":\"示例\","
    "\"recommendation\":\"strong_buy|buy|watch|avoid\",\"summary\":\"一句话摘要\","
    "\"investment_logic\":[\"投资逻辑\"],"
    "\"positive_findings\":[{\"category\":\"policy\",\"description\":\"...\",\"evidence\":\"...\","
    "\"confidence\":0.7,\"source_url\":\"https://...\",\"date\":\"2026-01-01\"}],"
    "\"growth_catalysts\":[{\"catalyst_type\":\"policy\",\"description\":\"...\","
    "\"timeframe\":\"near_term|medium_term|long_term\",\"confidence\":0.7}],"
    "\"technical_analysis\":[\"技术分析\"],\"capital_validation\":[\"资金验证\"],"
    "\"trade_plan\":[\"交易建议\"],\"risks\":[\"风险提示\"],"
    "\"source_urls\":[\"https://...\"],\"research_depth\":\"standard|deep\"}}\n"
    "要求：真实URL；机会先于风险；突出箱体上沿、温和放量、均线收敛与资金验证。"
)


def _build_stock_section_system_prompt(tool_context: Dict[str, Any]) -> str:
    signal_keys = ", ".join(sorted(tool_context.get("signal_row", {}).keys())) or "(empty)"
    profile_keys = ", ".join(sorted(tool_context.get("stock_profile", {}).keys())) or "(empty)"
    return (
        _STOCK_SECTION_SYSTEM_PROMPT_TEMPLATE
        .replace("__DUCKDB_SCHEMA__", _build_duckdb_schema_prompt(tool_context))
        .replace("__SIGNAL_ROW_KEYS__", signal_keys)
        .replace("__STOCK_PROFILE_KEYS__", profile_keys)
    )


def _infer_tool_status(tool: str, result: str) -> Tuple[str, Optional[str]]:
    text = str(result or "")
    if text.startswith("tool_error:"):
        match = re.search(r"failed with ([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*)", text)
        return "error", match.group(1) if match else "ToolError"
    if tool == "duckdb" and text.startswith("duckdb_error:"):
        match = re.search(r"duckdb_error:\s*([A-Za-z ]+Error)", text)
        if match:
            return "error", match.group(1).replace(" ", "")
        if "Binder Error" in text:
            return "error", "BinderError"
        if "IO Error" in text:
            return "error", "IOError"
        return "error", "DuckDBError"
    if tool == "python" and text.startswith("python_error["):
        match = re.search(r"python_error\[([^\]]+)\]", text)
        return "error", match.group(1) if match else "PythonError"
    return "success", None


def _build_tool_feedback_message(
    tool: str,
    tool_input: str,
    result: str,
    status: str,
    repeated_failure: bool,
    remaining_iterations: int = 0,
) -> str:
    guidance: List[str] = []
    if status == "error" and tool == "python":
        guidance.append(
            "Python运行失败。先检查可用变量(stock_profile, signal_row, audit_rows, chart_notes, candidates_df)；"
            "必要时用 show(...) 打印中间结果，或简化代码。"
        )
    elif status == "error" and tool == "duckdb":
        guidance.append(
            "DuckDB查询失败。请修正表/字段，或把后续变换改到Python里完成；"
            "DuckDB更适合拉取原始历史行，不适合复杂推导。"
        )
    elif status == "success" and tool == "duckdb":
        guidance.append("DuckDB已返回原始数据；如需汇总、筛选、打分，请优先改用Python完成。")
    elif status == "success" and tool == "python":
        guidance.append("Python已返回可用分析结果；若信息足够，请直接输出最终JSON。")
    if repeated_failure:
        guidance.append("你刚刚重复触发了相似错误，不要重复同类查询/代码；请更换工具或明显改变方案。")
    if remaining_iterations <= 1:
        guidance.append("剩余调用次数不多，请尽快返回最终JSON。")
    guidance.append("如果信息已经足够，请直接返回最终JSON。")
    return (
        f"TOOL_STATUS: {status}\n"
        f"TOOL_NAME: {tool}\n"
        f"TOOL_INPUT:\n{tool_input}\n\n"
        f"TOOL_RESULT:\n{result}\n\n"
        "NEXT_STEP:\n"
        + "\n".join(f"- {item}" for item in guidance)
    )


def _normalize_validation_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"confirmed", "web_only", "capital_only", "weak"}:
        return text
    return "weak"


def _normalize_recommendation(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().lower()
    if text in {"strong_buy", "strongbuy", "强烈推荐"} or "强烈" in text:
        return "strong_buy", "强烈推荐"
    if text in {"buy", "推荐"} or text == "pass":
        return "buy", "推荐"
    if text in {"avoid", "回避"} or text == "fail":
        return "avoid", "回避"
    return "watch", "观察"


def _domain_label(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url[:40]


def _render_html_list(items: List[str], empty_text: str = "暂无") -> str:
    if not items:
        return f"<p class=\"muted\">{html_escape(empty_text)}</p>"
    return "<ul class=\"bullet-list\">" + "".join(f"<li>{html_escape(item)}</li>" for item in items) + "</ul>"


def _render_source_links(urls: List[str]) -> str:
    if not urls:
        return "<p class=\"muted\">暂无来源</p>"
    links = []
    for url in urls:
        links.append(
            f"<a class=\"source-link\" href=\"{html_escape(url)}\" target=\"_blank\" rel=\"noreferrer\">"
            f"{html_escape(_domain_label(url))}</a>"
        )
    return "<div class=\"source-links\">" + "".join(links) + "</div>"


def _normalize_positive_findings(items: Any, fallback: Optional[List[PositiveFinding]] = None) -> List[PositiveFinding]:
    normalized: List[PositiveFinding] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        normalized.append(
            PositiveFinding(
                category=str(item.get("category", "other")).strip() or "other",
                description=description,
                evidence=str(item.get("evidence", "")).strip(),
                confidence=_coerce_float(item.get("confidence"), 0.5),
                source_url=(_normalize_source_urls(item.get("source_url"), item.get("url")) or [""])[0],
                date=str(item.get("date", "")).strip() or None,
            )
        )
    if normalized:
        return normalized
    return list(fallback or [])


def _normalize_growth_catalysts(items: Any, fallback: Optional[List[GrowthCatalyst]] = None) -> List[GrowthCatalyst]:
    normalized: List[GrowthCatalyst] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        normalized.append(
            GrowthCatalyst(
                catalyst_type=str(item.get("catalyst_type", "market_expansion")).strip() or "market_expansion",
                description=description,
                timeframe=str(item.get("timeframe", "medium_term")).strip() or "medium_term",
                confidence=_coerce_float(item.get("confidence"), 0.5),
            )
        )
    if normalized:
        return normalized
    return list(fallback or [])


def _fallback_market_overview(themes: List[ThemeItem]) -> List[ReportThemeOverview]:
    items: List[ReportThemeOverview] = []
    for theme in themes:
        logic = _coerce_lines(theme.summary) or _coerce_lines(theme.evidence) or ["题材逻辑待补充。"]
        capital = _coerce_lines(theme.capital_signal) or ["龙虎榜资金信号待补充。"]
        watch_items = [f"持续跟踪 {theme.name} 的政策催化、成交额和资金持续性。"]
        items.append(
            ReportThemeOverview(
                name=theme.name,
                validation_status=_normalize_validation_status(theme.validation_status),
                logic=logic,
                capital_validation=capital,
                watch_items=watch_items,
                source_urls=_normalize_source_urls(theme.sources),
            )
        )
    return items


def _normalize_market_overview_item(raw: Optional[dict], fallback_theme: ThemeItem) -> ReportThemeOverview:
    raw = raw or {}
    return ReportThemeOverview(
        name=str(raw.get("name") or fallback_theme.name),
        validation_status=_normalize_validation_status(raw.get("validation_status", fallback_theme.validation_status)),
        logic=_coerce_lines(raw.get("logic")) or _coerce_lines(fallback_theme.summary) or ["题材逻辑待补充。"],
        capital_validation=_coerce_lines(raw.get("capital_validation")) or _coerce_lines(fallback_theme.capital_signal) or ["龙虎榜资金信号待补充。"],
        watch_items=_coerce_lines(raw.get("watch_items")) or [f"持续跟踪 {fallback_theme.name} 的政策催化与资金延续性。"],
        source_urls=_normalize_source_urls(raw.get("source_urls"), raw.get("logic"), fallback_theme.sources),
    )


def _build_stock_context(
    row: dict,
    stock_audits: List[AuditResult],
    chart_artifacts: Dict[str, ChartArtifact],
    signals: Dict[str, Dict[str, object]],
    theme_context: str,
) -> dict:
    """Build per-stock context dict for DeepSeek report generation."""
    ts_code = row["ts_code"]
    chart = chart_artifacts.get(ts_code)
    return {
        "ts_code": ts_code,
        "name": row.get("name", ts_code),
        "theme_context": theme_context,
        "stock_data": {k: v for k, v in row.items()},
        "audits": [
            {
                "theme": a.theme,
                "verdict": a.verdict,
                "rationale": a.rationale,
                "sources": a.sources,
                "confidence_score": a.confidence_score,
                "capital_signal_summary": a.capital_signal_summary,
                "positive_findings": [
                    {
                        "category": f.category,
                        "description": f.description,
                        "evidence": f.evidence[:200],
                        "confidence": f.confidence,
                        "source_url": f.source_url,
                        "date": f.date,
                    }
                    for f in (a.positive_findings or [])
                ],
                "growth_catalysts": [
                    {
                        "catalyst_type": c.catalyst_type,
                        "description": c.description,
                        "timeframe": c.timeframe,
                        "confidence": c.confidence,
                    }
                    for c in (a.growth_catalysts or [])
                ],
            }
            for a in stock_audits
        ],
        "chart_notes": chart.spike_dates if chart else [],
        "signals": signals.get(ts_code, {}),
    }


def _generate_market_overview(theme_summary: list, trace_path: Path, themes: List[ThemeItem]) -> List[ReportThemeOverview]:
    """Generate structured market overview via DeepSeek with fallback."""
    if not themes:
        return []
    messages = [
        {"role": "system", "content": _MARKET_OVERVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"themes": theme_summary}, ensure_ascii=False)},
    ]
    trace_append(trace_path, "overview_request", {})
    content = deepseek_chat(messages) if deepseek_chat else None
    if not content:
        return _fallback_market_overview(themes)
    trace_append(trace_path, "overview_response", {"content": truncate(content, 8000)})
    parsed = safe_json_loads(content)
    raw_items = parsed.get("themes") if isinstance(parsed, dict) else None
    if not isinstance(raw_items, list):
        return _fallback_market_overview(themes)
    by_name = {
        str(item.get("name")).strip(): item
        for item in raw_items
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    return [_normalize_market_overview_item(by_name.get(theme.name), theme) for theme in themes]


def _fallback_stock_section(
    row: dict,
    stock_audits: List[AuditResult],
    chart: Optional[ChartArtifact],
    signal: Optional[Dict[str, object]],
) -> ReportStockSection:
    """Deterministic stock section fallback when DeepSeek output is invalid."""
    ts_code = str(row.get("ts_code", ""))
    name = str(row.get("name", ts_code))
    matched = row.get("matched_themes", [])
    if not isinstance(matched, list):
        matched = []
    verdicts = {normalize_verdict(a.verdict) for a in stock_audits}
    if "fail" in verdicts:
        recommendation, label = ("avoid", "回避")
    elif bool(signal and signal.get("ready_to_break")):
        recommendation, label = ("buy", "推荐")
    else:
        recommendation, label = ("watch", "观察")
    spike_note = "、".join(chart.spike_dates[:6]) if chart and chart.spike_dates else "未检测到明显量能异动"
    technical_analysis = [
        f"量能异动日：{spike_note}",
        f"温和放量倍数：{_coerce_float((signal or {}).get('turnover_mult'), 0.0):.2f}",
        f"距箱体上沿：{_coerce_float((signal or {}).get('dist_to_box_top'), 0.0) * 100:.1f}%",
    ]
    capital_validation = _coerce_lines([a.capital_signal_summary for a in stock_audits if a.capital_signal_summary])
    if not capital_validation:
        capital_validation = [f"尽调结论：{a.theme} {normalize_verdict(a.verdict)} - {a.rationale}" for a in stock_audits[:3]] or ["资金验证待补充。"]
    risks = [
        f"{a.theme}: {a.rationale}"
        for a in stock_audits
        if normalize_verdict(a.verdict) in {"warn", "fail"}
    ] or ["题材轮动快，需关注市场主线是否持续。"]
    positive_findings: List[PositiveFinding] = []
    growth_catalysts: List[GrowthCatalyst] = []
    for audit in stock_audits:
        positive_findings.extend(audit.positive_findings or [])
        growth_catalysts.extend(audit.growth_catalysts or [])
    return ReportStockSection(
        ts_code=ts_code,
        name=name,
        matched_themes=matched,
        recommendation=recommendation,
        recommendation_label=label,
        research_depth="deep" if any(a.research_depth == "deep" for a in stock_audits) else "standard",
        summary=(stock_audits[0].rationale if stock_audits else "结构化回退内容，供HTML报告兜底渲染。"),
        investment_logic=[
            f"题材归属：{', '.join(matched[:3]) if matched else '技术形态入选'}",
            f"量能与位置：放量倍数 {_coerce_float((signal or {}).get('turnover_mult'), 0.0):.2f}，ready_to_break={bool((signal or {}).get('ready_to_break'))}",
        ],
        positive_findings=positive_findings[:5],
        growth_catalysts=growth_catalysts[:5],
        technical_analysis=technical_analysis,
        capital_validation=capital_validation,
        trade_plan=[
            "等待箱体上沿附近的确认信号再考虑分批介入。",
            "若放量失败或主线转弱，优先降低仓位。",
        ],
        risks=risks,
        source_urls=_normalize_source_urls([a.sources for a in stock_audits]),
        chart=chart,
        audit_summaries=list(stock_audits),
    )


def _normalize_stock_section_payload(
    payload: dict,
    row: dict,
    stock_audits: List[AuditResult],
    chart: Optional[ChartArtifact],
    signal: Optional[Dict[str, object]],
) -> ReportStockSection:
    fallback = _fallback_stock_section(row, stock_audits, chart, signal)
    recommendation, label = _normalize_recommendation(payload.get("recommendation"))
    positive_findings = _normalize_positive_findings(payload.get("positive_findings"), fallback.positive_findings)
    growth_catalysts = _normalize_growth_catalysts(payload.get("growth_catalysts"), fallback.growth_catalysts)
    source_urls = _normalize_source_urls(
        payload.get("source_urls"),
        [finding.source_url for finding in positive_findings],
        payload.get("summary"),
        [audit.sources for audit in stock_audits],
    ) or fallback.source_urls
    return ReportStockSection(
        ts_code=str(payload.get("ts_code") or row.get("ts_code", "")),
        name=str(payload.get("name") or row.get("name", "")),
        matched_themes=row.get("matched_themes", []) if isinstance(row.get("matched_themes", []), list) else [],
        recommendation=recommendation,
        recommendation_label=label,
        research_depth=str(payload.get("research_depth", fallback.research_depth)).strip() or fallback.research_depth,
        summary=_clean_bullet_text(str(payload.get("summary", fallback.summary)).strip()) or fallback.summary,
        investment_logic=_coerce_lines(payload.get("investment_logic")) or fallback.investment_logic,
        positive_findings=positive_findings,
        growth_catalysts=growth_catalysts,
        technical_analysis=_coerce_lines(payload.get("technical_analysis")) or fallback.technical_analysis,
        capital_validation=_coerce_lines(payload.get("capital_validation")) or fallback.capital_validation,
        trade_plan=_coerce_lines(payload.get("trade_plan")) or fallback.trade_plan,
        risks=_coerce_lines(payload.get("risks")) or fallback.risks,
        source_urls=source_urls,
        chart=chart,
        audit_summaries=list(stock_audits),
    )


def _generate_stock_section(
    ctx: dict,
    trace_path: Path,
    candidates_df: pd.DataFrame,
    row: dict,
    stock_audits: List[AuditResult],
    chart: Optional[ChartArtifact],
    tool_stats: Optional[Dict[str, Any]] = None,
) -> ReportStockSection:
    """Generate one structured stock section via DeepSeek with tool loop."""
    ts_code = ctx["ts_code"]
    name = ctx["name"]
    tool_context = {
        "candidates_df": candidates_df,
        "stock_profile": ctx.get("stock_data", {}),
        "signal_row": ctx.get("signals", {}),
        "audit_rows": ctx.get("audits", []),
        "chart_notes": ctx.get("chart_notes", []),
    }
    messages = [
        {"role": "system", "content": _build_stock_section_system_prompt(tool_context)},
        {"role": "user", "content": json.dumps(ctx, ensure_ascii=False, default=str)},
    ]
    last_failure_signature = None
    last_failure_tool = None
    repeated_failure_count = 0
    max_iterations = 5
    trace_append(trace_path, "stock_section_request", {"ts_code": ts_code, "name": name})
    for iteration in range(max_iterations):
        content = deepseek_chat(messages) if deepseek_chat else None
        if not content:
            break
        trace_append(trace_path, "stock_section_response", {"ts_code": ts_code, "content": truncate(content, 8000)})
        parsed = safe_json_loads(content)
        tool = parsed.get("tool") if isinstance(parsed, dict) else None
        if tool:
            tool_input = parsed.get("input", "")
            start = time.perf_counter()
            result = _execute_agent_tool(tool, tool_input, tool_context)
            duration_ms = int((time.perf_counter() - start) * 1000)
            status, error_class = _infer_tool_status(tool, result)
            logger.info(
                "Tool call: %s status=%s duration=%dms error=%s input=%s",
                tool, status, duration_ms, error_class, truncate(str(tool_input), 200),
            )
            failure_signature = None
            if status == "error":
                failure_signature = f"{tool}|{error_class}|{truncate(result, 180)}"
            repeated_failure = bool(
                status == "error"
                and last_failure_signature
                and failure_signature == last_failure_signature
            )
            if repeated_failure:
                repeated_failure_count += 1
                if repeated_failure_count >= 2:
                    logger.warning("Breaking tool loop for %s: repeated failure limit reached", ts_code)
                    break
            else:
                repeated_failure_count = 0
            trace_append(
                trace_path,
                "stock_tool_result",
                {
                    "ts_code": ts_code,
                    "tool": tool,
                    "status": status,
                    "duration_ms": duration_ms,
                    "input_preview": truncate(str(tool_input), 800),
                    "error_class": error_class,
                    "result": truncate(result, 4000),
                },
            )
            if tool_stats is not None:
                tool_stats["total_calls"] = int(tool_stats.get("total_calls", 0)) + 1
                per_tool = tool_stats.setdefault("per_tool", {})
                counters = per_tool.setdefault(tool, {"success": 0, "error": 0})
                counters[status] = int(counters.get(status, 0)) + 1
                if tool == "python" and last_failure_tool == "duckdb":
                    tool_stats["python_after_duckdb_failure"] = int(tool_stats.get("python_after_duckdb_failure", 0)) + 1
            remaining = max_iterations - iteration - 1
            messages.append(
                {
                    "role": "user",
                    "content": _build_tool_feedback_message(
                        tool=tool,
                        tool_input=str(tool_input),
                        result=result,
                        status=status,
                        repeated_failure=repeated_failure,
                        remaining_iterations=remaining,
                    ),
                }
            )
            if status == "error":
                last_failure_signature = failure_signature
                last_failure_tool = tool
            else:
                last_failure_signature = None
                last_failure_tool = None
            continue
        payload = None
        if isinstance(parsed, dict):
            candidate_payload = parsed.get("stock") or parsed.get("stock_section")
            if isinstance(candidate_payload, dict):
                payload = candidate_payload
        if payload is not None:
            return _normalize_stock_section_payload(payload, row, stock_audits, chart, ctx.get("signals"))
        messages.append({"role": "user", "content": "仅返回JSON对象 {\"stock\": {...}}，不要Markdown，不要代码块。"})
    return _fallback_stock_section(row, stock_audits, chart, ctx.get("signals"))


def _build_report_model(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    audits: List[AuditResult],
    chart_artifacts: Dict[str, ChartArtifact],
    signals: Dict[str, Dict[str, object]],
    trace_path: Path,
) -> ReportModel:
    audit_map: Dict[str, List[AuditResult]] = {}
    for audit in audits:
        audit_map.setdefault(audit.ts_code, []).append(audit)
    candidates_with_flag = candidates.copy()
    if "off_theme" not in candidates_with_flag.columns:
        candidates_with_flag["off_theme"] = False
    if "filter_tier" not in candidates_with_flag.columns:
        candidates_with_flag["filter_tier"] = "Unknown"
    if "list_type" not in candidates_with_flag.columns:
        candidates_with_flag["list_type"] = "technical"

    theme_summary = [
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
    ]
    logger.info("Phase 5: generating market overview data...")
    overview_items = _generate_market_overview(theme_summary, trace_path, themes)
    core_rows = build_core_table_rows(candidates_with_flag, audits, top_n=min(10, len(candidates_with_flag)))
    theme_rows = build_theme_table_rows(candidates_with_flag, audits, top_n=5)
    theme_context = "; ".join(f"{t.name}({t.validation_status})" for t in themes)
    stock_sections: List[ReportStockSection] = []
    stock_limit = min(10, len(candidates_with_flag))
    tool_stats: Dict[str, Any] = {"total_calls": 0, "per_tool": {}, "python_after_duckdb_failure": 0}
    for idx, (_, row) in enumerate(candidates_with_flag.head(stock_limit).iterrows()):
        ts_code = str(row["ts_code"])
        name = row.get("name", ts_code)
        stock_audits = audit_map.get(ts_code, [])
        chart = chart_artifacts.get(ts_code)
        ctx = _build_stock_context(row.to_dict(), stock_audits, chart_artifacts, signals, theme_context)
        logger.info(f"Phase 5: generating stock section {idx + 1}/{stock_limit} for {name}({ts_code})...")
        stock_sections.append(
            _generate_stock_section(ctx, trace_path, candidates_with_flag, row.to_dict(), stock_audits, chart, tool_stats=tool_stats)
        )
    logger.info(
        "Phase 5 tool summary: total_calls=%s python=%s duckdb=%s web_search=%s python_after_duckdb_failure=%s",
        tool_stats.get("total_calls", 0),
        tool_stats.get("per_tool", {}).get("python", {}),
        tool_stats.get("per_tool", {}).get("duckdb", {}),
        tool_stats.get("per_tool", {}).get("web_search", {}),
        tool_stats.get("python_after_duckdb_failure", 0),
    )
    return ReportModel(
        title="A股趋势跟踪研报",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        theme_overviews=overview_items,
        core_table_rows=core_rows,
        theme_table_rows=theme_rows,
        stock_sections=stock_sections,
        risks=[
            "题材轮动快，注意情绪退潮风险。",
            "量能异动需配合市场主线验证。",
            "若出现监管函、立案调查等硬伤，直接剔除。",
        ],
    )


def render_report_markdown_debug(report: ReportModel) -> str:
    """Render structured report as markdown for debugging."""
    lines = [f"# {report.title}", "", f"_生成时间：{report.generated_at}_", "", "## 【市场风向标】", ""]
    for item in report.theme_overviews:
        lines.append(f"### {item.name} ({item.validation_status})")
        for bullet in item.logic:
            lines.append(f"- 主题逻辑：{bullet}")
        for bullet in item.capital_validation:
            lines.append(f"- 资金验证：{bullet}")
        for bullet in item.watch_items:
            lines.append(f"- 持续观察：{bullet}")
        if item.source_urls:
            lines.append(f"- 来源：{', '.join(item.source_urls)}")
        lines.append("")
    lines.append(
        _build_markdown_table(
            "## 【核心金股 - 技术形态精选】",
            ["股票", "所属主线", "估值", "形态特征", "置信度", "推荐理由"],
            report.core_table_rows,
        )
        if report.core_table_rows
        else "## 【核心金股 - 技术形态精选】\n\n（无数据）"
    )
    if report.theme_table_rows:
        lines.extend(
            [
                "",
                _build_markdown_table(
                    "## 【核心金股 - 题材驱动精选】",
                    ["股票", "匹配题材", "题材强度", "估值", "动量评分", "Alpha评分"],
                    report.theme_table_rows,
                ),
            ]
        )
    lines.extend(["", "## 【深度图解】", ""])
    for section in report.stock_sections:
        lines.append(f"### {section.name} {section.ts_code}")
        lines.append(f"- 结论：{section.recommendation_label}")
        lines.append(f"- 摘要：{section.summary}")
        for bullet in section.investment_logic:
            lines.append(f"- 投资逻辑：{bullet}")
        for finding in section.positive_findings:
            lines.append(f"- 正面催化：[{finding.category}] {finding.description} (置信度 {finding.confidence:.2f})")
        for catalyst in section.growth_catalysts:
            lines.append(f"- 增长催化：[{catalyst.catalyst_type}] {catalyst.description} ({catalyst.timeframe}, {catalyst.confidence:.2f})")
        for bullet in section.technical_analysis:
            lines.append(f"- 技术分析：{bullet}")
        for bullet in section.capital_validation:
            lines.append(f"- 资金验证：{bullet}")
        for bullet in section.trade_plan:
            lines.append(f"- 交易建议：{bullet}")
        for bullet in section.risks:
            lines.append(f"- 风险提示：{bullet}")
        if section.chart and section.chart.png_rel_path:
            lines.append(f"![{section.name} {section.ts_code}]({section.chart.png_rel_path})")
        if section.source_urls:
            lines.append(f"- 来源：{', '.join(section.source_urls)}")
        lines.append("")
    lines.extend(["## 【风险提示】", *[f"- {risk}" for risk in report.risks], ""])
    return "\n".join(lines)


def _render_html_table(title: str, rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    thead = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{html_escape(str(row.get(header, '')))}</td>" for header in headers) + "</tr>")
    return (
        f"<section class=\"table-section\"><h2>{html_escape(title)}</h2><div class=\"table-shell\">"
        f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div></section>"
    )


def render_report_html(report: ReportModel) -> str:
    """Render structured report as a self-contained interactive HTML page."""
    all_theme_names = sorted(
        {
            theme.name for theme in report.theme_overviews
        }.union(
            {theme for section in report.stock_sections for theme in section.matched_themes}
        )
    )
    theme_options = "<option value=\"\">全部主题</option>" + "".join(
        f"<option value=\"{html_escape(theme)}\">{html_escape(theme)}</option>" for theme in all_theme_names
    )
    overview_cards = []
    for item in report.theme_overviews:
        overview_cards.append(
            "<article class=\"theme-card\">"
            f"<div class=\"card-head\"><h3>{html_escape(item.name)}</h3><span class=\"status {html_escape(item.validation_status)}\">{html_escape(item.validation_status)}</span></div>"
            "<div class=\"card-grid\">"
            f"<section><h4>主题逻辑</h4>{_render_html_list(item.logic)}</section>"
            f"<section><h4>资金验证</h4>{_render_html_list(item.capital_validation)}</section>"
            f"<section><h4>持续观察</h4>{_render_html_list(item.watch_items)}</section>"
            f"<section><h4>来源</h4>{_render_source_links(item.source_urls)}</section>"
            "</div></article>"
        )

    stock_cards = []
    for idx, section in enumerate(report.stock_sections):
        theme_badges = "".join(f"<span class=\"pill\">{html_escape(theme)}</span>" for theme in (section.matched_themes or ["技术形态入选"]))
        findings_html = "".join(
            "<article class=\"mini-card\">"
            f"<div class=\"mini-meta\">{html_escape(finding.category)} · 置信度 {finding.confidence:.2f}</div>"
            f"<p>{html_escape(finding.description)}</p>"
            f"{('<p class=\"muted\">' + html_escape(finding.evidence) + '</p>') if finding.evidence else ''}"
            f"{('<a class=\"source-link\" href=\"' + html_escape(finding.source_url) + '\" target=\"_blank\" rel=\"noreferrer\">来源</a>') if finding.source_url else ''}"
            "</article>"
            for finding in section.positive_findings
        ) or "<p class=\"muted\">暂无显著正面催化。</p>"
        catalysts_html = "".join(
            "<article class=\"mini-card\">"
            f"<div class=\"mini-meta\">{html_escape(catalyst.catalyst_type)} · {html_escape(catalyst.timeframe)} · {catalyst.confidence:.2f}</div>"
            f"<p>{html_escape(catalyst.description)}</p>"
            "</article>"
            for catalyst in section.growth_catalysts
        ) or "<p class=\"muted\">暂无显著增长催化。</p>"
        stock_cards.append(
            f"<details class=\"stock-card\" {'open' if idx < 2 else ''} "
            f"data-search=\"{html_escape((section.name + ' ' + section.ts_code + ' ' + ' '.join(section.matched_themes)).lower())}\" "
            f"data-themes=\"{html_escape('|'.join(section.matched_themes))}\">"
            "<summary>"
            f"<div><h3>{html_escape(section.name)} <span>{html_escape(section.ts_code)}</span></h3><p>{html_escape(section.summary)}</p></div>"
            f"<div class=\"summary-meta\"><span class=\"status rec-{html_escape(section.recommendation)}\">{html_escape(section.recommendation_label)}</span>{theme_badges}</div>"
            "</summary>"
            "<div class=\"details-grid\">"
            f"<section><h4>投资逻辑</h4>{_render_html_list(section.investment_logic)}</section>"
            f"<section><h4>技术分析</h4>{_render_html_list(section.technical_analysis)}</section>"
            f"<section><h4>资金验证</h4>{_render_html_list(section.capital_validation)}</section>"
            f"<section><h4>交易建议</h4>{_render_html_list(section.trade_plan)}</section>"
            "</div>"
            f"<section><h4>正面催化发现</h4><div class=\"mini-grid\">{findings_html}</div></section>"
            f"<section><h4>增长催化剂</h4><div class=\"mini-grid\">{catalysts_html}</div></section>"
            f"<section><h4>风险提示</h4>{_render_html_list(section.risks)}</section>"
            f"<section><h4>交互图表</h4>{section.chart.plotly_html if section.chart else '<p class=\"muted\">图表不可用。</p>'}</section>"
            f"<section><h4>来源</h4>{_render_source_links(section.source_urls)}</section>"
            "</details>"
        )

    css = """
    :root {
        --bg: #f5efe2;
        --paper: rgba(255, 252, 246, 0.92);
        --ink: #182028;
        --muted: #5f6770;
        --line: rgba(24, 32, 40, 0.14);
        --accent: #c4542b;
        --accent-2: #006d77;
        --shadow: 0 18px 45px rgba(39, 35, 31, 0.12);
        --radius: 18px;
        --font: "Avenir Next", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        color: var(--ink);
        background:
            radial-gradient(circle at top left, rgba(196, 84, 43, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 109, 119, 0.18), transparent 24%),
            linear-gradient(180deg, #f7f2e8 0%, #f1eadf 100%);
        font-family: var(--font);
    }
    a { color: var(--accent-2); }
    .layout { max-width: 1320px; margin: 0 auto; padding: 32px 20px 64px; }
    .hero {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: calc(var(--radius) + 8px);
        padding: 28px;
        box-shadow: var(--shadow);
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0 0 8px; font-size: clamp(32px, 4vw, 52px); line-height: 1.02; }
    .hero p { margin: 0; color: var(--muted); max-width: 900px; }
    .hero-meta { margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; }
    .pill, .status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.6);
    }
    .status.confirmed, .status.rec-strong_buy { background: rgba(1, 135, 134, 0.13); color: #005d63; }
    .status.web_only, .status.rec-buy { background: rgba(196, 84, 43, 0.13); color: #8b3a1d; }
    .status.capital_only, .status.rec-watch { background: rgba(58, 91, 166, 0.13); color: #28498c; }
    .status.weak, .status.rec-avoid { background: rgba(120, 53, 15, 0.12); color: #6d3d17; }
    .nav {
        position: sticky;
        top: 0;
        z-index: 20;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        background: rgba(245, 239, 226, 0.86);
        backdrop-filter: blur(12px);
        padding: 12px 0 16px;
    }
    .nav a, .controls button {
        text-decoration: none;
        background: var(--paper);
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 10px 14px;
        box-shadow: 0 8px 18px rgba(34, 28, 24, 0.08);
    }
    .controls {
        display: grid;
        grid-template-columns: 1fr 220px auto auto;
        gap: 12px;
        margin: 16px 0 24px;
    }
    .controls input, .controls select {
        width: 100%;
        border-radius: 14px;
        border: 1px solid var(--line);
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.85);
        font: inherit;
    }
    section, .theme-card, .stock-card, .table-section {
        scroll-margin-top: 88px;
    }
    .theme-grid, .mini-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }
    .theme-card, .table-section, .risk-section {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 20px;
        margin-bottom: 18px;
    }
    .card-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
        margin-bottom: 12px;
    }
    .card-head h3 { margin: 0; font-size: 22px; }
    .card-grid, .details-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .bullet-list { margin: 0; padding-left: 18px; }
    .bullet-list li { margin: 0 0 8px; }
    .muted { color: var(--muted); }
    .table-shell { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }
    th { font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
    .stock-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        margin-bottom: 18px;
        overflow: hidden;
    }
    .stock-card summary {
        list-style: none;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 22px;
        align-items: center;
    }
    .stock-card summary::-webkit-details-marker { display: none; }
    .stock-card summary h3 { margin: 0 0 6px; font-size: 24px; }
    .stock-card summary h3 span { color: var(--muted); font-size: 16px; }
    .stock-card summary p { margin: 0; color: var(--muted); }
    .summary-meta { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .stock-card > section, .stock-card > .details-grid { padding: 0 22px 22px; }
    .mini-card {
        background: rgba(255, 255, 255, 0.58);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px;
    }
    .mini-meta { font-size: 12px; color: var(--muted); margin-bottom: 8px; font-weight: 700; }
    .source-links { display: flex; flex-wrap: wrap; gap: 8px; }
    .source-link {
        display: inline-flex;
        align-items: center;
        padding: 7px 10px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.65);
        text-decoration: none;
    }
    .risk-list {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .risk-item {
        background: rgba(196, 84, 43, 0.08);
        border: 1px solid rgba(196, 84, 43, 0.2);
        border-radius: 14px;
        padding: 14px;
    }
    @media (max-width: 920px) {
        .controls { grid-template-columns: 1fr; }
        .stock-card summary { flex-direction: column; align-items: flex-start; }
        .summary-meta { justify-content: flex-start; }
    }
    @media print {
        body { background: #fff; }
        .nav, .controls { display: none !important; }
        .layout { max-width: none; padding: 0; }
        .hero, .theme-card, .table-section, .risk-section, .stock-card { box-shadow: none; border-color: #d4d4d4; }
        details { break-inside: avoid; }
        details[open] summary { margin-bottom: 12px; }
    }
    """
    script = """
    document.addEventListener('DOMContentLoaded', () => {
      const search = document.getElementById('stock-search');
      const theme = document.getElementById('theme-filter');
      const cards = Array.from(document.querySelectorAll('.stock-card'));
      const count = document.getElementById('stock-count');
      const applyFilters = () => {
        const term = (search.value || '').trim().toLowerCase();
        const activeTheme = theme.value || '';
        let visible = 0;
        cards.forEach((card) => {
          const hay = card.dataset.search || '';
          const themes = (card.dataset.themes || '').split('|').filter(Boolean);
          const matchesTerm = !term || hay.includes(term);
          const matchesTheme = !activeTheme || themes.includes(activeTheme);
          const show = matchesTerm && matchesTheme;
          card.style.display = show ? '' : 'none';
          if (show) visible += 1;
        });
        count.textContent = `${visible} / ${cards.length}`;
      };
      document.getElementById('expand-all').addEventListener('click', () => cards.forEach((card) => { if (card.style.display !== 'none') card.open = true; }));
      document.getElementById('collapse-all').addEventListener('click', () => cards.forEach((card) => { card.open = false; }));
      search.addEventListener('input', applyFilters);
      theme.addEventListener('change', applyFilters);
      applyFilters();
    });
    """
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape(report.title)}</title>
  <style>{css}</style>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <div class="layout">
    <header class="hero" id="top">
      <h1>{html_escape(report.title)}</h1>
      <p>单文件自包含 HTML 研报，包含结构化结论、交互筛选和内嵌 Plotly 图表。Markdown 仅保留为调试输出。</p>
      <div class="hero-meta">
        <span class="pill">生成时间 {html_escape(report.generated_at)}</span>
        <span class="pill">{len(report.theme_overviews)} 个主题</span>
        <span class="pill">{len(report.stock_sections)} 只股票</span>
      </div>
    </header>
    <nav class="nav">
      <a href="#overview">市场风向标</a>
      <a href="#core-table">技术形态精选</a>
      <a href="#theme-table">题材驱动精选</a>
      <a href="#stocks">深度图解</a>
      <a href="#risks">风险提示</a>
    </nav>
    <section class="controls" aria-label="筛选器">
      <input id="stock-search" type="search" placeholder="搜索股票、代码或题材" />
      <select id="theme-filter">{theme_options}</select>
      <button id="expand-all" type="button">展开可见项</button>
      <button id="collapse-all" type="button">收起全部</button>
    </section>
    <p class="muted">当前显示 <strong id="stock-count"></strong> 只股票。</p>
    <section id="overview">
      <h2>【市场风向标】</h2>
      <div class="theme-grid">{''.join(overview_cards)}</div>
    </section>
    <section id="core-table">{_render_html_table("【核心金股 - 技术形态精选】", report.core_table_rows)}</section>
    <section id="theme-table">{_render_html_table("【核心金股 - 题材驱动精选】", report.theme_table_rows)}</section>
    <section id="stocks">
      <h2>【深度图解】</h2>
      {''.join(stock_cards)}
    </section>
    <section class="risk-section" id="risks">
      <h2>【风险提示】</h2>
      <div class="risk-list">{''.join(f'<article class="risk-item">{html_escape(item)}</article>' for item in report.risks)}</div>
    </section>
  </div>
  <script>{script}</script>
</body>
</html>"""


def phase5_report_with_deepseek(
    themes: List[ThemeItem],
    candidates: pd.DataFrame,
    audits: List[AuditResult],
    chart_artifacts: Dict[str, ChartArtifact],
    signals: Dict[str, Dict[str, object]],
) -> ReportArtifacts:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"report_{timestamp}.html"
    md_path = REPORT_DIR / f"report_{timestamp}.md"
    trace_path = REPORT_DIR / f"deepseek_trace_{timestamp}.jsonl"

    report = _build_report_model(themes, candidates, audits, chart_artifacts, signals, trace_path)
    html_path.write_text(render_report_html(report), encoding="utf-8")
    md_path.write_text(render_report_markdown_debug(report), encoding="utf-8")
    if not trace_path.exists():
        trace_path.write_text("", encoding="utf-8")
    return ReportArtifacts(
        html_path=html_path,
        markdown_debug_path=md_path,
        trace_path=trace_path,
    )


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
    chart_artifacts = phase4_plot_charts(candidates)
    signals = compute_signals(candidates)

    # Phase 5
    logger.info("Phase 5: Report Generation...")
    report_artifacts = phase5_report_with_deepseek(themes, candidates, audits, chart_artifacts, signals)

    logger.info("=" * 60)
    logger.info(f"HTML report generated: {report_artifacts.html_path}")
    logger.info(f"Debug markdown generated: {report_artifacts.markdown_debug_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
