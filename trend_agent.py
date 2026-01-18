#!/usr/bin/env python3
import hashlib
import io
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

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
DEBUG_DEEPSEEK = os.environ.get("DEBUG_DEEPSEEK", "").strip() in {"1", "true", "True", "YES", "yes"}
USE_QWEN_THEME_MATCH = os.environ.get("USE_QWEN_THEME_MATCH", "").strip() in {"1", "true", "True", "YES", "yes"}
REGULATORY_MAX_AGE_DAYS = int(os.environ.get("REGULATORY_MAX_AGE_DAYS", "730"))

def setup_matplotlib_chinese_fonts() -> None:
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

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = CHART_FONT_FALLBACKS
    mpl.rcParams["axes.unicode_minus"] = False


setup_matplotlib_chinese_fonts()


def _truncate(text: str, limit: int = 5000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _pretty(payload: object) -> str:
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            return str(payload)
    return str(payload)


def ds_print(msg: str) -> None:
    if DEBUG_DEEPSEEK:
        print(f"[DeepSeek] {msg}")


def trace_append(path: Optional[Path], event: str, payload: object) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8") if not path.exists() else None
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(), "event": event, "payload": payload}, ensure_ascii=False) + "\n")
    except Exception:
        return


@dataclass
class ThemeItem:
    name: str
    keywords: List[str]
    summary: str
    sources: List[str]


@dataclass
class AuditResult:
    ts_code: str
    name: str
    theme: str
    verdict: str
    rationale: str
    sources: List[str]


def run_search(query: str) -> str:
    if hasattr(zhipu_search, "invoke"):
        return zhipu_search.invoke(query)
    if callable(zhipu_search):
        return zhipu_search(query)
    return "❌ Search tool unavailable."


def extract_urls(text: str) -> List[str]:
    urls = re.findall(r"(https?://[^\s\]\)\"']+|www\.[^\s\]\)\"']+)", text)
    unique = []
    for url in urls:
        if url.startswith("www."):
            url = "https://" + url
        if url not in unique:
            unique.append(url)
    return unique


def parse_search_payload(raw: object) -> Dict[str, object]:
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


def parse_result_date(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    if re.fullmatch(r"\d{8}", text):
        try:
            return datetime.strptime(text, "%Y%m%d")
        except Exception:
            return None
    return None


def is_recent(dt: Optional[datetime], max_age_days: int) -> bool:
    if dt is None:
        return False
    return dt >= (datetime.now() - timedelta(days=max_age_days))


def qwen_match_themes(
    themes: List[ThemeItem], candidates: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    if not USE_QWEN_THEME_MATCH or not themes or candidates.empty or not qwen_chat:
        return candidates

    cache: Dict[str, object] = {}
    try:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    theme_list = [
        {"name": t.name, "keywords": t.keywords, "summary": t.summary}
        for t in themes
        if t.name
    ]
    if not theme_list:
        return candidates

    def row_fingerprint(row: pd.Series) -> str:
        text = " ".join(
            [
                str(row.get("name", "")),
                str(row.get("industry", "")),
                str(row.get("main_business", "")),
                str(row.get("business_scope", "")),
                str(row.get("introduction", "")),
            ]
        )
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{row.get('ts_code','')}::{h}"

    def get_text(row: pd.Series) -> str:
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
                matched_map[row["ts_code"]] = [str(x) for x in val["matched"] if str(x) in [t["name"] for t in theme_list]]
            continue
        batch.append({"ts_code": row["ts_code"], "text": get_text(row)})
        batch_keys.append(key)

    def flush(batch_items: List[dict], batch_fp: List[str]) -> None:
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
            cache[fp] = {"matched": picked, "note": notes.get(ts_code) if isinstance(notes, dict) else ""}

    # chunk into small batches
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

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    updated["matched_themes"] = updated["ts_code"].map(lambda c: matched_map.get(c, []))
    return updated


def safe_json_loads(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    payload = match.group(0) if match else text
    payload = payload.strip().strip("```").strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def normalize_verdict(value: object) -> str:
    text = str(value or "").strip().lower()
    if "fail" in text or "否决" in text:
        return "fail"
    if "warn" in text or "warning" in text or "存疑" in text or "谨慎" in text:
        return "warn"
    if "pass" in text or "通过" in text:
        return "pass"
    if text in {"fail", "warn", "pass"}:
        return text
    return "warn"


PRIMARY_SOURCE_DOMAINS = (
    "cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
)

THEME_SYNONYMS: Dict[str, List[str]] = {
    "芯片": ["半导体", "集成电路", "国产替代", "IC", "EDA", "光刻", "存储"],
    "半导体": ["芯片", "集成电路", "国产替代", "IC", "EDA", "光刻", "存储"],
    "医药": ["医疗", "创新药", "生物制药", "疫苗", "CRO", "医药生物"],
    "医疗": ["医药", "创新药", "生物制药", "疫苗", "CRO", "医药生物"],
    "军工": ["航空", "航天", "军工电子", "航材", "兵装", "武器"],
    "AI": ["人工智能", "算力", "大模型", "数据中心", "服务器", "光模块", "CPO"],
    "新能源": ["光伏", "风电", "锂电", "储能", "新能源车", "充电桩"],
}


def expand_theme_keywords(theme: ThemeItem) -> List[str]:
    keywords = set([theme.name] + list(theme.keywords or []))
    expanded = set()
    for kw in list(keywords):
        expanded.add(kw)
        for key, syns in THEME_SYNONYMS.items():
            if key and key in kw:
                for s in syns:
                    expanded.add(s)
        if kw in THEME_SYNONYMS:
            for s in THEME_SYNONYMS[kw]:
                expanded.add(s)
    return [k for k in expanded if isinstance(k, str) and k.strip()]


def stock_symbol(ts_code: str) -> str:
    return str(ts_code or "").split(".")[0]


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
    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
    }
    local_ctx = dict(context)
    local_ctx["__builtins__"] = safe_builtins
    try:
        exec(code, local_ctx, local_ctx)
    except Exception as exc:
        return f"python_error: {exc}"
    output = ""
    result = local_ctx.get("result")
    if result is not None:
        return f"{output}\nresult: {result}".strip()
    return output or "ok"


def run_duckdb_sql(sql: str, context: Dict[str, pd.DataFrame]) -> str:
    con = duckdb.connect()
    for name, df in context.items():
        con.register(name, df)
    try:
        df = con.execute(sql).df()
        return df.head(20).to_markdown(index=False)
    except Exception as exc:
        return f"duckdb_error: {exc}"


def init_llm() -> ChatZhipuAI:
    return ChatZhipuAI(model="glm-4-flash", temperature=0.2)


def phase1_market_intel(llm: ChatZhipuAI) -> List[ThemeItem]:
    queries = [
        "A股 近两周 核心题材",
        "近期 龙虎榜 机构游资 重点板块",
        f"{datetime.now():%Y年%m月} A股 涨停复盘",
    ]

    raw_results = []
    all_urls = []
    for query in queries:
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

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是A股游资策略师，遵循“重势重质”原则，提炼当前市场3-5个核心主线并给出关键词。",
            ),
            (
                "user",
                "基于搜索结果，输出JSON："
                '{{"themes":[{{"name":"","keywords":["",""],"summary":"","sources":["url1"]}}],'
                '"market_summary":""}}\n\n搜索结果:\n{results}',
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
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
        if not theme.name:
            return False
        bad_tokens = ["股票", "辨识度", "传统经济", "蓝筹", "龙虎榜", "游资", "机构", "复盘", "涨停"]
        if any(tok in theme.name for tok in bad_tokens):
            return False
        if len(theme.name) > 12:
            return False
        return True

    filtered = [t for t in themes if is_actionable_theme(t)]
    return filtered or themes


def phase2_quant_filter(themes: List[ThemeItem]) -> pd.DataFrame:
    screen_df = screen_all_stocks()
    if screen_df is None or screen_df.empty:
        return pd.DataFrame()

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

    # Optional: use Qwen (small LLM) to match themes semantically to avoid brittle keyword matching.
    filtered = qwen_match_themes(themes, filtered, cache_path=Path(".cache/qwen_theme_match.json"))
    if "matched_themes" in filtered.columns:
        filtered = filtered.rename(columns={"matched_themes": "matched_themes_llm"})

    theme_keywords = []
    for theme in themes:
        expanded = expand_theme_keywords(theme)
        if theme.name and expanded:
            theme_keywords.append((theme.name, expanded))

    def match_row(row: pd.Series) -> Dict[str, List[str]]:
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
        print("⚠️ 形态符合的标的中没有命中当前主线白名单，返回 Top15 形态股（标记为 off_theme）供复核。")
        filtered["off_theme"] = True
        return filtered.head(15)

    filtered = filtered[filtered["matched_themes"].apply(bool)]
    filtered["off_theme"] = False
    return filtered.head(15)


def apply_audit_filter(
    candidates: pd.DataFrame, audits: List[AuditResult]
) -> tuple[pd.DataFrame, List[AuditResult]]:
    verdict_rank = {"fail": 2, "warn": 1, "pass": 0}
    worst = {}
    for audit in audits:
        rank = verdict_rank.get(audit.verdict, 1)
        prev = worst.get(audit.ts_code, -1)
        if rank > prev:
            worst[audit.ts_code] = rank

    def verdict_label(rank: int) -> str:
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
        print(f"Audit verdict distribution: {counts}")
    filtered = candidates[candidates["audit_verdict"] != "fail"]
    allowed_codes = set(filtered["ts_code"])
    filtered_audits = [audit for audit in audits if audit.ts_code in allowed_codes]
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
            ds_print(f"audit_start stock={name} theme={theme}")
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
                        ds_print(f"audit_plan pass={pass_id} plan=\n{_truncate(_pretty(plan), 1200)}")
                        trace_append(trace_path, "audit_plan", {"ts_code": row["ts_code"], "name": name, "theme": theme, "pass_id": pass_id, "plan": plan})
                    if plan and plan.get("stop"):
                        ds_print(f"audit_plan_stop pass={pass_id} reason={plan.get('reason','')}")
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
                    ds_print(f"audit_search pass={pass_id} q={query}")
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
                            "raw_snippet": _truncate(raw_clean, 1200),
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
                    ds_print("audit_hard_fail matched_patterns")
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
                ds_print(f"audit_llm verdict={verdict} rationale={_truncate(rationale, 400)}")
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
            ds_print(f"audit_done stock={name} theme={theme} verdict={verdict}")
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
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_notes = {}
    setup_matplotlib_chinese_fonts()
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
            savefig=str(chart_path),
            warn_too_much_data=CHART_DAYS + 10,
        )
        chart_notes[ts_code] = spike_dates
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
            }
            for t in themes
        ],
        "candidates": candidates.head(12).to_dict("records"),
        "audits": audit_df.to_dict("records"),
        "chart_notes": chart_notes,
        "signals": signals,
    }

    system_prompt = (
        "你是由A股游资策略师、量化研究员和数据可视化专家组成的投研团队负责人，遵循“重势、通过滤、待时机”。你可以请求工具来补充证据。"
        "工具调用格式为JSON："
        '{"tool":"web_search|duckdb|python","input":"..."}'
        "当你准备好输出最终报告时，返回JSON："
        '{"final_report":"# Markdown ..."}'
        "注意：引用来源必须使用提供的真实URL，不要写占位符。"
        "报告必须落到“待时机”的交易触发：箱体上沿附近 + 温和放量(1.2-3.0x) + 均线粘合。"
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
    ds_print(f"report_trace_path={trace_path}")
    trace_append(trace_path, "report_init", {"system": _truncate(system_prompt, 2000)})
    trace_append(trace_path, "report_context", {"summary_keys": list(summary.keys())})

    report_md = None
    tool_context = {
        "candidates": candidates,
        "audits": audit_df,
    }

    for _ in range(5):
        ds_print(f"report_round messages={len(messages)} last_user={_truncate(str(messages[-1].get('content','')), 600)}")
        trace_append(trace_path, "report_request", {"messages": [{"role": m.get("role"), "content": _truncate(str(m.get("content","")), 1200)} for m in messages[-3:]]})
        content = deepseek_chat(messages) if deepseek_chat else None
        if not content:
            ds_print("report_no_response")
            trace_append(trace_path, "report_no_response", {})
            break
        ds_print(f"report_raw_response={_truncate(content, 1200)}")
        trace_append(trace_path, "report_raw_response", {"content": _truncate(content, 12000)})
        parsed = safe_json_loads(content)
        tool = parsed.get("tool")
        if tool:
            if tool == "web_search":
                tool_input = parsed.get("input", "")
                ds_print(f"report_tool_call tool=web_search input={_truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "web_search", "input": tool_input})
                result = run_search(tool_input)
            elif tool == "duckdb":
                tool_input = parsed.get("input", "")
                ds_print(f"report_tool_call tool=duckdb input={_truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "duckdb", "input": tool_input})
                result = run_duckdb_sql(tool_input, tool_context)
            elif tool == "python":
                tool_input = parsed.get("input", "")
                ds_print(f"report_tool_call tool=python input={_truncate(str(tool_input), 400)}")
                trace_append(trace_path, "report_tool_call", {"tool": "python", "input": tool_input})
                result = run_python(tool_input, tool_context)
            else:
                result = "unknown_tool"
            ds_print(f"report_tool_result={_truncate(result, 800)}")
            trace_append(trace_path, "report_tool_result", {"tool": tool, "result": _truncate(result, 20000)})
            messages.append({"role": "user", "content": f"TOOL_RESULT:\n{result}"})
            continue
        report_md = parsed.get("final_report")
        if report_md:
            ds_print("report_final_received")
            trace_append(trace_path, "report_final_received", {"length": len(report_md)})
            break
        if content.lstrip().startswith("#"):
            report_md = content
            ds_print("report_markdown_fallback")
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


def build_pdf(md_path: Path) -> Optional[Path]:
    pdf_path = md_path.with_suffix(".pdf")
    build_sh = md_path.parent / "build.sh"
    build_sh.write_text(
        f"#!/usr/bin/env bash\npandoc {md_path} -o {pdf_path} --pdf-engine=xelatex -V CJKmainfont=\"{CHART_FONT}\"\n",
        encoding="utf-8",
    )
    build_sh.chmod(0o755)

    try:
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o",
                str(pdf_path),
                "--pdf-engine=xelatex",
                "-V",
                f"CJKmainfont={CHART_FONT}",
            ],
            check=True,
        )
        return pdf_path
    except Exception as exc:
        print(f"⚠️ Pandoc 生成失败: {exc}")
        return None


def main() -> None:
    llm = init_llm()
    print("Phase 1: Market Intelligence...")
    themes = phase1_market_intel(llm)
    if not themes:
        print("⚠️ 未生成题材列表，请检查搜索工具或LLM输出。")

    print("Phase 2: Quantitative Mining...")
    candidates = phase2_quant_filter(themes)
    if candidates.empty:
        print("⚠️ 初筛无结果：技术过滤或题材白名单匹配导致为空。")
        return

    print("Phase 3: Deep Research...")
    audit_trace = REPORT_DIR / f"audit_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    ds_print(f"audit_trace_path={audit_trace}")
    audits = phase3_deep_audit(llm, candidates, trace_path=audit_trace)
    candidates, audits = apply_audit_filter(candidates, audits)
    if candidates.empty:
        print("⚠️ 尽调后无合格标的。")
        return

    print("Phase 4: Visualization...")
    chart_notes = phase4_plot_charts(candidates)
    signals = compute_signals(candidates)

    print("Phase 5: Report...")
    md_path = phase5_report_with_deepseek(themes, candidates, audits, chart_notes, signals)
    pdf_path = build_pdf(md_path)

    print(f"✅ 报告已生成: {md_path}")
    if pdf_path:
        print(f"✅ PDF 已生成: {pdf_path}")
    else:
        print("⚠️ PDF 未生成，请检查 Pandoc。")


if __name__ == "__main__":
    main()
