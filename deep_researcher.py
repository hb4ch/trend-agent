import json
import logging
import os
import urllib.request
import re
import time
import dataclasses
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from zai import ZhipuAiClient
from langchain_core.tools import Tool

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


# Import forced LLM logging from llm_provider
try:
    from llm_provider import FORCE_LLM_LOGGING, format_messages_for_screen, format_response_for_screen, log_to_screen
except ImportError:
    # Fallback if llm_provider is not available
    FORCE_LLM_LOGGING = os.environ.get("FORCE_LLM_LOGGING", "").strip() in {"1", "true", "True", "YES", "yes"}
    format_messages_for_screen = None
    format_response_for_screen = None
    def log_to_screen(msg): print(msg, flush=True)


DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31B-nvfp4")
GEMMA_BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://192.168.3.46:8000/v1")
GEMMA_API_KEY = os.environ.get("GEMMA_API_KEY", "dummy")
GEMMA_TEMPERATURE = 1.0
GEMMA_TOP_P = 0.95
GEMMA_TOP_K = 64

DEBUG_DEEPSEEK = os.environ.get("DEBUG_DEEPSEEK", "").strip() in {"1", "true", "True", "YES", "yes"}
DEBUG_ZHIPU_SEARCH = os.environ.get("DEBUG_ZHIPU_SEARCH", "").strip() in {"1", "true", "True", "YES", "yes"}
DEBUG_OPENAI_COMPAT = (
    os.environ.get("DEBUG_OPENAI_COMPAT", "")
    .strip()
    in {"1", "true", "True", "YES", "yes"}
)

ZHIPU_SEARCH_ENGINE = os.environ.get("ZHIPU_SEARCH_ENGINE", "search_std")
ZHIPU_SEARCH_COUNT = int(os.environ.get("ZHIPU_SEARCH_COUNT", "15"))
ZHIPU_SEARCH_CONTENT_SIZE = os.environ.get("ZHIPU_SEARCH_CONTENT_SIZE", "high")
ZHIPU_SEARCH_RECENCY_FILTER = os.environ.get("ZHIPU_SEARCH_RECENCY_FILTER", "oneMonth")
ZHIPU_SEARCH_TIMEOUT_S = int(os.environ.get("ZHIPU_SEARCH_TIMEOUT_S", "60"))
ZHIPU_SEARCH_RETRIES = int(os.environ.get("ZHIPU_SEARCH_RETRIES", "3"))


def _truncate(text: str, limit: int = 8000) -> str:
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


def _debug_print(prefix: str, payload: object) -> None:
    if not DEBUG_DEEPSEEK:
        return
    try:
        print(f"[DeepSeek] {prefix}:\n{_truncate(_pretty(payload), 6000)}")
    except Exception:
        print(f"[DeepSeek] {prefix}:\n{_truncate(str(payload), 6000)}")


def _debug_zhipu(prefix: str, payload: object) -> None:
    if not DEBUG_ZHIPU_SEARCH:
        return
    try:
        print(f"[ZhipuSearch] {prefix}:\n{_truncate(_pretty(payload), 6000)}")
    except Exception:
        print(f"[ZhipuSearch] {prefix}:\n{_truncate(str(payload), 6000)}")


def _to_dict(obj: object) -> object:
    def deep_convert(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): deep_convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [deep_convert(v) for v in value]
        if isinstance(value, tuple):
            return [deep_convert(v) for v in value]

        if dataclasses.is_dataclass(value):
            try:
                return deep_convert(dataclasses.asdict(value))
            except Exception:
                pass

        # Try __dict__ FIRST for Pydantic models to avoid serialization warnings
        # This bypasses Pydantic's type validation during dump
        if hasattr(value, "__dict__"):
            try:
                raw = {
                    k: v
                    for k, v in value.__dict__.items()  # type: ignore[attr-defined]
                    if not str(k).startswith("_")
                }
                return deep_convert(raw)
            except Exception:
                pass

        if hasattr(value, "model_dump"):
            try:
                # Use warnings=False to suppress Pydantic serialization warnings
                # This happens when zai models have type mismatches (list vs single object)
                import warnings as pydantic_warnings
                with pydantic_warnings.catch_warnings():
                    pydantic_warnings.simplefilter("ignore")
                    return deep_convert(value.model_dump())  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return deep_convert(value.dict())  # type: ignore[attr-defined]
            except Exception:
                pass
        return value

    return deep_convert(obj)


def _walk(obj: object):
    stack = [obj]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                stack.append(v)


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("www."):
        return "https://" + url
    return url


def _extract_web_results(payload: object) -> List[dict]:
    results: List[dict] = []
    seen = set()
    direct_lists = []
    if isinstance(payload, dict):
        for key in ("data", "results", "list", "items", "search_result"):
            val = payload.get(key)
            if isinstance(val, list):
                direct_lists.append(val)
    for lst in direct_lists:
        for node in lst:
            if not isinstance(node, dict):
                continue
            url = node.get("url") or node.get("link") or node.get("href")
            if not isinstance(url, str):
                continue
            url = _normalize_url(url)
            if not url or url in seen:
                continue
            seen.add(url)
            title = node.get("title") if isinstance(node.get("title"), str) else ""
            snippet = (
                node.get("snippet")
                if isinstance(node.get("snippet"), str)
                else node.get("content") if isinstance(node.get("content"), str) else ""
            )
            date = ""
            for dk in (
                "date",
                "publish_date",
                "publish_time",
                "publishTime",
                "published_at",
                "time",
                "datetime",
            ):
                dv = node.get(dk)
                if isinstance(dv, str) and dv.strip():
                    date = dv.strip()
                    break
            results.append({"title": title, "url": url, "snippet": snippet, "date": date})
            if len(results) >= 10:
                return results

    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        url = node.get("url") or node.get("link") or node.get("href")
        if not isinstance(url, str):
            continue
        url = _normalize_url(url)
        if not url or url in seen:
            continue
        seen.add(url)
        title = node.get("title") if isinstance(node.get("title"), str) else ""
        snippet = (
            node.get("snippet")
            if isinstance(node.get("snippet"), str)
            else node.get("content") if isinstance(node.get("content"), str) else ""
        )
        date = node.get("date") if isinstance(node.get("date"), str) else ""
        results.append({"title": title, "url": url, "snippet": snippet, "date": date})
    return results[:10]


def _safe_json_from_text(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        val = json.loads(candidate)
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def deepseek_chat(messages: List[dict]) -> Optional[str]:
    return openai_compatible_chat(messages=messages, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, debug=DEBUG_DEEPSEEK, debug_prefix="DeepSeek")


def openai_compatible_chat(
    messages: List[dict],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    timeout_s: int = 300,
    debug: bool = False,
    debug_prefix: str = "OpenAICompat",
) -> Optional[str]:
    if not api_key:
        return None
    url_base = (base_url or "").rstrip("/")
    if not url_base:
        return None
    url = f"{url_base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k

    # Log input if forced logging is enabled
    if FORCE_LLM_LOGGING and format_messages_for_screen:
        log_to_screen(format_messages_for_screen(messages, model))

    if debug or DEBUG_OPENAI_COMPAT:
        try:
            print(
                f"[{debug_prefix}] request:\n"
                + _truncate(
                    _pretty(
                        {
                            "url": url,
                            "model": model,
                            "temperature": temperature,
                            "messages": [
                                {"role": m.get("role"), "content": _truncate(str(m.get("content", "")), 800)}
                                for m in messages
                            ],
                        }
                    ),
                    6000,
                )
            )
        except Exception:
            pass

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    max_retries = 4
    retry_backoff_s = 3
    msg_count = len(messages)
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            usage = body.get("usage", {})
            logger.info(
                "LLM call: model=%s msgs=%d elapsed=%dms tokens_in=%s tokens_out=%s",
                model, msg_count, elapsed_ms,
                usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"),
            )

            # Log output if forced logging is enabled
            if FORCE_LLM_LOGGING and format_response_for_screen:
                log_to_screen(format_response_for_screen(content, model))

            if debug or DEBUG_OPENAI_COMPAT:
                try:
                    print(f"[{debug_prefix}] response:\n" + _truncate(content, 4000))
                except Exception:
                    pass
            return content
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                retry_after = e.headers.get("Retry-After") if getattr(e, "headers", None) else None
                sleep_s = retry_backoff_s * (2 ** attempt)
                if retry_after:
                    try:
                        sleep_s = max(sleep_s, float(retry_after))
                    except Exception:
                        pass
                if debug or DEBUG_OPENAI_COMPAT:
                    print(f"[{debug_prefix}] HTTP 429 rate limited, retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_s)
                continue
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning("LLM call failed: model=%s msgs=%d elapsed=%dms error=HTTP_%d", model, msg_count, elapsed_ms, e.code)
            if debug or DEBUG_OPENAI_COMPAT:
                print(f"[{debug_prefix}] HTTP error: {e.code} - {e.reason}")
            return None
        except urllib.error.URLError as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning("LLM call failed: model=%s msgs=%d elapsed=%dms error=URLError", model, msg_count, elapsed_ms)
            if debug or DEBUG_OPENAI_COMPAT:
                print(f"[{debug_prefix}] URL error: {e.reason}")
            return None
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning("LLM call failed: model=%s msgs=%d elapsed=%dms error=%s", model, msg_count, elapsed_ms, type(e).__name__)
            if debug or DEBUG_OPENAI_COMPAT:
                print(f"[{debug_prefix}] error: {str(e)}")
            return None
    return None


def gemma_chat(messages: List[dict]) -> Optional[str]:
    return openai_compatible_chat(
        messages=messages,
        model=GEMMA_MODEL,
        api_key=GEMMA_API_KEY,
        base_url=GEMMA_BASE_URL,
        temperature=GEMMA_TEMPERATURE,
        top_p=GEMMA_TOP_P,
        top_k=GEMMA_TOP_K,
        debug=False,
        debug_prefix="Gemma",
    )


def deepseek_plan_queries(name: str, theme: str, evidence: str, pass_id: int) -> Optional[dict]:
    messages = [
        {
            "role": "system",
            "content": """你是A股研究协调员，负责决策是否还要继续进行调研任务。调研的目的是为了决策是否投资一只股票。
                你要根据当前已有的证据,判断是否还需要继续深挖。如果需要继续深挖,请列出后续需要调研的查询列表。这个查询列表会被用来进行后续的网络搜索和信息收集。
                如果你认为已有的证据已经足够,不需要继续深挖,请直接输出{"stop":true,"reason":"...","queries":[]}。严格按照json格式输出。"""
        },
        {
            "role": "user",
            "content": (
                "基于已有证据，判断是否需要继续深挖，并输出JSON："
                '{"stop":true|false,"reason":"","queries":["..."]}\n\n'
                f"股票:{name}\n题材:{theme}\nPass:{pass_id}\n已有证据:\n{evidence}"
            ),
        },
    ]
    _debug_print(
        "plan_queries_input",
        {"name": name, "theme": theme, "pass_id": pass_id, "evidence": _truncate(evidence, 1500)},
    )
    content = deepseek_chat(messages)
    if not content:
        return None
    parsed = _safe_json_from_text(content)
    if parsed is not None:
        _debug_print("plan_queries_output", parsed)
        return parsed
    _debug_print("plan_queries_parse_error", {"raw": _truncate(content, 2000)})
    return None

# 确保你的 .env 文件里是 ZHIPUAI_API_KEY
# 格式通常是: vector_... 或其他 (取决于你的 key 类型)

class ZhipuSearchTool:
    def __init__(self):
        # 显式初始化，确保读取环境变量
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("❌ 未找到 ZHIPUAI_API_KEY，请检查 .env 文件")
        self.client = ZhipuAiClient(api_key=api_key)

    def search(self, query: str, engine: str = "") -> str:
        """
        调用智谱 ZAI 原生 web_search 接口（更 raw、更稳定地拿到链接）

        Args:
            query: Search query string
            engine: Override search engine (e.g. "search_std"). Empty = use default.
        """
        print(f"📡 Zhipu/Zai Searching: {query}...")
        try:
            # Support "site:domain" shortcut by mapping it to domain filter.
            domain_filter = None
            cleaned_query = query
            m = re.search(r"site:([^\s]+)", query)
            if m:
                domain_filter = m.group(1).strip()
                cleaned_query = re.sub(r"site:[^\s]+", "", query).strip()

            search_engine = engine or ZHIPU_SEARCH_ENGINE
            kwargs = {
                "search_engine": search_engine,
                "search_query": cleaned_query,
                "count": max(1, min(50, ZHIPU_SEARCH_COUNT)),
                "search_recency_filter": ZHIPU_SEARCH_RECENCY_FILTER,
                "content_size": ZHIPU_SEARCH_CONTENT_SIZE,
            }
            if domain_filter:
                kwargs["search_domain_filter"] = domain_filter

            last_error = None
            resp_obj = None
            for attempt in range(1, max(1, ZHIPU_SEARCH_RETRIES) + 1):
                try:
                    response = self.client.web_search.web_search(**kwargs)
                    resp_obj = _to_dict(response)
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < ZHIPU_SEARCH_RETRIES:
                        time.sleep(0.5 * attempt)

            if resp_obj is None:
                normalized = {
                    "results": [],
                    "summary": "检索失败（连接/接口错误），请稍后重试。",
                    "error": last_error or "unknown_error",
                    "meta": {"query": cleaned_query, "domain_filter": domain_filter, **kwargs},
                }
                _debug_zhipu("normalized", normalized)
                return json.dumps(normalized, ensure_ascii=False)

            _debug_zhipu("raw_response", resp_obj)

            results = _extract_web_results(resp_obj)
            summary = "未找到相关链接。" if not results else ""

            normalized = {
                "results": results,
                "summary": summary,
                "meta": {"query": cleaned_query, "domain_filter": domain_filter, **kwargs},
            }
            _debug_zhipu("normalized", normalized)
            return json.dumps(normalized, ensure_ascii=False)
        except Exception as e:
            normalized = {"results": [], "summary": "检索异常。", "error": str(e)}
            _debug_zhipu("normalized", normalized)
            return json.dumps(normalized, ensure_ascii=False)

# 实例化工具
# ============ Opportunity Query Templates (without site: restrictions for broader search) ============
# These templates are used for opportunity discovery - finding positive catalysts
# Each category has 2 queries (10 total for deep mode)
OPPORTUNITY_QUERY_TEMPLATES = {
    "policy_driver": [
        "{name} 政策 补贴 支持 扶持 产业",
        "{name} 国家项目 专项资金 政府采购",
    ],
    "tech_breakthrough": [
        "{name} 研发 专利 技术突破 创新",
        "{name} 技术领先 行业首创 自主研发",
    ],
    "market_expansion": [
        "{name} 新产品 新市场 海外拓展 出海",
        "{name} 产能扩张 新工厂 产线投产",
    ],
    "competitive_moat": [
        "{name} 龙头 市占率 竞争优势 行业地位",
        "{name} 技术壁垒 护城河 核心竞争力",
    ],
    "contract_evidence": [
        "{name} 中标 订单 大客户 框架协议",
        "{name} 签约 合同 战略合作 供货",
    ],
}

# Source tier weights for confidence scoring
SOURCE_TIER_WEIGHTS = {
    "cninfo.com.cn": 1.0,
    "sse.com.cn": 1.0,
    "szse.cn": 1.0,
    "eastmoney.com": 0.85,
    "10jqka.com.cn": 0.85,
    "cls.cn": 0.80,
    "yicai.com": 0.80,
    "caixin.com": 0.80,
    "sina.com.cn": 0.75,
    "gelonghui.com": 0.75,
    "xueqiu.com": 0.70,
    "gov.cn": 0.90,
    "ndrc.gov.cn": 0.90,
    "miit.gov.cn": 0.90,
    "most.gov.cn": 0.90,
    "tianyancha.com": 0.70,
    "qichacha.com": 0.70,
}


def get_source_tier_weight(url: str) -> float:
    """Get confidence weight based on source domain."""
    if not url:
        return 0.5
    for domain, weight in SOURCE_TIER_WEIGHTS.items():
        if domain in url:
            return weight
    return 0.6  # Default for unknown sources


def generate_opportunity_queries(name: str, theme: str = "", categories: Optional[List[str]] = None) -> List[dict]:
    """
    Generate opportunity discovery queries for a stock.

    Args:
        name: Stock name
        theme: Optional theme for context
        categories: Optional list of categories to query (default: all)

    Returns:
        List of query dicts with 'query' and 'category' keys
    """
    if categories is None:
        categories = list(OPPORTUNITY_QUERY_TEMPLATES.keys())

    queries = []
    for category in categories:
        templates = OPPORTUNITY_QUERY_TEMPLATES.get(category, [])
        for template in templates:
            query = template.format(name=name, theme=theme) if theme else template.format(name=name, theme="")
            queries.append({"query": query.strip(), "category": category})

    return queries


def deepseek_plan_opportunity_queries(
    name: str,
    theme: str,
    current_findings: str,
    evidence_gaps: List[str],
) -> Optional[dict]:
    """
    Use DeepSeek to plan additional opportunity discovery queries based on evidence gaps.

    Args:
        name: Stock name
        theme: Theme context
        current_findings: Summary of current positive findings
        evidence_gaps: List of evidence gaps to fill

    Returns:
        Dict with 'queries' list and 'focus_areas' list
    """
    messages = [
        {
            "role": "system",
            "content": """你是A股机会挖掘专家，负责设计搜索策略来发现投资机会。
你需要根据当前已发现的正面信息和证据缺口，设计后续搜索查询。

**搜索策略原则：**
1. 不要使用 site: 限制，让搜索覆盖更广
2. 查询应该简洁有力，3-6个关键词
3. 重点挖掘：订单、客户、政策、技术、产能等实质性利好
4. 每个查询针对一个具体的信息点

**输出JSON格式：**
{"queries":["查询1","查询2",...],"focus_areas":["重点1","重点2"]}"""
        },
        {
            "role": "user",
            "content": (
                f"股票：{name}\n"
                f"题材：{theme}\n"
                f"当前发现：{current_findings}\n"
                f"证据缺口：{', '.join(evidence_gaps)}\n\n"
                "请设计3-5个搜索查询来填补证据缺口，发现更多投资机会。"
            ),
        },
    ]

    content = deepseek_chat(messages)
    if not content:
        return None

    parsed = _safe_json_from_text(content)
    if parsed and isinstance(parsed.get("queries"), list):
        return parsed
    return None


def deepseek_synthesize_opportunity_results(
    name: str,
    theme: str,
    current_findings: str,
    followup_results: List[dict],
) -> Optional[dict]:
    """
    Use DeepSeek to synthesize structured opportunity findings from follow-up search results.

    Args:
        name: Stock name
        theme: Theme context
        current_findings: Summary of current positive findings
        followup_results: Compact search result payloads with title/snippet/url/date

    Returns:
        Dict with structured positive_findings / growth_catalysts or None on failure.
    """
    messages = [
        {
            "role": "system",
            "content": """你是A股机会挖掘专家，负责从搜索结果中提炼可核验的投资机会证据。
只保留有明确外部来源支撑的结论，禁止臆测。

输出严格JSON：
{
  "positive_findings":[
    {
      "category":"policy_driver|tech_breakthrough|market_expansion|competitive_moat|contract_evidence",
      "description":"简短描述",
      "evidence":"基于搜索结果的证据摘要",
      "confidence":0.0,
      "source_url":"https://...",
      "date":"2026-01-01"
    }
  ],
  "growth_catalysts":[
    {
      "catalyst_type":"policy|tech_breakthrough|market_expansion|competitive_moat|contract_evidence",
      "description":"简短描述",
      "timeframe":"near_term|medium_term|long_term",
      "confidence":0.0
    }
  ],
  "queries_exhausted":false,
  "reason":"可选说明"
}

要求：
1. positive_findings 只保留有有效 source_url 的项目
2. description/evidence 必须紧扣搜索结果
3. 如果证据不足，可返回空列表，但仍返回合法JSON"""
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "name": name,
                    "theme": theme,
                    "current_findings": current_findings,
                    "followup_results": followup_results,
                },
                ensure_ascii=False,
            ),
        },
    ]

    content = deepseek_chat(messages)
    if not content:
        return None

    parsed = _safe_json_from_text(content)
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("positive_findings", []), list):
        parsed["positive_findings"] = []
    if not isinstance(parsed.get("growth_catalysts", []), list):
        parsed["growth_catalysts"] = []
    return parsed


def extract_positive_findings(
    search_results: List[dict],
    name: str,
    category: str,
) -> List[dict]:
    """
    Extract positive findings from search results.

    Args:
        search_results: List of search result dicts
        name: Stock name for relevance filtering
        category: Category of the search (policy_driver, tech_breakthrough, etc.)

    Returns:
        List of finding dicts with description, evidence, confidence, source_url, date
    """
    findings = []

    # Keywords that indicate positive findings by category
    positive_keywords = {
        "policy_driver": ["补贴", "扶持", "政策支持", "专项资金", "政府采购", "入选", "获批"],
        "tech_breakthrough": ["专利", "首创", "突破", "领先", "自主研发", "核心技术", "创新"],
        "market_expansion": ["出海", "海外", "新市场", "产能扩张", "投产", "新工厂", "拓展"],
        "competitive_moat": ["龙头", "市占率", "竞争优势", "行业第一", "领先地位", "壁垒"],
        "contract_evidence": ["中标", "签约", "订单", "大客户", "框架协议", "供货", "合同"],
    }

    keywords = positive_keywords.get(category, [])

    for result in search_results:
        if not isinstance(result, dict):
            continue

        title = result.get("title", "") or ""
        snippet = result.get("snippet", "") or ""
        url = result.get("url", "") or ""
        date = result.get("date", "") or ""

        # Check if relevant to the stock
        text = f"{title} {snippet}"
        if name not in text:
            continue

        # Check for positive keywords
        matched_keywords = [kw for kw in keywords if kw in text]
        if not matched_keywords:
            continue

        # Calculate confidence based on source tier and keyword matches
        source_weight = get_source_tier_weight(url)
        keyword_weight = min(1.0, len(matched_keywords) * 0.3)
        confidence = (source_weight + keyword_weight) / 2

        findings.append({
            "category": category,
            "description": title[:100],
            "evidence": snippet[:300],
            "confidence": round(confidence, 2),
            "source_url": url,
            "date": date,
        })

    return findings


try:
    zhipu_tool = ZhipuSearchTool()
    zhipu_search = Tool(
        name="zhipu_search",
        func=zhipu_tool.search,
        description="Search tool powered by Zhipu AI (GLM-4) for real-time A-share news.",
    )
    print("✅ 智谱搜索工具 (ZhipuAI) 加载成功")
except Exception as e:
    print(f"⚠️ 智谱工具加载失败: {e}")

    def zhipu_search(query: str) -> str:
        return "❌ Zhipu 搜索不可用，请检查 ZHIPUAI_API_KEY 环境变量。"
