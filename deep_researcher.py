import json
import os
import urllib.request
import re
from typing import List, Optional

from zai import ZhipuAiClient
from langchain_core.tools import Tool

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", DEEPSEEK_BASE_URL)
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", DEEPSEEK_API_KEY)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")

DEBUG_DEEPSEEK = os.environ.get("DEBUG_DEEPSEEK", "").strip() in {"1", "true", "True", "YES", "yes"}
DEBUG_ZHIPU_SEARCH = os.environ.get("DEBUG_ZHIPU_SEARCH", "").strip() in {"1", "true", "True", "YES", "yes"}
DEBUG_SILICONFLOW = os.environ.get("DEBUG_SILICONFLOW", "").strip() in {"1", "true", "True", "YES", "yes"}


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
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return obj.__dict__  # type: ignore[attr-defined]
        except Exception:
            pass
    return obj


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
    return siliconflow_chat(messages=messages, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, debug=DEBUG_DEEPSEEK, debug_prefix="DeepSeek")


def siliconflow_chat(
    messages: List[dict],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    timeout_s: int = 60,
    debug: bool = False,
    debug_prefix: str = "SiliconFlow",
) -> Optional[str]:
    if not api_key:
        return None
    url_base = (base_url or SILICONFLOW_BASE_URL or "").rstrip("/")
    if not url_base:
        return None
    url = f"{url_base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if debug or DEBUG_SILICONFLOW:
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
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if debug or DEBUG_SILICONFLOW:
            try:
                print(f"[{debug_prefix}] response:\n" + _truncate(content, 4000))
            except Exception:
                pass
        return content
    except Exception:
        if debug or DEBUG_SILICONFLOW:
            print(f"[{debug_prefix}] error: request_failed")
        return None


def qwen_chat(messages: List[dict]) -> Optional[str]:
    return siliconflow_chat(
        messages=messages,
        model=QWEN_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.1,
        debug=False,
        debug_prefix="Qwen",
    )


def deepseek_plan_queries(name: str, theme: str, evidence: str, pass_id: int) -> Optional[dict]:
    messages = [
        {
            "role": "system",
            "content": "你是A股研究协调员，决定是否需要更深的检索并给出查询。",
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
    try:
        parsed = json.loads(content.strip().strip("```"))
        _debug_print("plan_queries_output", parsed)
        return parsed
    except json.JSONDecodeError:
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

    def search(self, query: str) -> str:
        """
        调用智谱 ZAI 原生 web_search 接口（更 raw、更稳定地拿到链接）
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

            kwargs = {
                "search_engine": "search_pro",
                "search_query": cleaned_query,
                "count": 15,
                "search_recency_filter": "noLimit",
                "content_size": "high",
            }
            if domain_filter:
                kwargs["search_domain_filter"] = domain_filter

            response = self.client.web_search.web_search(**kwargs)
            resp_obj = _to_dict(response)
            _debug_zhipu("raw_response", resp_obj)

            results = _extract_web_results(resp_obj)
            summary = ""
            if not results:
                summary = "未找到相关链接。"

            normalized = {"results": results, "summary": summary}
            _debug_zhipu("normalized", normalized)
            return json.dumps(normalized, ensure_ascii=False)
        except Exception as e:
            return f"❌ Search Error: {e}"

# 实例化工具
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
