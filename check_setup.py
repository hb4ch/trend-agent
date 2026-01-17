import os
from zai import ZhipuAiClient
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatZhipuAI # 也可以用社区版集成，但官方SDK写Tool更稳
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

api_key = os.environ.get("ZHIPUAI_API_KEY")

print(f"✅ 环境变量加载成功 (Key 长度: {len(api_key) if api_key else 0})")

# 2. 测试智谱原生 SDK (用于搜索)
try:
    if not api_key:
        raise ValueError("未设置 ZHIPUAI_API_KEY")
    client = ZhipuAiClient(api_key=api_key)
    response = client.web_search.web_search(
        search_engine="search_pro",
        search_query="今天A股上证指数收盘多少？",
        count=5,
        search_recency_filter="noLimit",
        content_size="high",
    )
    print("\n📡 [智谱原生 web_search 测试] 成功:")
    print(str(response)[:200] + "...")
except Exception as e:
    print(f"\n❌ [智谱原生 web_search 测试] 失败: {e}")

# 3. 测试 LangChain 导入
try:
    # Prefer langchain_core's pydantic compatibility if available, otherwise fall back to pydantic
    from pydantic import BaseModel, Field  # type: ignore
    print("\n🔗 [LangChain] 导入成功")
except Exception as e:
    print(f"\n❌ [LangChain] 导入失败: {e}")

print("\n🚀 环境准备就绪！")
