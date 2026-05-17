#!/usr/bin/env python3
"""
Test Gemma theme matching against a local vLLM endpoint.

Sends the exact same prompt format as gemma_match_themes() in trend_agent.py.
Tests whether the model can correctly:
1. Match semiconductor stocks to AI/chip themes
2. Match chemical stocks to phosphorus/agriculture themes
3. Reject unrelated stocks (baijiu, food, textile)
"""

import json
import os
import sys
from openai import OpenAI

# Bypass proxy for local network
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",192.168.3.46"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.3.46"

VLLM_BASE_URL = "http://192.168.3.46:8000/v1"
VLLM_API_KEY = "dummy"  # vLLM doesn't need a real key
VLLM_MODEL = "gemma-4-31B-nvfp4"

# --- Themes (realistic, matching recent pipeline runs) ---
THEMES = [
    {
        "name": "AI算力基础设施与应用",
        "keywords": ["AI", "算力", "芯片", "半导体", "GPU", "服务器", "数据中心", "封装测试"],
        "summary": "AI算力需求爆发，带动芯片、封测、服务器等产业链景气上行",
    },
    {
        "name": "AI芯片与上游核心元器件/材料",
        "keywords": ["芯片设计", "集成电路", "半导体材料", "EDA", "存储器", "模拟芯片"],
        "summary": "国产替代加速，芯片设计和上游材料受益",
    },
    {
        "name": "周期资源（磷化工/农化）与部分顺周期板块",
        "keywords": ["磷化工", "农化", "化肥", "磷矿", "纯碱", "周期"],
        "summary": "磷化工涨价周期，磷矿资源受限推动景气上行",
    },
]

# --- Stocks (mix of should-match and should-not-match) ---
STOCKS = [
    {
        "ts_code": "600584.SH",
        "text": "name=长电科技 | code=600584.SH | industry=半导体 | main_business=主营业务:集成电路封装测试,分立器件制造销售. | business_scope=研制,开发,生产,销售半导体,电子原件,专用电子电气装置 | intro=公司是全球领先的集成电路制造与技术服务提供商，向全球半导体客户提供全方位、一站式芯片成品制造解决方案，涵盖微系统集成、设计仿真、晶圆中测、芯片及器件封装、成品测试，广泛应用于汽车电子、人工智能、高性能计算、高密度存储、网络通信等领域。",
    },
    {
        "ts_code": "002049.SZ",
        "text": "name=紫光国微 | code=002049.SZ | industry=半导体 | main_business=主营业务:集成电路芯片设计与销售,包括智能芯片产品,特种集成电路产品和存储器芯片产品 | business_scope=集成电路芯片设计及服务;集成电路设计;集成电路芯片及产品销售 | intro=公司是新紫光集团旗下核心上市公司，在集成电路设计领域深耕二十余年，已成为国内集成电路设计企业龙头之一。在特种集成电路和智能安全芯片领域，具有广泛的品牌影响力和知名度。",
    },
    {
        "ts_code": "600877.SH",
        "text": "name=电科芯片 | code=600877.SH | industry=半导体 | main_business=摩托车 | business_scope=电子元器件制造,集成电路设计,集成电路制造,集成电路销售,电力电子元器件销售,5G通信技术服务 | intro=公司业务为硅基模拟半导体芯片及其应用产品的设计、研发、制造、测试、销售，主要产品包括硅基模拟半导体相关芯片、器件、模组、整体解决方案，可广泛应用于物联网、消费电子、绿色能源、安全电子、汽车电子及智能电源等领域。",
    },
    {
        "ts_code": "000422.SZ",
        "text": "name=湖北宜化 | code=000422.SZ | industry=农药化肥 | main_business=主要产品:尿素,磷酸二铵,聚氯乙烯,烧碱,合成氨,季戊四醇 | business_scope=肥料生产;危险化学品生产;发电业务 | intro=公司是我国第一家氮肥类上市公司，为国内化肥龙头企业。在\"磷矿之都\"宜昌周边设立磷化工生产基地，在新疆、内蒙古、青海等西部地区设立煤化工、氯碱化工生产基地。公司产品主要包括尿素、磷酸二铵、聚氯乙烯、烧碱、合成氨等。",
    },
    {
        "ts_code": "000525.SZ",
        "text": "name=红太阳 | code=000525.SZ | industry=农药化肥 | main_business=主要产品:农药,化肥. | business_scope=农药生产,三药中间体及精细化工产品的生产,销售;化肥经营 | intro=公司是一家国资控股的专业化、全球化农化企业，专注于生化环保农药、生化动物营养和中间体的研发、生产、销售和技术服务。",
    },
    {
        "ts_code": "000683.SZ",
        "text": "name=博源化工 | code=000683.SZ | industry=化工原料 | main_business=主要业务:天然碱和无机盐系列产品的生产和经营.主要产品:纯碱,小苏打. | business_scope=化工产品生产;化工产品销售;专用化学产品制造 | intro=公司是国内最大的以天然碱法制纯碱和小苏打的生产企业。致力于打造循环经济特征鲜明的天然碱化工、煤化工产业基地。",
    },
    {
        "ts_code": "000568.SZ",
        "text": "name=泸州老窖 | code=000568.SZ | industry=白酒 | main_business=主要产品:\"泸州老窖\"系列酒. | business_scope=酒类产品的生产,销售(白酒,葡萄酒及果酒,啤酒,其他酒等) | intro=公司是在明清36家酿酒作坊群的基础上发展起来的国有大型酿酒企业，是中国白酒浓香技艺的开创者、浓香标准的制定者和浓香品牌的塑造者，被誉为浓香鼻祖。",
    },
    {
        "ts_code": "000858.SZ",
        "text": "name=五粮液 | code=000858.SZ | industry=白酒 | main_business=主要产品:五粮液系列酒. | business_scope=酒类产品及相关辅助产品的生产经营 | intro=公司是以酒业为核心主业，以大机械、大金融、大物流、大包装、大健康多元发展的特大型国有企业集团。",
    },
    {
        "ts_code": "000895.SZ",
        "text": "name=双汇发展 | code=000895.SZ | industry=食品 | main_business=主要产品:高温肉制品,低温肉制品,鲜冻猪产品,商业连锁服务. | business_scope=牲畜屠宰;生猪屠宰;食品生产;食品销售 | intro=作为农业产业化国家重点龙头企业，本公司在全国建立了30多家现代化肉类加工基地，连续多年领跑中国肉类行业。",
    },
    {
        "ts_code": "000726.SZ",
        "text": "name=鲁泰Ａ | code=000726.SZ | industry=纺织 | main_business=主要产品:衬衣色织面,衬衣,皮棉,纺纱. | business_scope=面料纺织加工;面料印染加工;服装制造 | intro=公司是目前全球高档色织面料生产商和国际一线品牌衬衫制造商，拥有从纺织、染整、制衣生产到品牌营销的完整产业链。",
    },
]

# Expected results for validation
EXPECTED = {
    # Should match AI/chip themes
    "600584.SH": ["AI算力基础设施与应用", "AI芯片与上游核心元器件/材料"],
    "002049.SZ": ["AI芯片与上游核心元器件/材料"],
    "600877.SH": ["AI芯片与上游核心元器件/材料"],
    # Should match phosphorus/chemical theme
    "000422.SZ": ["周期资源（磷化工/农化）与部分顺周期板块"],
    "000525.SZ": ["周期资源（磷化工/农化）与部分顺周期板块"],
    "000683.SZ": ["周期资源（磷化工/农化）与部分顺周期板块"],
    # Should NOT match any theme
    "000568.SZ": [],
    "000858.SZ": [],
    "000895.SZ": [],
    "000726.SZ": [],
}

SYSTEM_PROMPT = (
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
    '- 输出严格JSON：{"matches":{"ts_code":["theme1","theme2"]},"notes":{"ts_code":"reason"}}\n'
    "- notes中说明关联逻辑（直接/间接/概念延伸）"
)


def run_test(batch_size: int = 4):
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Test in batches like the real code does
    all_matches = {}
    for i in range(0, len(STOCKS), batch_size):
        batch = STOCKS[i : i + batch_size]
        print(f"\n--- Batch {i // batch_size + 1} ({len(batch)} stocks) ---")

        user_content = json.dumps(
            {"themes": THEMES, "stocks": batch}, ensure_ascii=False
        )

        try:
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=1.0,
                top_p=0.95,
                extra_body={"top_k": 64},
            )
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        raw = response.choices[0].message.content
        print(f"Raw response:\n{raw}\n")

        # Strip any stray thinking tags if present
        import re
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

        # Parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    print(f"FAILED to parse JSON from response")
                    parsed = {}
            else:
                print(f"FAILED to find JSON in response")
                parsed = {}

        matches = parsed.get("matches", {})
        notes = parsed.get("notes", {})
        for item in batch:
            ts_code = item["ts_code"]
            matched = matches.get(ts_code, [])
            if not isinstance(matched, list):
                matched = []
            all_matches[ts_code] = matched
            note = notes.get(ts_code, "")
            print(f"  {ts_code}: {matched}  ({note})")

    # --- Evaluate ---
    print("\n" + "=" * 70)
    print("EVALUATION (exact string match — same as pipeline)")
    print("=" * 70)

    theme_names = {t["name"] for t in THEMES}

    correct_exact = 0
    correct_fuzzy = 0
    total = len(EXPECTED)

    def normalize_spaces(s: str) -> str:
        """Collapse whitespace for fuzzy comparison."""
        return re.sub(r"\s+", "", s)

    for ts_code, expected in EXPECTED.items():
        actual = all_matches.get(ts_code, [])
        expected_set = set(expected)
        actual_set = set(actual)
        actual_fuzzy = {
            next((tn for tn in theme_names if normalize_spaces(tn) == normalize_spaces(a)), a)
            for a in actual
        }

        stock_name = next(
            (s["text"].split(" | ")[0].replace("name=", "") for s in STOCKS if s["ts_code"] == ts_code),
            ts_code,
        )

        if expected_set:
            exact_hit = bool(expected_set & actual_set)
            fuzzy_hit = bool(expected_set & actual_fuzzy)
        else:
            exact_hit = not actual_set
            fuzzy_hit = not actual_fuzzy

        if exact_hit:
            correct_exact += 1
        if fuzzy_hit:
            correct_fuzzy += 1

        if exact_hit:
            status = "PASS"
        elif fuzzy_hit:
            status = "FUZZY"  # correct match but whitespace mismatch
        else:
            status = "FAIL"

        print(f"  [{status}] {stock_name} ({ts_code})")
        print(f"         expected: {expected}")
        print(f"         actual:   {actual}")
        if not exact_hit and fuzzy_hit:
            print(f"         NOTE: whitespace mismatch — model inserted spaces in theme name")

    print(f"\nExact match score:  {correct_exact}/{total} ({100 * correct_exact / total:.0f}%)")
    print(f"Fuzzy match score:  {correct_fuzzy}/{total} ({100 * correct_fuzzy / total:.0f}%)")
    if correct_exact == total:
        print("ALL TESTS PASSED (exact)")
    elif correct_fuzzy == total:
        print("ALL TESTS PASSED (fuzzy) — but pipeline needs whitespace normalization!")
        print("\nACTION NEEDED: The model inserts spaces in theme names (e.g. 'AI 算力' vs 'AI算力').")
        print("The pipeline's exact-match filter will reject these. Add whitespace normalization")
        print("when comparing model output to theme name whitelist.")
    else:
        print(f"FAILED: {total - correct_fuzzy} true mismatches")


if __name__ == "__main__":
    run_test()
