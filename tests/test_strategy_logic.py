import json
from pathlib import Path

import pandas as pd
from langchain_core.runnables import RunnableLambda

import deep_researcher
import screen_growth_stocks
import trend_agent
from trend_agent import (
    AuditResult,
    GrowthCatalyst,
    PositiveFinding,
    StrategyConfig,
    ThemeItem,
)


def test_theme_matches_reflect_current_theme_set(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_invoke(provider, messages, temperature=0.1):
        call_count["n"] += 1
        payload = json.loads(messages[-1]["content"])
        theme_name = payload["themes"][0]["name"]
        ts_code = payload["stocks"][0]["ts_code"]
        return json.dumps({"matches": {ts_code: [theme_name]}, "notes": {ts_code: "ok"}}, ensure_ascii=False)

    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)

    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
                "main_business": "主题A业务、主题B业务与金融服务",
                "business_scope": "银行",
                "introduction": "介绍",
            }
        ]
    )
    out_a = trend_agent.gemma_match_themes(
        themes=[ThemeItem(name="主题A", keywords=["a"], summary="", sources=[])],
        candidates=candidates,
    )
    out_b = trend_agent.gemma_match_themes(
        themes=[ThemeItem(name="主题B", keywords=["b"], summary="", sources=[])],
        candidates=candidates,
    )

    assert out_a.iloc[0]["matched_themes"] == ["主题A"]
    assert out_b.iloc[0]["matched_themes"] == ["主题B"]
    assert call_count["n"] == 2


def _price_df(last_close: float) -> pd.DataFrame:
    n = 130
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = [9.5] * (n - 1) + [last_close]
    high = [10.0] * (n - 1) + [max(10.0, last_close)]
    low = [8.8] * n
    open_ = [9.4] * n
    vol = [1000] * n
    turn = [1.0] * (n - 10) + [1.5] * 10
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "turnover_rate": turn,
        }
    ).set_index("trade_date")


def test_compute_signals_breakout_boundaries(monkeypatch):
    price_map = {
        "A": _price_df(9.8),
        "B": _price_df(10.0),
        "C": _price_df(10.1),
        "D": _price_df(10.5),
    }
    monkeypatch.setattr(trend_agent, "load_price_data", lambda ts: price_map[ts])

    candidates = pd.DataFrame(
        [{"ts_code": code, "name": code} for code in ["A", "B", "C", "D"]]
    )
    signals = trend_agent.compute_signals(candidates)

    assert signals["A"]["ready_to_break"] is True
    assert signals["B"]["ready_to_break"] is True
    assert signals["C"]["already_breakout"] is True
    assert signals["C"]["ready_to_break"] is False
    assert signals["D"]["extended_breakout"] is True
    assert signals["D"]["ready_to_break"] is False


def test_invoke_with_timeout_retries_retries_on_timeout(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def flaky_call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("simulated timeout")
        return "ok"

    monkeypatch.setattr(trend_agent.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(trend_agent.random, "uniform", lambda a, b: 0.0)

    out = trend_agent.invoke_with_timeout_retries(
        flaky_call,
        description="test call",
        max_retries=3,
        base_delay_sec=5.0,
        max_delay_sec=90.0,
    )

    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [5.0, 10.0]


def test_read_tushare_token_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TUSHARE_API_TOKEN", "env-token")
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("TUSHARE_TOKEN_FILE", str(token_file))

    assert trend_agent._read_tushare_token() == "env-token"


def test_read_tushare_token_uses_local_project_file(monkeypatch, tmp_path):
    monkeypatch.delenv("TUSHARE_API_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN_FILE", raising=False)
    monkeypatch.setattr(trend_agent, "PROJECT_ROOT", Path(tmp_path))
    (tmp_path / "tokens.txt").write_text("local-token", encoding="utf-8")

    assert trend_agent._read_tushare_token() == "local-token"


def test_invoke_with_timeout_retries_raises_after_exhaustion(monkeypatch):
    sleeps = []

    monkeypatch.setattr(trend_agent.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(trend_agent.random, "uniform", lambda a, b: 0.0)

    def always_times_out():
        raise TimeoutError("simulated timeout")

    try:
        trend_agent.invoke_with_timeout_retries(
            always_times_out,
            description="test call",
            max_retries=2,
            base_delay_sec=5.0,
            max_delay_sec=90.0,
        )
        assert False, "expected timeout to be raised"
    except TimeoutError:
        pass

    assert sleeps == [5.0, 10.0]


def test_hard_fail_requires_relevance_and_material_reduce():
    unrelated = {
        "title": "其他公司公告减持5%",
        "snippet": "与目标股票无关",
        "url": "https://example.com",
        "date": "2026-01-10",
    }
    assert trend_agent.detect_hard_fail_reason(
        unrelated,
        name="目标公司",
        symbol="000001",
        require_recency=True,
        max_age_days=365,
        reduce_threshold=0.03,
    ) is None

    small_reduce = {
        "title": "目标公司拟减持不超过1%",
        "snippet": "000001.SZ 股东减持计划",
        "url": "https://example.com/1",
        "date": "2026-01-10",
    }
    assert trend_agent.detect_hard_fail_reason(
        small_reduce,
        name="目标公司",
        symbol="000001",
        require_recency=True,
        max_age_days=365,
        reduce_threshold=0.03,
    ) is None

    large_reduce = {
        "title": "目标公司拟减持不超过5%",
        "snippet": "000001.SZ 股东大比例减持计划",
        "url": "https://example.com/2",
        "date": "2026-01-10",
    }
    assert trend_agent.detect_hard_fail_reason(
        large_reduce,
        name="目标公司",
        symbol="000001",
        require_recency=True,
        max_age_days=365,
        reduce_threshold=0.03,
    ) == "material_reduction"


def test_phase2_off_theme_fallback_mixed_labels(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    monkeypatch.setattr(
        trend_agent,
        "screen_all_stocks",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "A",
                    "industry": "I1",
                    "consolidation_score": 65,
                    "ma_spread": 0.18,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.3,
                    "composite_score": 70.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "name": "B",
                    "industry": "I2",
                    "consolidation_score": 62,
                    "ma_spread": 0.19,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.2,
                    "composite_score": 68.0,
                },
            ]
        ),
    )

    def fake_match(themes, candidates, config=None, relaxed_validation=False, named_stock_codes=None):
        out = candidates.copy()
        out["matched_themes"] = [[] for _ in range(len(out))]
        return out

    heuristic_calls = {"n": 0}

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        heuristic_calls["n"] += 1
        out = candidates.copy()
        if heuristic_calls["n"] < 4:
            out["matched_themes"] = [[] for _ in range(len(out))]
        else:
            out["matched_themes"] = [["AI"], []]
        return out

    monkeypatch.setattr(trend_agent, "gemma_match_themes", fake_match)
    monkeypatch.setattr(trend_agent, "heuristic_match_themes", fake_heuristic)
    cfg = StrategyConfig(toplist_exclusion_mode="penalty", theme_match_policy="aggressive")
    themes = [ThemeItem(name="AI", keywords=["算力"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)

    row_a = out[out["ts_code"] == "000001.SZ"].iloc[0]
    row_b = out[out["ts_code"] == "000002.SZ"].iloc[0]
    assert bool(row_a["off_theme"]) is False
    assert bool(row_b["off_theme"]) is True
    assert set(out["filter_tier"].tolist()) == {"OFF_THEME_FALLBACK"}


def test_phase2_conservative_blocks_heuristic_false_positive(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    monkeypatch.setattr(
        trend_agent,
        "screen_all_stocks",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "600059.SH",
                    "name": "古越龙山",
                    "industry": "红黄酒",
                    "main_business": "主要产品:绍兴花雕酒、加饭酒",
                    "business_scope": "黄酒、白酒、玻璃制品的技术开发与销售",
                    "introduction": "黄酒行业龙头企业。",
                    "consolidation_score": 77,
                    "ma_spread": 0.04,
                    "ma_spread_std": 0.01,
                    "volume_boost": 1.9,
                    "composite_score": 52.0,
                }
            ]
        ),
    )

    def fake_match(themes, candidates, config=None, relaxed_validation=False, named_stock_codes=None):
        out = candidates.copy()
        out["matched_themes"] = [[] for _ in range(len(out))]
        return out

    heuristic_calls = {"n": 0}

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        heuristic_calls["n"] += 1
        out = candidates.copy()
        out["matched_themes"] = [["化工与周期材料"]]
        return out

    monkeypatch.setattr(trend_agent, "gemma_match_themes", fake_match)
    monkeypatch.setattr(trend_agent, "heuristic_match_themes", fake_heuristic)
    cfg = StrategyConfig(toplist_exclusion_mode="penalty", theme_match_policy="conservative")
    themes = [ThemeItem(name="化工与周期材料", keywords=["化工", "玻璃"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)

    row = out[out["ts_code"] == "600059.SH"].iloc[0]
    assert heuristic_calls["n"] == 0
    assert row["matched_themes"] == []
    assert bool(row["off_theme"]) is True
    assert row["filter_tier"] == "OFF_THEME_FALLBACK"


def test_phase2_conservative_no_gemma_match_returns_off_theme(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    monkeypatch.setattr(
        trend_agent,
        "screen_all_stocks",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "A",
                    "industry": "I1",
                    "consolidation_score": 65,
                    "ma_spread": 0.18,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.3,
                    "composite_score": 70.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "name": "B",
                    "industry": "I2",
                    "consolidation_score": 62,
                    "ma_spread": 0.19,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.2,
                    "composite_score": 68.0,
                },
            ]
        ),
    )

    def fake_match(themes, candidates, config=None, relaxed_validation=False, named_stock_codes=None):
        out = candidates.copy()
        out["matched_themes"] = [[] for _ in range(len(out))]
        return out

    heuristic_calls = {"n": 0}

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        heuristic_calls["n"] += 1
        out = candidates.copy()
        out["matched_themes"] = [["AI"], ["AI"]]
        return out

    monkeypatch.setattr(trend_agent, "gemma_match_themes", fake_match)
    monkeypatch.setattr(trend_agent, "heuristic_match_themes", fake_heuristic)
    cfg = StrategyConfig(toplist_exclusion_mode="penalty", theme_match_policy="conservative")
    themes = [ThemeItem(name="AI", keywords=["算力"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)

    assert heuristic_calls["n"] == 0
    assert set(out["filter_tier"].tolist()) == {"OFF_THEME_FALLBACK"}
    assert out["off_theme"].all()
    assert out["matched_themes"].map(len).sum() == 0


def test_end_to_end_fixture_ranking_and_audit_distribution():
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "industry": "I1",
                "matched_themes": ["AI"],
                "ma_spread": 0.10,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.1,
            },
            {
                "ts_code": "000002.SZ",
                "name": "B",
                "industry": "I2",
                "matched_themes": ["AI"],
                "ma_spread": 0.20,
                "theme_strength_score": 0.7,
                "toplist_recency_score": 0.5,
            },
            {
                "ts_code": "000003.SZ",
                "name": "C",
                "industry": "I3",
                "matched_themes": ["机器人"],
                "ma_spread": 0.22,
                "theme_strength_score": 0.8,
                "toplist_recency_score": 0.3,
            },
        ]
    )
    audits = [
        AuditResult(
            ts_code="000001.SZ",
            name="A",
            theme="AI",
            verdict="pass",
            rationale="ok",
            sources=["https://cninfo.com.cn/a"],
            positive_findings=[PositiveFinding("contract", "中标订单", "e", 0.9, "https://cninfo.com.cn/a")],
            growth_catalysts=[GrowthCatalyst("market_expansion", "扩产", "near_term", 0.8)],
        ),
        AuditResult(
            ts_code="000002.SZ",
            name="B",
            theme="AI",
            verdict="warn",
            rationale="warn",
            sources=["https://eastmoney.com/b"],
        ),
        AuditResult(
            ts_code="000003.SZ",
            name="C",
            theme="机器人",
            verdict="fail",
            rationale="fail",
            sources=["https://cninfo.com.cn/c"],
        ),
    ]
    filtered, filtered_audits = trend_agent.apply_audit_filter(candidates, audits)
    assert set(filtered["ts_code"].tolist()) == {"000001.SZ", "000002.SZ"}
    assert set(a.verdict for a in filtered_audits) == {"pass", "warn"}

    signals = {
        "000001.SZ": {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.6},
        "000002.SZ": {"breakout_window_ok": False, "already_breakout": True, "extended_breakout": False, "turnover_mult": 3.8},
    }
    ranked = trend_agent.rank_candidates_for_alpha(filtered, filtered_audits, signals, config=StrategyConfig())
    ordered = ranked["ts_code"].tolist()
    assert ordered[0] == "000001.SZ"
    assert "audit_risk_score" in ranked.columns
    assert "alpha_rank_score" in ranked.columns


def test_heuristic_match_themes_recovers_non_empty_matches():
    themes = [
        ThemeItem(name="商业航天与卫星互联网", keywords=["卫星", "航天"], summary="", sources=[]),
        ThemeItem(name="电网投资", keywords=["电网", "变压器"], summary="", sources=[]),
    ]
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "航天科技",
                "industry": "军工",
                "main_business": "卫星通信设备研发",
                "business_scope": "航天电子产品",
                "introduction": "",
                "matched_themes": [],
            }
        ]
    )
    out = trend_agent.heuristic_match_themes(themes, candidates)
    assert out.iloc[0]["matched_themes"]


def test_upsert_core_table_replaces_sparse_llm_core_section():
    report = (
        "# 报告\n\n"
        "## 【市场风向标】\n内容\n\n"
        "## 【核心金股】\n\n"
        "| 股票 | 所属主线 |\n| --- | --- |\n| A | OFF |\n\n"
        "## 【深度图解】\n后续"
    )
    candidates = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "A", "matched_themes": ["AI"], "off_theme": False, "consolidation_score": 70, "volume_boost": 1.5, "filter_tier": "Strict"},
            {"ts_code": "000002.SZ", "name": "B", "matched_themes": [], "off_theme": True, "consolidation_score": 68, "volume_boost": 1.3, "filter_tier": "Fallback"},
        ]
    )
    table = trend_agent.build_deterministic_core_table(candidates, audits=[], top_n=2)
    merged = trend_agent.upsert_core_table_in_report(report, table)
    assert merged.count("## 【核心金股 - 技术形态精选】") == 1
    assert "| A(000001.SZ) |" in merged
    assert "| B(000002.SZ) |" in merged


def test_normalize_stock_section_payload_strips_placeholder_urls():
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}
    payload = {
        "summary": "公司逻辑 url1",
        "investment_logic": ["- 箱体突破 url2"],
        "source_urls": ["https://example.com/report"],
    }
    section = trend_agent._normalize_stock_section_payload(payload, row, [], None, {})
    assert "url1" not in section.summary.lower()
    assert all("url" not in item.lower() for item in section.investment_logic)
    assert section.source_urls == ["https://example.com/report"]


def test_generate_market_overview_falls_back_from_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "deepseek_chat", lambda messages: "## 【市场风向标】\n- markdown response")
    themes = [ThemeItem(name="AI应用", keywords=["AI"], summary="主题逻辑", sources=["https://example.com/theme"], validation_status="confirmed")]
    items = trend_agent._generate_market_overview(
        [{"name": "AI应用", "summary": "主题逻辑"}],
        tmp_path / "trace.jsonl",
        themes,
    )
    assert len(items) == 1
    assert items[0].name == "AI应用"
    assert items[0].validation_status == "confirmed"
    assert items[0].logic


def test_market_overview_weak_sources_triggers_correction(monkeypatch, tmp_path):
    calls = []

    def fake_chat(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "AI应用",
                            "validation_status": "confirmed",
                            "logic": ["AI应用景气度提升"],
                            "capital_validation": ["资金活跃"],
                            "watch_items": ["跟踪订单"],
                            "source_urls": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "themes": [
                    {
                        "name": "AI应用",
                        "validation_status": "confirmed",
                        "logic": ["AI应用景气度提升"],
                        "capital_validation": ["资金活跃"],
                        "watch_items": ["跟踪订单"],
                        "source_urls": ["https://example.com/theme"],
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_chat)
    trace_path = tmp_path / "trace.jsonl"
    themes = [ThemeItem(name="AI应用", keywords=["AI"], summary="主题逻辑", sources=[], validation_status="confirmed")]
    items = trend_agent._generate_market_overview(
        [{"name": "AI应用", "summary": "主题逻辑", "sources": ["https://example.com/theme"]}],
        trace_path,
        themes,
    )

    assert len(calls) == 2
    assert items[0].source_urls == ["https://example.com/theme"]
    trace_text = trace_path.read_text(encoding="utf-8")
    assert '"event": "overview_source_check"' in trace_text
    assert '"event": "overview_correction_request"' in trace_text
    assert '"event": "overview_correction_response"' in trace_text


def test_market_overview_strong_sources_skips_correction(monkeypatch, tmp_path):
    calls = []

    def fake_chat(messages):
        calls.append(messages)
        return json.dumps(
            {
                "themes": [
                    {
                        "name": "机器人",
                        "validation_status": "web_only",
                        "logic": ["政策催化"],
                        "capital_validation": ["资金待验证"],
                        "watch_items": ["跟踪成交额"],
                        "source_urls": ["https://example.com/robot"],
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_chat)
    themes = [ThemeItem(name="机器人", keywords=["机器人"], summary="主题逻辑", sources=[], validation_status="web_only")]
    items = trend_agent._generate_market_overview(
        [{"name": "机器人", "summary": "主题逻辑"}],
        tmp_path / "trace.jsonl",
        themes,
    )
    assert len(calls) == 1
    assert items[0].source_urls == ["https://example.com/robot"]


def test_market_overview_failed_correction_falls_back(monkeypatch, tmp_path):
    responses = iter(
        [
            json.dumps(
                {
                    "themes": [
                        {
                            "name": "低空经济",
                            "validation_status": "confirmed",
                            "logic": ["政策催化"],
                            "capital_validation": ["资金活跃"],
                            "watch_items": ["跟踪政策"],
                            "source_urls": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "not json",
        ]
    )
    monkeypatch.setattr(trend_agent, "deepseek_chat", lambda messages: next(responses))
    themes = [
        ThemeItem(
            name="低空经济",
            keywords=["低空"],
            summary="低空经济政策持续。",
            sources=["https://example.com/low-altitude"],
            validation_status="confirmed",
        )
    ]
    items = trend_agent._generate_market_overview(
        [{"name": "低空经济", "summary": "低空经济政策持续。"}],
        tmp_path / "trace.jsonl",
        themes,
    )
    assert items[0].source_urls == ["https://example.com/low-altitude"]
    assert items[0].logic == ["低空经济政策持续。"]


def test_phase1_weak_extraction_triggers_corrective_pass(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def get_llm(self, *args, **kwargs):
            def respond(_payload):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {"themes": [{"name": "股票", "keywords": [], "summary": "", "sources": []}]},
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "themes": [
                            {
                                "name": "AI应用",
                                "keywords": ["AI"],
                                "summary": "AI应用订单与政策共振。",
                                "sources": ["https://example.com/ai"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

            return RunnableLambda(respond)

    class FakeDragonTigerList:
        def identify_hot_themes(self, days=30):
            return {}

    provider = FakeProvider()
    monkeypatch.setattr(trend_agent, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDragonTigerList)
    monkeypatch.setattr(trend_agent, "deepseek_merge_themes", lambda web_themes, capital_themes, llm_provider: web_themes)
    monkeypatch.setattr(
        trend_agent,
        "run_search",
        lambda query: json.dumps(
            {
                "summary": "AI应用活跃",
                "urls": ["https://example.com/ai"],
                "results": [{"title": "AI应用", "snippet": "AI应用订单与政策共振", "url": "https://example.com/ai"}],
            },
            ensure_ascii=False,
        ),
    )

    themes = trend_agent.phase1_market_intel(RunnableLambda(lambda _: ""))
    assert provider.calls == 2
    assert themes[0].name == "AI应用"
    assert themes[0].sources == ["https://example.com/ai"]


def test_deepseek_merge_themes_weak_fusion_triggers_corrective_pass():
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def get_llm(self, *args, **kwargs):
            def respond(_payload):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {
                            "themes": [
                                {
                                    "name": "AI应用",
                                    "validation_status": "confirmed",
                                    "keywords": ["AI"],
                                    "summary": "",
                                    "capital_signal": "",
                                    "sources": [],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "themes": [
                            {
                                "name": "AI应用",
                                "validation_status": "confirmed",
                                "keywords": ["AI"],
                                "summary": "Web热度与资金共同确认。",
                                "capital_signal": "龙虎榜净买入活跃。",
                                "sources": ["https://example.com/ai"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

            return RunnableLambda(respond)

    web_themes = [
        ThemeItem(
            name="AI应用",
            keywords=["AI"],
            summary="AI应用活跃",
            sources=["https://example.com/ai"],
        )
    ]
    capital_themes = {
        "AI应用": {
            "hit_count": 3,
            "net_buy": 120000000,
            "hot_stocks": ["测试股"],
            "institution_mix": {"north": 0.1, "inst": 0.4, "hot_money": 0.5},
            "trend": "增强",
        }
    }
    provider = FakeProvider()
    merged = trend_agent.deepseek_merge_themes(web_themes, capital_themes, provider)
    assert provider.calls == 2
    assert merged[0].summary == "Web热度与资金共同确认。"
    assert merged[0].capital_signal == "龙虎榜净买入活跃。"


def test_deepseek_merge_themes_valid_fusion_stays_single_pass():
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def get_llm(self, *args, **kwargs):
            def respond(_payload):
                self.calls += 1
                return json.dumps(
                    {
                        "themes": [
                            {
                                "name": "机器人",
                                "validation_status": "web_only",
                                "keywords": ["机器人"],
                                "summary": "Web热度确认。",
                                "capital_signal": "",
                                "sources": ["https://example.com/robot"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

            return RunnableLambda(respond)

    provider = FakeProvider()
    merged = trend_agent.deepseek_merge_themes(
        [ThemeItem(name="机器人", keywords=["机器人"], summary="机器人活跃", sources=["https://example.com/robot"])],
        {},
        provider,
    )
    assert provider.calls == 1
    assert merged[0].name == "机器人"


def test_render_report_html_is_self_contained():
    report = trend_agent.ReportModel(
        title="测试研报",
        generated_at="2026-03-22 12:00:00",
        theme_overviews=[
            trend_agent.ReportThemeOverview(
                name="AI应用",
                validation_status="confirmed",
                logic=["逻辑A"],
                capital_validation=["资金A"],
                watch_items=["观察A"],
                source_urls=["https://example.com/theme"],
            )
        ],
        core_table_rows=[
            {"股票": "测试股(000001.SZ)", "所属主线": "AI应用", "形态特征": "横盘分70", "置信度": "0.80", "推荐理由": "alpha评分高"}
        ],
        theme_table_rows=[],
        stock_sections=[
            trend_agent.ReportStockSection(
                ts_code="000001.SZ",
                name="测试股",
                matched_themes=["AI应用"],
                recommendation="buy",
                recommendation_label="推荐",
                research_depth="standard",
                summary="摘要",
                investment_logic=["逻辑"],
                positive_findings=[],
                growth_catalysts=[],
                technical_analysis=["技术"],
                capital_validation=["资金"],
                trade_plan=["计划"],
                risks=["风险"],
                business_quality_score=68.0,
                business_quality_label="强",
                business_quality_summary="最近12季度经营质量改善。",
                business_quality_bullets=["营收改善", "现金流改善"],
                quarters_analyzed=12,
                source_urls=["https://example.com/stock"],
                chart=trend_agent.ChartArtifact(
                    ts_code="000001.SZ",
                    spike_dates=["2026-03-20"],
                    plotly_html="<div id='chart-a'></div><script>Plotly.newPlot('chart-a',[])</script>",
                ),
            )
        ],
        risks=["总风险"],
    )
    html = trend_agent.render_report_html(report)
    assert html.startswith("<!DOCTYPE html>")
    assert "<script src=" not in html
    assert "<link " not in html
    assert "../charts/" not in html
    assert "Plotly.newPlot" in html
    assert 'id="stock-search"' in html
    assert 'id="theme-filter"' in html
    assert "12季度经营趋势与业务质量" in html
    assert "已分析季度数:" in html


def test_phase5_report_with_deepseek_writes_html_and_md(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "REPORT_DIR", tmp_path / "reports")
    report = trend_agent.ReportModel(
        title="测试研报",
        generated_at="2026-03-22 12:00:00",
        theme_overviews=[],
        core_table_rows=[],
        theme_table_rows=[],
        stock_sections=[],
        risks=["风险"],
    )
    monkeypatch.setattr(trend_agent, "_build_report_model", lambda *args, **kwargs: report)
    artifacts = trend_agent.phase5_report_with_deepseek([], pd.DataFrame(), [], {}, {})
    assert artifacts.html_path.exists()
    assert artifacts.markdown_debug_path.exists()
    assert artifacts.html_path.suffix == ".html"
    assert artifacts.markdown_debug_path.suffix == ".md"
    assert "<!DOCTYPE html>" in artifacts.html_path.read_text(encoding="utf-8")
    assert "# 测试研报" in artifacts.markdown_debug_path.read_text(encoding="utf-8")


def test_run_duckdb_sql_ignores_non_dataframe_context():
    context = {
        "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "score": 1.23}]),
        "signals": {"000001.SZ": {"ready_to_break": True}},
    }
    result = trend_agent.run_duckdb_sql("SELECT ts_code, score FROM candidates_df", context)
    assert "000001.SZ" in result
    assert "score" in result


def test_run_duckdb_sql_returns_error_for_unknown_table():
    context = {
        "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "score": 1.23}]),
        "signals": {"000001.SZ": {"ready_to_break": True}},
    }
    result = trend_agent.run_duckdb_sql("SELECT * FROM table_that_does_not_exist", context)
    assert result.startswith("duckdb_error:")
    assert "available_tables=" in result


def test_run_duckdb_sql_returns_error_for_invalid_sql():
    context = {
        "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "score": 1.23}]),
    }
    result = trend_agent.run_duckdb_sql("SELECT FROM candidates_df", context)
    assert result.startswith("duckdb_error:")
    assert "available_tables=candidates_df" in result


def test_run_duckdb_sql_rejects_write_queries():
    context = {
        "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "score": 1.23}]),
    }
    result = trend_agent.run_duckdb_sql("DROP TABLE candidates_df", context)
    assert result == "duckdb_error: only read-only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN/PRAGMA queries are allowed"


def test_build_duckdb_schema_prompt_lists_real_tables():
    prompt = trend_agent._build_duckdb_schema_prompt(
        {
            "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
            "signal_row": {"ready_to_break": True, "turnover_mult": 1.5},
            "stock_profile": {"ts_code": "000001.SZ", "name": "测试股"},
            "audit_rows": [{"theme": "AI", "verdict": "warn", "confidence_score": 0.5}],
        }
    )
    assert "candidates_df(ts_code, name)" in prompt
    assert "signal_row(ready_to_break, turnover_mult)" in prompt
    assert "stock_profile(ts_code, name)" in prompt
    assert "audit_rows(theme, verdict, confidence_score)" in prompt
    assert "stock_basic(" in prompt
    assert "stock_company(" in prompt
    assert "stock_ticks(" in prompt
    assert "stock_basic_daily(" in prompt
    assert "top_list(" in prompt
    assert "top_inst(" in prompt


def test_run_duckdb_sql_stock_basic_daily_query_works_under_context(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "DATA_ROOT", tmp_path / "data")
    data_root = tmp_path / "data"
    (data_root / "stock_ticks").mkdir(parents=True)
    trend_agent.duckdb.execute(
        f"""
        COPY (
            SELECT
                '000731.SZ' AS ts_code,
                '2026-02-03' AS trade_date,
                10.0 AS open,
                10.5 AS high,
                9.8 AS low,
                10.2 AS close,
                9.9 AS pre_close,
                0.3 AS change,
                3.0 AS pct_chg,
                123456.0 AS vol,
                999999.0 AS amount,
                4.2 AS turnover_rate,
                1.1 AS volume_ratio,
                15.0 AS pe,
                14.5 AS pe_ttm,
                2.1 AS pb,
                3.2 AS ps,
                3.1 AS ps_ttm,
                0.0 AS dv_ratio,
                0.0 AS dv_ttm,
                1000000.0 AS total_share,
                800000.0 AS float_share,
                700000.0 AS free_share,
                500000000.0 AS total_mv,
                400000000.0 AS circ_mv
        ) TO '{(data_root / "stock_ticks" / "000731.SZ.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    sql = (
        "SELECT date, open, high, low, close, volume, turnover_rate "
        "FROM stock_basic_daily WHERE ts_code = '000731.SZ' "
        "AND date >= '2026-02-01' ORDER BY date DESC LIMIT 30"
    )
    result = trend_agent.run_duckdb_sql(sql, {"candidates_df": pd.DataFrame([{"ts_code": "000731.SZ"}])})
    assert "2026-02-03" in result
    assert "123456" in result
    assert "turnover_rate" in result


def test_build_stock_section_system_prompt_includes_duckdb_schema():
    prompt = trend_agent._build_stock_section_system_prompt(
        {
            "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
            "signal_row": {"ready_to_break": True},
            "stock_profile": {"ts_code": "000001.SZ", "name": "测试股"},
        }
    )
    assert "DuckDB tool is read-only." in prompt
    assert "stock_basic_daily" in prompt
    assert "Use stock_basic_daily for daily price queries" in prompt
    assert "candidates_df(ts_code, name)" in prompt
    assert "stock_profile(dict)" in prompt
    assert "signal_row(dict)" in prompt
    assert "优先顺序：1) 直接使用上下文；2) 用python做检查" in prompt
    assert "缺少近期外部证据时，必须优先通过web_search补证" in prompt
    assert "不要为了重复验证财务数值而额外发起web_search" in prompt


def test_run_python_exposes_open_runtime_utilities(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "DATA_ROOT", tmp_path / "data")
    data_root = tmp_path / "data"
    (data_root / "stock_ticks").mkdir(parents=True)
    trend_agent.duckdb.execute(
        f"""
        COPY (
            SELECT
                '000001.SZ' AS ts_code,
                '2026-03-01' AS trade_date,
                10.0 AS open,
                10.5 AS high,
                9.8 AS low,
                10.2 AS close,
                9.9 AS pre_close,
                0.3 AS change,
                3.0 AS pct_chg,
                123456.0 AS vol,
                999999.0 AS amount,
                4.2 AS turnover_rate,
                1.1 AS volume_ratio,
                15.0 AS pe,
                14.5 AS pe_ttm,
                2.1 AS pb,
                3.2 AS ps,
                3.1 AS ps_ttm,
                0.0 AS dv_ratio,
                0.0 AS dv_ttm,
                1000000.0 AS total_share,
                800000.0 AS float_share,
                700000.0 AS free_share,
                500000000.0 AS total_mv,
                400000000.0 AS circ_mv
        ) TO '{(data_root / "stock_ticks" / "000001.SZ.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    context = {
        "stock_profile": {"ts_code": "000001.SZ", "name": "测试股"},
        "signal_row": {"ready_to_break": True, "turnover_mult": 1.8},
        "audit_rows": [{"theme": "AI", "verdict": "warn", "rationale": "题材验证不足", "sources": ["https://example.com"]}],
        "chart_notes": ["2026-02-28"],
        "candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
    }
    result = trend_agent.run_python(
        "show(current_stock_df()); prices = recent_prices(days=30); result = {'rows': len(prices), 'ready': signal_row['ready_to_break'], 'sources': audit_summary()['sources']}",
        context,
    )
    assert "stdout:" in result
    assert "turnover_mult" in result
    assert "result_type: dict" in result
    assert '"rows": 1' in result
    assert '"ready": true' in result


def test_run_python_error_includes_runtime_hints():
    result = trend_agent.run_python("result = boom", {"stock_profile": {"ts_code": "000001.SZ"}})
    assert result.startswith("python_error[NameError]:")
    assert "available_vars=stock_profile, signal_row, audit_rows, chart_notes, candidates_df, pd, np, duckdb" in result
    assert "utilities=show, to_df, current_stock_df, recent_prices, audit_summary" in result


def test_run_duckdb_sql_lazy_registers_only_referenced_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "DATA_ROOT", tmp_path / "data")
    data_root = tmp_path / "data"
    (data_root / "boom").mkdir(parents=True)
    (data_root / "boom" / "bad.parquet").write_text("not parquet", encoding="utf-8")
    monkeypatch.setattr(
        trend_agent,
        "DUCKDB_REPO_TABLE_SPECS",
        {
            "bad_table": {
                "path_parts": ("boom", "bad.parquet"),
                "columns": ["x"],
                "sql": "CREATE VIEW bad_table AS SELECT FROM",
            }
        },
    )
    result = trend_agent.run_duckdb_sql(
        "SELECT ts_code, score FROM candidates_df",
        {"candidates_df": pd.DataFrame([{"ts_code": "000001.SZ", "score": 1.23}])},
    )
    assert "000001.SZ" in result
    assert "score" in result


def test_run_duckdb_sql_invalid_column_reports_schema_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "DATA_ROOT", tmp_path / "data")
    data_root = tmp_path / "data"
    (data_root / "stock_ticks").mkdir(parents=True)
    trend_agent.duckdb.execute(
        f"""
        COPY (
            SELECT
                '000731.SZ' AS ts_code,
                '2026-02-03' AS trade_date,
                10.2 AS close,
                123456.0 AS vol,
                4.2 AS turnover_rate,
                999999.0 AS amount,
                10.0 AS open,
                10.5 AS high,
                9.8 AS low,
                2.1 AS pb,
                15.0 AS pe,
                500000000.0 AS total_mv,
                400000000.0 AS circ_mv
        ) TO '{(data_root / "stock_ticks" / "000731.SZ.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    result = trend_agent.run_duckdb_sql(
        "SELECT ts_code, name, close FROM stock_basic_daily WHERE ts_code = '000731.SZ'",
        {"stock_profile": {"ts_code": "000731.SZ"}},
    )
    assert result.startswith("duckdb_error:")
    assert "referenced_table=stock_basic_daily" in result
    assert "valid_columns=ts_code, date, open, high, low, close, volume, turnover_rate, amount, pe, pb, total_mv, circ_mv" in result
    assert "suggestion=SELECT date, close, volume, turnover_rate FROM stock_basic_daily" in result


def test_execute_agent_tool_catches_duckdb_exception(monkeypatch):
    monkeypatch.setattr(trend_agent, "run_duckdb_sql", lambda sql, context: (_ for _ in ()).throw(RuntimeError("bad duckdb")))
    result = trend_agent._execute_agent_tool("duckdb", "select 1", {"candidates_df": pd.DataFrame()})
    assert result == "tool_error: duckdb failed with RuntimeError: bad duckdb"


def test_execute_agent_tool_runs_web_search(monkeypatch):
    monkeypatch.setattr(trend_agent, "run_search", lambda query: f"search_result: {query}")
    result = trend_agent._execute_agent_tool("web_search", "AI应用 最新进展", {})
    assert result == "search_result: AI应用 最新进展"


def test_execute_agent_tool_catches_web_search_exception(monkeypatch):
    monkeypatch.setattr(trend_agent, "run_search", lambda query: (_ for _ in ()).throw(RuntimeError("search down")))
    result = trend_agent._execute_agent_tool("web_search", "AI应用 最新进展", {})
    assert result == "tool_error: web_search failed with RuntimeError: search down"


def test_execute_agent_tool_runs_python(monkeypatch):
    monkeypatch.setattr(trend_agent, "run_python", lambda code, context: "python_result: ok")
    result = trend_agent._execute_agent_tool("python", "result = 1", {"stock_data": {"ts_code": "000001.SZ"}})
    assert result == "python_result: ok"


def test_execute_agent_tool_catches_python_exception(monkeypatch):
    monkeypatch.setattr(trend_agent, "run_python", lambda code, context: (_ for _ in ()).throw(ValueError("bad python")))
    result = trend_agent._execute_agent_tool("python", "result = 1", {})
    assert result == "tool_error: python failed with ValueError: bad python"


def test_execute_agent_tool_reports_unknown_tool():
    result = trend_agent._execute_agent_tool("shell", "echo 1", {})
    assert result == "tool_error: unknown_tool 'shell'"


def test_generate_stock_section_retries_after_duckdb_error(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "duckdb", "input": "SELECT * FROM stock_basic_daily"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "buy",
                    "summary": "修正查询后完成分析",
                    "investment_logic": ["逻辑A"],
                    "technical_analysis": ["技术A"],
                    "capital_validation": ["资金A"],
                    "trade_plan": ["计划A"],
                    "risks": ["风险A"],
                    "source_urls": ["https://example.com/report", "https://example.com/report-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(
        trend_agent,
        "run_duckdb_sql",
        lambda sql, context: "duckdb_error: Table with name stock_basic_daily does not exist | available_tables=candidates_df",
    )

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": True},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "修正查询后完成分析"
    assert "TOOL_STATUS: error" in seen_tool_feedback["value"]
    assert "TOOL_NAME: duckdb" in seen_tool_feedback["value"]
    assert "duckdb_error:" in seen_tool_feedback["value"]
    assert "available_tables=candidates_df" in seen_tool_feedback["value"]
    assert "DuckDB查询失败" in seen_tool_feedback["value"]
    trace_lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    tool_event = next(json.loads(line) for line in trace_lines if json.loads(line)["event"] == "stock_tool_result")
    assert tool_event["event"] == "stock_tool_result"
    assert tool_event["payload"]["status"] == "error"
    assert "duration_ms" in tool_event["payload"]
    assert tool_event["payload"]["input_preview"] == "SELECT * FROM stock_basic_daily"


def test_generate_stock_section_retries_after_web_search_result(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "web_search", "input": "川润股份 AI应用 最新进展"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "buy",
                    "summary": "结合搜索结果完成分析",
                    "investment_logic": ["逻辑A"],
                    "technical_analysis": ["技术A"],
                    "capital_validation": ["资金A"],
                    "trade_plan": ["计划A"],
                    "risks": ["风险A"],
                    "source_urls": ["https://example.com/report", "https://example.com/report-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(trend_agent, "run_search", lambda query: f"search_result: {query}")

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": True},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "结合搜索结果完成分析"
    assert "TOOL_STATUS: success" in seen_tool_feedback["value"]
    assert "search_result: 川润股份 AI应用 最新进展" in seen_tool_feedback["value"]


def test_generate_stock_section_retries_after_python_result(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "python", "input": "result = context['signals'].get('ready_to_break', False)"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "结合Python结果完成分析",
                    "investment_logic": ["逻辑B"],
                    "technical_analysis": ["技术B"],
                    "capital_validation": ["资金B"],
                    "trade_plan": ["计划B"],
                    "risks": ["风险B"],
                    "source_urls": ["https://example.com/python-result", "https://example.com/python-result-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(trend_agent, "run_python", lambda code, context: "result: True")

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": True},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": []}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "结合Python结果完成分析"
    assert "TOOL_STATUS: success" in seen_tool_feedback["value"]
    assert "result: True" in seen_tool_feedback["value"]
    assert "Python已返回可用分析结果" in seen_tool_feedback["value"]


def test_generate_stock_section_retries_after_web_search_exception(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "web_search", "input": "测试股 新闻"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "搜索异常后仍完成分析",
                    "investment_logic": ["逻辑C"],
                    "technical_analysis": ["技术C"],
                    "capital_validation": ["资金C"],
                    "trade_plan": ["计划C"],
                    "risks": ["风险C"],
                    "source_urls": ["https://example.com/search-error", "https://example.com/search-error-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(
        trend_agent,
        "_execute_agent_tool",
        lambda tool, tool_input, tool_context: "tool_error: web_search failed with RuntimeError: search timeout",
    )

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": False},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": []}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "搜索异常后仍完成分析"
    assert "tool_error: web_search failed with RuntimeError: search timeout" in seen_tool_feedback["value"]
    assert "TOOL_STATUS: error" in seen_tool_feedback["value"]


def test_generate_stock_section_retries_after_python_exception(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "python", "input": "raise Exception('boom')"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "Python异常后仍完成分析",
                    "investment_logic": ["逻辑D"],
                    "technical_analysis": ["技术D"],
                    "capital_validation": ["资金D"],
                    "trade_plan": ["计划D"],
                    "risks": ["风险D"],
                    "source_urls": ["https://example.com/python-error", "https://example.com/python-error-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(
        trend_agent,
        "_execute_agent_tool",
        lambda tool, tool_input, tool_context: "tool_error: python failed with ValueError: bad python",
    )

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": False},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": []}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "Python异常后仍完成分析"
    assert "tool_error: python failed with ValueError: bad python" in seen_tool_feedback["value"]
    assert "Python运行失败" in seen_tool_feedback["value"]


def test_generate_stock_section_does_not_crash_on_tool_exception(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = {"value": ""}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "duckdb", "input": "SELECT * FROM candidates_df"}, ensure_ascii=False)
        seen_tool_feedback["value"] = messages[-1]["content"]
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "工具异常后仍成功回退",
                    "investment_logic": ["逻辑B"],
                    "technical_analysis": ["技术B"],
                    "capital_validation": ["资金B"],
                    "trade_plan": ["计划B"],
                    "risks": ["风险B"],
                    "source_urls": ["https://example.com/tool-fallback", "https://example.com/tool-fallback-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(
        trend_agent,
        "_execute_agent_tool",
        lambda tool, tool_input, tool_context: "tool_error: duckdb failed with RuntimeError: boom",
    )

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": False},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": []}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "工具异常后仍成功回退"
    assert "tool_error: duckdb failed with RuntimeError: boom" in seen_tool_feedback["value"]


def test_generate_stock_section_switches_to_python_after_duckdb_error(monkeypatch, tmp_path):
    calls = {"n": 0}
    seen_tool_feedback = []

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "duckdb", "input": "SELECT bad_col FROM stock_basic_daily"}, ensure_ascii=False)
        if calls["n"] == 2:
            seen_tool_feedback.append(messages[-1]["content"])
            return json.dumps({"tool": "python", "input": "result = {'ready': signal_row.get('ready_to_break')}"}, ensure_ascii=False)
        seen_tool_feedback.append(messages[-1]["content"])
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "DuckDB失败后改用Python完成分析",
                    "investment_logic": ["逻辑E"],
                    "technical_analysis": ["技术E"],
                    "capital_validation": ["资金E"],
                    "trade_plan": ["计划E"],
                    "risks": ["风险E"],
                    "source_urls": ["https://example.com/python-after-duckdb", "https://example.com/python-after-duckdb-2"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    def fake_tool(tool, tool_input, tool_context):
        if tool == "duckdb":
            return "duckdb_error: Binder Error: Referenced column bad_col not found | available_tables=candidates_df, stock_basic_daily"
        if tool == "python":
            return "result_type: dict\nresult_preview:\n{\n  \"ready\": true\n}"
        raise AssertionError(tool)

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(trend_agent, "_execute_agent_tool", fake_tool)

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": True},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": []}
    tool_stats = {"total_calls": 0, "per_tool": {}, "python_after_duckdb_failure": 0}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
        tool_stats=tool_stats,
    )
    assert section.summary == "DuckDB失败后改用Python完成分析"
    assert tool_stats["python_after_duckdb_failure"] == 1
    assert any("DuckDB查询失败" in msg for msg in seen_tool_feedback)
    assert any("TOOL_NAME: python" in msg for msg in seen_tool_feedback)


def test_generate_stock_section_requests_web_search_when_sources_are_weak(monkeypatch, tmp_path):
    calls = {"n": 0}
    user_messages = []

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "stock": {
                        "ts_code": "000001.SZ",
                        "name": "测试股",
                        "recommendation": "watch",
                        "summary": "先给出结论，但来源不足",
                        "investment_logic": ["逻辑A"],
                        "positive_findings": [
                            {"category": "policy", "description": "发现催化", "evidence": "e", "confidence": 0.6, "source_url": ""}
                        ],
                        "technical_analysis": ["技术A"],
                        "capital_validation": ["资金A"],
                        "trade_plan": ["计划A"],
                        "risks": ["风险A"],
                        "source_urls": [],
                        "research_depth": "standard",
                    }
                },
                ensure_ascii=False,
            )
        if calls["n"] == 2:
            user_messages.append(messages[-1]["content"])
            return json.dumps({"tool": "web_search", "input": "测试股 AI应用 最新公告 客户 订单"}, ensure_ascii=False)
        user_messages.append(messages[-1]["content"])
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "补完来源后的结论",
                    "investment_logic": ["逻辑A"],
                    "technical_analysis": ["技术A"],
                    "capital_validation": ["资金A"],
                    "trade_plan": ["计划A"],
                    "risks": ["风险A"],
                    "source_urls": ["https://example.com/a", "https://example.com/b"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)
    monkeypatch.setattr(trend_agent, "_execute_agent_tool", lambda tool, tool_input, tool_context: "search_result: ok")

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": False},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "补完来源后的结论"
    assert section.source_urls == ["https://example.com/a", "https://example.com/b"]
    assert any("请先调用 web_search 补充近期、可点击、可核验的外部来源" in msg for msg in user_messages)


def test_generate_stock_section_accepts_payload_with_strong_sources(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_deepseek_chat(messages):
        calls["n"] += 1
        return json.dumps(
            {
                "stock": {
                    "ts_code": "000001.SZ",
                    "name": "测试股",
                    "recommendation": "watch",
                    "summary": "来源充足",
                    "investment_logic": ["逻辑A"],
                    "positive_findings": [
                        {"category": "policy", "description": "发现催化", "evidence": "e", "confidence": 0.6, "source_url": "https://example.com/a"}
                    ],
                    "technical_analysis": ["技术A"],
                    "capital_validation": ["资金A"],
                    "trade_plan": ["计划A"],
                    "risks": ["风险A"],
                    "source_urls": ["https://example.com/a", "https://example.com/b"],
                    "research_depth": "standard",
                }
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "deepseek_chat", fake_deepseek_chat)

    ctx = {
        "ts_code": "000001.SZ",
        "name": "测试股",
        "stock_data": {"ts_code": "000001.SZ", "name": "测试股"},
        "signals": {"ready_to_break": False},
        "audits": [],
        "chart_notes": [],
    }
    row = {"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}
    section = trend_agent._generate_stock_section(
        ctx,
        tmp_path / "trace.jsonl",
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股"}]),
        row,
        [],
        None,
    )
    assert section.summary == "来源充足"
    assert calls["n"] == 1


def test_gemma_match_guard_blocks_scope_only_false_positive(monkeypatch, tmp_path):
    def fake_invoke(provider, messages, temperature=0.1):
        payload = json.loads(messages[-1]["content"])
        ts_code = payload["stocks"][0]["ts_code"]
        return json.dumps(
            {"matches": {ts_code: ["算力基建与通信（光模块/通信设备）"]}, "notes": {ts_code: "直接匹配"}},
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "300784.SZ",
                "name": "利安科技",
                "industry": "塑料",
                "main_business": "主营业务为精密注塑模具和注塑产品",
                "business_scope": "一般项目: 通信设备制造; 人工智能硬件销售; 电子产品销售",
                "introduction": "模塑一体化生产企业。",
            }
        ]
    )
    themes = [
        ThemeItem(
            name="算力基建与通信（光模块/通信设备）",
            keywords=["算力", "光模块", "通信设备"],
            summary="",
            sources=[],
        )
    ]
    out = trend_agent.gemma_match_themes(themes, candidates)
    assert out.iloc[0]["matched_themes"] == []


def test_gemma_match_guard_keeps_valid_chemical_match(monkeypatch, tmp_path):
    def fake_invoke(provider, messages, temperature=0.1):
        payload = json.loads(messages[-1]["content"])
        ts_code = payload["stocks"][0]["ts_code"]
        return json.dumps(
            {"matches": {ts_code: ["化工与周期材料"]}, "notes": {ts_code: "直接匹配"}},
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "601065.SH",
                "name": "江盐集团",
                "industry": "化工原料",
                "main_business": "专注于盐化工产品研发、生产和销售",
                "business_scope": "食盐、工业盐、盐化工产品的生产和销售",
                "introduction": "盐资源综合开发利用企业。",
            }
        ]
    )
    themes = [
        ThemeItem(
            name="化工与周期材料",
            keywords=["盐化工", "化工产品"],
            summary="",
            sources=[],
        )
    ]
    out = trend_agent.gemma_match_themes(themes, candidates)
    assert out.iloc[0]["matched_themes"] == ["化工与周期材料"]


class _DummyResponse:
    def __init__(self, retry_after: str = "0"):
        self.status_code = 429
        self.headers = {"Retry-After": retry_after}


class DummyRateLimitError(Exception):
    def __init__(self, message: str = "rate limited", retry_after: str = "0"):
        super().__init__(message)
        self.status_code = 429
        self.response = _DummyResponse(retry_after=retry_after)


def test_gemma_match_retries_then_succeeds(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_sleep(_seconds):
        return None

    def fake_invoke(provider, messages, temperature=0.1):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise DummyRateLimitError("TPM limit reached", retry_after="0")
        payload = json.loads(messages[-1]["content"])
        ts_code = payload["stocks"][0]["ts_code"]
        return json.dumps({"matches": {ts_code: ["AI应用"]}, "notes": {ts_code: "ok"}}, ensure_ascii=False)

    monkeypatch.setattr(trend_agent.time, "sleep", fake_sleep)
    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)

    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "测试A",
                "industry": "软件",
                "main_business": "AI应用产品研发和销售",
                "business_scope": "软件",
                "introduction": "AI应用相关介绍",
            }
        ]
    )
    themes = [ThemeItem(name="AI应用", keywords=["AI应用"], summary="", sources=[])]
    cfg = StrategyConfig(
        gemma_rate_limit_max_retries=6,
        gemma_rate_limit_base_delay_sec=0.01,
        gemma_rate_limit_max_delay_sec=0.05,
        gemma_request_interval_sec=0.0,
    )
    out = trend_agent.gemma_match_themes(
        themes,
        candidates,
        config=cfg,
    )
    assert call_count["n"] == 3
    assert out.iloc[0]["matched_themes"] == ["AI应用"]


def test_gemma_match_exhausted_rate_limit_degrades_to_empty(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_sleep(_seconds):
        return None

    def fake_invoke(provider, messages, temperature=0.1):
        call_count["n"] += 1
        raise DummyRateLimitError("TPM limit reached", retry_after="0")

    monkeypatch.setattr(trend_agent.time, "sleep", fake_sleep)
    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)

    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "测试A",
                "industry": "软件",
                "main_business": "主营软件产品",
                "business_scope": "软件",
                "introduction": "介绍",
            }
        ]
    )
    themes = [ThemeItem(name="AI应用", keywords=["AI应用"], summary="", sources=[])]
    cfg = StrategyConfig(
        gemma_rate_limit_max_retries=1,
        gemma_rate_limit_base_delay_sec=0.01,
        gemma_rate_limit_max_delay_sec=0.05,
        gemma_request_interval_sec=0.0,
    )
    out = trend_agent.gemma_match_themes(
        themes,
        candidates,
        config=cfg,
    )
    assert call_count["n"] == 2
    assert out.iloc[0]["matched_themes"] == []


def test_gemma_match_respects_configured_batch_size(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_invoke(provider, messages, temperature=0.1):
        call_count["n"] += 1
        payload = json.loads(messages[-1]["content"])
        matches = {stock["ts_code"]: ["AI应用"] for stock in payload["stocks"]}
        notes = {stock["ts_code"]: "ok" for stock in payload["stocks"]}
        return json.dumps({"matches": matches, "notes": notes}, ensure_ascii=False)

    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)

    candidates = pd.DataFrame(
        [
            {
                "ts_code": f"00000{i}.SZ",
                "name": f"测试{i}",
                "industry": "软件",
                "main_business": "AI应用产品研发",
                "business_scope": "软件",
                "introduction": "AI应用",
            }
            for i in range(1, 6)
        ]
    )
    themes = [ThemeItem(name="AI应用", keywords=["AI应用"], summary="", sources=[])]
    cfg = StrategyConfig(gemma_batch_size=2, gemma_request_interval_sec=0.0)
    out = trend_agent.gemma_match_themes(
        themes,
        candidates,
        config=cfg,
    )
    assert call_count["n"] == 3
    assert out["matched_themes"].apply(lambda x: x == ["AI应用"]).all()


def test_phase2_quant_filter_survives_gemma_rate_limit(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    def fake_sleep(_seconds):
        return None

    def fake_invoke(provider, messages, temperature=0.1):
        raise DummyRateLimitError("TPM limit reached", retry_after="0")

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        out = candidates.copy()
        matched = []
        for idx, _ in enumerate(out.index):
            matched.append(["AI"] if idx == 0 else [])
        out["matched_themes"] = matched
        return out

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    monkeypatch.setattr(trend_agent.time, "sleep", fake_sleep)
    monkeypatch.setattr(trend_agent, "invoke_llm_messages", fake_invoke)
    monkeypatch.setattr(trend_agent, "heuristic_match_themes", fake_heuristic)
    monkeypatch.setattr(
        trend_agent,
        "screen_all_stocks",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "A",
                    "industry": "I1",
                    "main_business": "主营业务A",
                    "business_scope": "业务范围A",
                    "introduction": "介绍A",
                    "consolidation_score": 65,
                    "ma_spread": 0.18,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.3,
                    "composite_score": 70.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "name": "B",
                    "industry": "I2",
                    "main_business": "主营业务B",
                    "business_scope": "业务范围B",
                    "introduction": "介绍B",
                    "consolidation_score": 62,
                    "ma_spread": 0.19,
                    "ma_spread_std": 0.04,
                    "volume_boost": 1.2,
                    "composite_score": 68.0,
                },
            ]
        ),
    )

    cfg = StrategyConfig(
        theme_match_policy="aggressive",
        gemma_batch_size=1,
        gemma_rate_limit_max_retries=1,
        gemma_rate_limit_base_delay_sec=0.01,
        gemma_rate_limit_max_delay_sec=0.05,
        gemma_request_interval_sec=0.0,
    )
    themes = [ThemeItem(name="AI", keywords=["算力"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)
    assert not out.empty
    assert "matched_themes" in out.columns


def test_compute_industry_relative_valuation_labels_extreme_outlier():
    df = pd.DataFrame(
        [
            {"ts_code": "A", "industry": "半导体", "pe": 20.0, "pb": 2.0, "ps_ttm": 3.0},
            {"ts_code": "B", "industry": "半导体", "pe": 22.0, "pb": 2.2, "ps_ttm": 3.2},
            {"ts_code": "C", "industry": "半导体", "pe": 24.0, "pb": 2.4, "ps_ttm": 3.4},
            {"ts_code": "D", "industry": "半导体", "pe": 26.0, "pb": 2.6, "ps_ttm": 3.6},
            {"ts_code": "E", "industry": "半导体", "pe": 28.0, "pb": 2.8, "ps_ttm": 3.8},
            {"ts_code": "F", "industry": "半导体", "pe": 150.0, "pb": 15.0, "ps_ttm": 20.0},
        ]
    )

    out = screen_growth_stocks.compute_industry_relative_valuation(df, outlier_percentile=0.90, peer_min_samples=5)
    cheap = out[out["ts_code"] == "A"].iloc[0]
    expensive = out[out["ts_code"] == "F"].iloc[0]

    assert cheap["valuation_label"] == "合理"
    assert expensive["valuation_label"] == "显著高估"
    assert bool(expensive["valuation_outlier"]) is True
    assert expensive["valuation_stretch_score"] > cheap["valuation_stretch_score"]


def test_compute_industry_relative_valuation_falls_back_to_market_for_sparse_industry():
    df = pd.DataFrame(
        [
            {"ts_code": "A", "industry": "银行", "pe": 5.0, "pb": 0.8, "ps_ttm": 1.0},
            {"ts_code": "B", "industry": "银行", "pe": 6.0, "pb": 0.9, "ps_ttm": 1.1},
            {"ts_code": "C", "industry": "银行", "pe": 7.0, "pb": 1.0, "ps_ttm": 1.2},
            {"ts_code": "D", "industry": "银行", "pe": 8.0, "pb": 1.1, "ps_ttm": 1.3},
            {"ts_code": "E", "industry": "银行", "pe": 9.0, "pb": 1.2, "ps_ttm": 1.4},
            {"ts_code": "F", "industry": "软件", "pe": 30.0, "pb": 3.0, "ps_ttm": 5.0},
            {"ts_code": "G", "industry": "软件", "pe": -1.0, "pb": 4.0, "ps_ttm": 6.0},
        ]
    )

    out = screen_growth_stocks.compute_industry_relative_valuation(df, peer_min_samples=5)
    sparse = out[out["ts_code"] == "F"].iloc[0]

    assert sparse["pe_baseline_source"] == "market"
    assert sparse["pb_baseline_source"] == "market"
    assert sparse["valuation_data_points"] >= 2


def test_phase2_theme_ranking_penalizes_expensive_matches(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    monkeypatch.setattr(
        trend_agent,
        "screen_all_stocks",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "便宜股",
                    "industry": "软件",
                    "main_business": "AI软件",
                    "business_scope": "AI软件",
                    "introduction": "AI软件",
                    "consolidation_score": 72,
                    "momentum_score": 75,
                    "volume_quality_score": 70,
                    "volume_boost": 1.5,
                    "composite_score": 78.0,
                    "valuation_quality_score": 82.0,
                    "valuation_stretch_score": 18.0,
                    "valuation_label": "合理",
                },
                {
                    "ts_code": "000002.SZ",
                    "name": "贵股",
                    "industry": "软件",
                    "main_business": "AI软件",
                    "business_scope": "AI软件",
                    "introduction": "AI软件",
                    "consolidation_score": 72,
                    "momentum_score": 75,
                    "volume_quality_score": 70,
                    "volume_boost": 1.5,
                    "composite_score": 78.0,
                    "valuation_quality_score": 20.0,
                    "valuation_stretch_score": 92.0,
                    "valuation_label": "显著高估",
                },
            ]
        ),
    )

    def fake_match(themes, candidates, config=None, relaxed_validation=False, named_stock_codes=None):
        out = candidates.copy()
        out["matched_themes"] = [["AI应用"] for _ in range(len(out))]
        return out

    monkeypatch.setattr(trend_agent, "gemma_match_themes", fake_match)
    themes = [ThemeItem(name="AI应用", keywords=["AI"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=StrategyConfig(toplist_exclusion_mode="penalty"))

    theme_rows = out[out["list_type"] == "theme_driven"].sort_values("alpha_rank_score", ascending=False)
    assert theme_rows.iloc[0]["ts_code"] == "000001.SZ"
    assert theme_rows.iloc[0]["valuation_label"] == "合理"


def test_phase2_theme_pre_audit_cap_expands_to_20_and_keeps_technical(monkeypatch):
    class FakeDTL:
        def load_recent_toplist(self, days=60):
            return pd.DataFrame(columns=["ts_code"])

    monkeypatch.setattr(trend_agent, "DragonTigerList", FakeDTL)
    rows = []
    for i in range(25):
        rows.append(
            {
                "ts_code": f"{i:06d}.SZ",
                "name": f"S{i}",
                "industry": f"I{i % 3}",
                "main_business": "AI软件",
                "business_scope": "AI软件",
                "introduction": "AI软件",
                "consolidation_score": 70 + (i % 5),
                "momentum_score": 80 - i * 0.5,
                "volume_quality_score": 68.0,
                "volume_boost": 1.4,
                "composite_score": 85.0 - i,
                "valuation_quality_score": 60.0,
                "valuation_stretch_score": 40.0,
                "valuation_label": "适中溢价",
            }
        )
    monkeypatch.setattr(trend_agent, "screen_all_stocks", lambda: pd.DataFrame(rows))

    def fake_match(themes, candidates, config=None, relaxed_validation=False, named_stock_codes=None):
        out = candidates.copy()
        out["matched_themes"] = [["AI应用"] for _ in range(len(out))]
        return out

    monkeypatch.setattr(trend_agent, "gemma_match_themes", fake_match)
    out = trend_agent.phase2_quant_filter(
        [ThemeItem(name="AI应用", keywords=["AI"], summary="", sources=[], validation_status="confirmed")],
        config=StrategyConfig(theme_pre_audit_cap=20, theme_post_audit_cap=5),
    )
    theme_rows = out[out["list_type"] == "theme_driven"]
    tech_rows = out[out["list_type"] == "technical"]
    assert len(theme_rows) == 20
    assert not tech_rows.empty


def test_rank_candidates_for_alpha_prefers_reasonable_valuation():
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "industry": "I1",
                "matched_themes": ["AI"],
                "list_type": "theme_driven",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.1,
                "valuation_quality_score": 80.0,
                "valuation_stretch_score": 20.0,
                "valuation_label": "合理",
            },
            {
                "ts_code": "000002.SZ",
                "name": "B",
                "industry": "I1",
                "matched_themes": ["AI"],
                "list_type": "theme_driven",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.1,
                "valuation_quality_score": 25.0,
                "valuation_stretch_score": 90.0,
                "valuation_label": "显著高估",
            },
        ]
    )
    signals = {
        "000001.SZ": {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8},
        "000002.SZ": {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8},
    }

    ranked = trend_agent.rank_candidates_for_alpha(candidates, audits=[], signals=signals, config=StrategyConfig())
    assert ranked.iloc[0]["ts_code"] == "000001.SZ"


def test_phase3_deep_audit_feeds_opportunity_followup_results_back_to_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "analyze_business_quality_with_fallbacks", lambda ts_code, name, max_quarters=12: {
        "quarters_analyzed": 4,
        "business_quality_score": 58.0,
        "business_quality_label": "中性",
        "business_quality_summary": "经营平稳",
        "business_quality_bullets": ["营收平稳"],
        "financial_data_source": "local",
    })
    monkeypatch.setattr(deep_researcher, "generate_opportunity_queries", lambda name, theme: [])
    monkeypatch.setattr(
        deep_researcher,
        "deepseek_plan_opportunity_queries",
        lambda name, theme, current_findings, evidence_gaps: {
            "queries": [f"{name} 订单 公告"],
            "focus_areas": ["订单"],
        },
    )
    synthesis_calls = []
    monkeypatch.setattr(
        deep_researcher,
        "deepseek_synthesize_opportunity_results",
        lambda name, theme, current_findings, followup_results: (
            synthesis_calls.append(
                {
                    "name": name,
                    "theme": theme,
                    "current_findings": current_findings,
                    "followup_results": followup_results,
                }
            )
            or {
                "positive_findings": [
                    {
                        "category": "contract_evidence",
                        "description": "中标订单落地",
                        "evidence": "公司公告披露中标。",
                        "confidence": 0.8,
                        "source_url": "https://example.com/order",
                        "date": "2026-04-10",
                    }
                ],
                "growth_catalysts": [
                    {
                        "catalyst_type": "contract_evidence",
                        "description": "订单进入兑现期",
                        "timeframe": "near_term",
                        "confidence": 0.75,
                    }
                ],
                "reason": "followup synthesized",
            }
        ),
    )
    monkeypatch.setattr(
        trend_agent,
        "run_search",
        lambda query: json.dumps(
            {
                "results": [
                    {
                        "title": "公司中标公告",
                        "snippet": "测试股公告披露中标大单",
                        "url": "https://example.com/order",
                        "date": "2026-04-10",
                    }
                ],
                "urls": ["https://example.com/order"],
            },
            ensure_ascii=False,
        ),
    )
    monkeypatch.setattr(trend_agent, "invoke_with_timeout_retries", lambda fn, **kwargs: fn())

    llm = RunnableLambda(
        lambda _: json.dumps(
            {
                "verdict": "pass",
                "rationale": "无明显风险",
                "sources": ["https://example.com/order"],
            },
            ensure_ascii=False,
        )
    )
    candidates = pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}])
    trace_path = tmp_path / "audit_trace.jsonl"
    audits = trend_agent.phase3_deep_audit(
        llm,
        candidates,
        trace_path=trace_path,
        themes=[ThemeItem(name="AI应用", keywords=["AI"], summary="", sources=[])],
    )

    assert len(audits) == 1
    assert synthesis_calls
    assert synthesis_calls[0]["followup_results"][0]["results"][0]["url"] == "https://example.com/order"
    assert any(f.description == "中标订单落地" for f in audits[0].positive_findings)
    assert any(c.description == "订单进入兑现期" for c in audits[0].growth_catalysts)
    trace_text = trace_path.read_text(encoding="utf-8")
    assert '"event": "opportunity_followup"' in trace_text
    assert '"event": "opportunity_llm"' in trace_text


def test_phase3_deep_audit_opportunity_llm_failure_falls_back_to_local_extraction(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "analyze_business_quality_with_fallbacks", lambda ts_code, name, max_quarters=12: {
        "quarters_analyzed": 4,
        "business_quality_score": 58.0,
        "business_quality_label": "中性",
        "business_quality_summary": "经营平稳",
        "business_quality_bullets": ["营收平稳"],
        "financial_data_source": "local",
    })
    monkeypatch.setattr(deep_researcher, "generate_opportunity_queries", lambda name, theme: [])
    monkeypatch.setattr(
        deep_researcher,
        "deepseek_plan_opportunity_queries",
        lambda name, theme, current_findings, evidence_gaps: {"queries": [f"{name} 订单 公告"], "focus_areas": ["订单"]},
    )
    monkeypatch.setattr(deep_researcher, "deepseek_synthesize_opportunity_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trend_agent,
        "run_search",
        lambda query: json.dumps(
            {
                "results": [
                    {
                        "title": "测试股中标公告",
                        "snippet": "测试股 中标 框架协议",
                        "url": "https://example.com/local-order",
                        "date": "2026-04-10",
                    }
                ],
                "urls": ["https://example.com/local-order"],
            },
            ensure_ascii=False,
        ),
    )
    monkeypatch.setattr(trend_agent, "invoke_with_timeout_retries", lambda fn, **kwargs: fn())

    llm = RunnableLambda(
        lambda _: json.dumps(
            {
                "verdict": "pass",
                "rationale": "无明显风险",
                "sources": ["https://example.com/local-order"],
            },
            ensure_ascii=False,
        )
    )
    candidates = pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}])
    trace_path = tmp_path / "audit_trace.jsonl"
    audits = trend_agent.phase3_deep_audit(
        llm,
        candidates,
        trace_path=trace_path,
        themes=[ThemeItem(name="AI应用", keywords=["AI"], summary="", sources=[])],
    )

    assert len(audits) == 1
    assert any(f.source_url == "https://example.com/local-order" for f in audits[0].positive_findings)
    trace_text = trace_path.read_text(encoding="utf-8")
    assert '"event": "opportunity_llm"' in trace_text


def test_build_veto_planning_evidence_uses_full_accumulated_state():
    merged = {
        f"pass1_{i}": {
            "query": f"查询{i}",
            "urls": [f"https://example.com/{i}"],
            "results": [
                {
                    "title": f"普通结果{i}",
                    "snippet": f"普通摘要{i}",
                    "url": f"https://example.com/{i}",
                    "date": "2026-04-10",
                }
            ],
        }
        for i in range(6)
    }
    merged["pass1_0"]["results"][0]["title"] = "早期监管函线索"
    merged["pass1_0"]["results"][0]["snippet"] = "早期结果提示监管函，但不是最后四条证据。"
    snapshot = {
        "quarters_analyzed": 8,
        "business_quality_label": "中性",
        "business_quality_score": 55.0,
        "business_quality_summary": "营收平稳。",
        "business_quality_bullets": ["现金流改善"],
    }

    evidence = trend_agent.build_veto_planning_evidence("本地简报", snapshot, merged, max_chars=4000)
    assert "早期监管函线索" in evidence
    assert "查询0" in evidence
    assert "营收平稳" in evidence


def test_phase3_veto_plan_trace_includes_compact_evidence_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(trend_agent, "analyze_business_quality_with_fallbacks", lambda ts_code, name, max_quarters=12: {
        "quarters_analyzed": 6,
        "business_quality_score": 52.0,
        "business_quality_label": "中性",
        "business_quality_summary": "经营基本稳定",
        "business_quality_bullets": ["收入波动不大"],
        "financial_data_source": "local",
    })
    monkeypatch.setattr(deep_researcher, "generate_opportunity_queries", lambda name, theme: [])
    monkeypatch.setattr(deep_researcher, "deepseek_plan_opportunity_queries", lambda *args, **kwargs: None)
    monkeypatch.setattr(deep_researcher, "deepseek_synthesize_opportunity_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(trend_agent, "invoke_with_timeout_retries", lambda fn, **kwargs: fn())

    captured = {}

    def fake_plan_queries(name, theme, evidence, pass_id):
        captured["evidence"] = evidence
        return {"stop": True, "reason": "evidence reviewed", "queries": []}

    monkeypatch.setattr(trend_agent, "deepseek_plan_queries", fake_plan_queries)
    monkeypatch.setattr(
        trend_agent,
        "run_search",
        lambda query: json.dumps(
            {
                "summary": "早期监管函线索，但未触发硬否决。",
                "results": [
                    {
                        "title": "早期监管函线索",
                        "snippet": "测试股收到监管函并已回复，需纳入后续规划上下文。",
                        "url": "https://example.com/reg-letter",
                        "date": "2026-04-10",
                    }
                ],
                "urls": ["https://example.com/reg-letter"],
            },
            ensure_ascii=False,
        ),
    )

    llm = RunnableLambda(
        lambda _: json.dumps(
            {
                "verdict": "warn",
                "rationale": "需要继续查证",
                "sources": ["https://example.com/reg-letter"],
            },
            ensure_ascii=False,
        )
    )
    candidates = pd.DataFrame([{"ts_code": "000001.SZ", "name": "测试股", "matched_themes": ["AI应用"]}])
    trace_path = tmp_path / "audit_trace.jsonl"
    audits = trend_agent.phase3_deep_audit(
        llm,
        candidates,
        trace_path=trace_path,
        themes=[ThemeItem(name="AI应用", keywords=["AI"], summary="", sources=[])],
    )

    assert len(audits) == 1
    assert "早期监管函线索" in captured["evidence"]
    trace_text = trace_path.read_text(encoding="utf-8")
    assert '"event": "veto_plan"' in trace_text
    assert '"evidence_chars"' in trace_text
    assert '"query_count"' in trace_text


def test_rank_candidates_for_alpha_trims_theme_post_audit_to_top_5_and_backfills_technical():
    candidates = []
    signals = {}
    for i in range(7):
        ts_code = f"T{i:06d}.SZ"
        candidates.append(
            {
                "ts_code": ts_code,
                "name": f"T{i}",
                "industry": f"TI{i}",
                "matched_themes": ["AI"],
                "list_type": "theme_driven",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.0,
                "valuation_quality_score": 70.0,
                "valuation_stretch_score": 30.0,
                "valuation_label": "合理",
            }
        )
        signals[ts_code] = {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8}
    for i in range(5):
        ts_code = f"K{i:06d}.SZ"
        candidates.append(
            {
                "ts_code": ts_code,
                "name": f"K{i}",
                "industry": f"KI{i}",
                "matched_themes": [],
                "list_type": "technical",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 0.0,
                "toplist_recency_score": 0.0,
                "valuation_quality_score": 70.0,
                "valuation_stretch_score": 30.0,
                "valuation_label": "合理",
            }
        )
        signals[ts_code] = {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8}

    ranked = trend_agent.rank_candidates_for_alpha(
        pd.DataFrame(candidates),
        audits=[],
        signals=signals,
        config=StrategyConfig(theme_post_audit_cap=5),
    )
    theme_count = int(ranked["list_type"].isin(["theme_driven", "both"]).sum())
    tech_count = int((ranked["list_type"] == "technical").sum())
    assert theme_count == 5
    assert tech_count >= 5
    assert len(ranked) == 10


def test_deterministic_tables_include_valuation_column():
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "industry": "I1",
                "matched_themes": ["AI"],
                "off_theme": False,
                "theme_strength_score": 1.0,
                "momentum_score": 75.0,
                "alpha_rank_score": 88.0,
                "consolidation_score": 70,
                "volume_boost": 1.5,
                "valuation_label": "适中溢价",
                "list_type": "theme_driven",
            },
            {
                "ts_code": "000002.SZ",
                "name": "B",
                "industry": "I2",
                "matched_themes": [],
                "off_theme": True,
                "alpha_rank_score": 70.0,
                "consolidation_score": 68,
                "volume_boost": 1.3,
                "valuation_label": "合理",
                "list_type": "technical",
            },
        ]
    )

    theme_table = trend_agent.build_deterministic_theme_table(candidates, audits=[], top_n=1)
    core_table = trend_agent.build_deterministic_core_table(candidates, audits=[], top_n=1)

    assert "| 股票 | 匹配题材 | 题材强度 | 估值 | 动量评分 | Alpha评分 |" in theme_table
    assert "| 股票 | 所属主线 | 估值 | 形态特征 | 置信度 | 推荐理由 |" in core_table
    assert "适中溢价" in theme_table
    assert "合理" in core_table


def test_analyze_business_quality_improving_unprofitable_company(monkeypatch):
    monkeypatch.setattr(
        trend_agent,
        "load_financial_quarters",
        lambda ts_code, max_quarters=12: pd.DataFrame(
            {
                "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
                "revenue": [100, 105, 108, 112, 120, 130, 138, 148],
                "gross_margin": [18.0, 18.5, 19.0, 19.5, 20.5, 21.0, 21.8, 22.4],
                "n_cashflow_act": [-20, -12, -8, -4, 2, 6, 10, 15],
                "net_income": [-35, -32, -28, -24, -20, -16, -10, -5],
            }
        ),
    )
    snap = trend_agent.analyze_business_quality("000001.SZ")
    assert snap["quarters_analyzed"] == 8
    assert snap["business_quality_score"] > 60
    assert snap["business_quality_label"] == "强"
    assert any("亏损" in bullet or "营收" in bullet for bullet in snap["business_quality_bullets"])


def test_analyze_business_quality_deteriorating_business(monkeypatch):
    monkeypatch.setattr(
        trend_agent,
        "load_financial_quarters",
        lambda ts_code, max_quarters=12: pd.DataFrame(
            {
                "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
                "revenue": [180, 176, 170, 166, 155, 145, 132, 120],
                "gross_margin": [28.0, 27.5, 27.0, 26.0, 24.5, 23.0, 21.5, 20.0],
                "n_cashflow_act": [30, 28, 24, 20, 12, 4, -6, -15],
                "net_income": [20, 18, 15, 12, 8, 4, -2, -8],
            }
        ),
    )
    snap = trend_agent.analyze_business_quality("000002.SZ")
    assert snap["business_quality_score"] < 45
    assert snap["business_quality_label"] == "偏弱"


def test_analyze_business_quality_missing_data_is_neutral(monkeypatch):
    monkeypatch.setattr(trend_agent, "load_financial_quarters", lambda ts_code, max_quarters=12: pd.DataFrame())
    snap = trend_agent.analyze_business_quality("000003.SZ")
    assert snap["quarters_analyzed"] == 0
    assert snap["business_quality_score"] == 50.0
    assert snap["business_quality_label"] == "中性"
    assert snap["financial_data_source"] == "none"


def test_analyze_business_quality_prefers_tushare_before_web(monkeypatch):
    monkeypatch.setattr(trend_agent, "load_financial_quarters", lambda ts_code, max_quarters=12: pd.DataFrame())
    monkeypatch.setattr(
        trend_agent,
        "load_financial_quarters_from_tushare",
        lambda ts_code, max_quarters=12: pd.DataFrame(
            {
                "end_date": pd.date_range("2023-03-31", periods=6, freq="QE"),
                "revenue": [100, 106, 110, 118, 126, 135],
                "gross_margin": [18.0, 18.3, 18.8, 19.5, 20.2, 21.0],
                "n_cashflow_act": [-5, 2, 6, 9, 12, 16],
                "net_income": [-12, -9, -5, -2, 1, 4],
            }
        ),
    )
    monkeypatch.setattr(
        trend_agent,
        "load_financial_quarters_from_web",
        lambda ts_code, name, max_quarters=12: (_ for _ in ()).throw(AssertionError("web fallback should not run")),
    )
    snap = trend_agent.analyze_business_quality_with_fallbacks("000001.SZ", "测试股")
    assert snap["quarters_analyzed"] == 6
    assert snap["business_quality_score"] > 50
    assert snap["financial_data_source"] == "tushare"


def test_analyze_business_quality_uses_web_when_tushare_missing(monkeypatch):
    monkeypatch.setattr(trend_agent, "load_financial_quarters", lambda ts_code, max_quarters=12: pd.DataFrame())
    monkeypatch.setattr(trend_agent, "load_financial_quarters_from_tushare", lambda ts_code, max_quarters=12: pd.DataFrame())

    def fake_search(query: str) -> str:
        if "年度报告摘要" in query:
            return json.dumps(
                {
                    "results": [
                        {
                            "title": "2024年年度报告摘要",
                            "url": "https://static.cninfo.com.cn/finalpage/2025-03-20/summary.pdf",
                            "date": "2025-03-20",
                            "snippet": "分季度主要会计数据 第一季度 第二季度 第三季度 第四季度 营业收入 100 120 135 150 归属于上市公司股东的净利润 -10 -6 -2 4 经营活动产生的现金流量净额 5 8 12 15",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "results": [
                    {
                        "title": "2025年第一季度报告",
                        "url": "https://static.cninfo.com.cn/finalpage/2025-04-25/q1.pdf",
                        "date": "2025-04-25",
                        "snippet": "营业收入 165 归属于上市公司股东的净利润 6 经营活动产生的现金流量净额 18",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(trend_agent, "run_search", fake_search)
    snap = trend_agent.analyze_business_quality_with_fallbacks("000001.SZ", "测试股")
    assert snap["quarters_analyzed"] >= 2
    assert snap["financial_data_source"] == "web"
    assert "巨潮资讯搜索结果抽取" in " ".join(snap["business_quality_bullets"])


def test_load_financial_quarters_from_tushare_throttles_between_calls(monkeypatch):
    sleeps = []

    class FakePro:
        def income(self, **kwargs):
            return pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20251231"], "revenue": [100.0]})

        def cashflow(self, **kwargs):
            return pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20251231"], "n_cashflow_act": [12.0]})

        def fina_indicator(self, **kwargs):
            return pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20251231"], "grossprofit_margin": [22.0]})

    monkeypatch.setattr(trend_agent, "_get_tushare_pro_client", lambda: FakePro())
    monkeypatch.setattr(trend_agent.time, "sleep", lambda seconds: sleeps.append(seconds))

    df = trend_agent.load_financial_quarters_from_tushare("000001.SZ", max_quarters=4)
    assert not df.empty
    assert sleeps == [trend_agent.TUSHARE_CALL_INTERVAL_SEC, trend_agent.TUSHARE_CALL_INTERVAL_SEC]


def test_build_tool_feedback_nudges_web_search_for_missing_external_evidence():
    message = trend_agent._build_tool_feedback_message(
        tool="web_search",
        tool_input="测试股 最新订单",
        result="tool_error: web_search failed with RuntimeError: timeout",
        status="error",
        repeated_failure=True,
        remaining_iterations=2,
    )
    assert "web_search失败" in message
    assert "不要把缺失的外部事实当成已验证结论" in message
    assert "改用更具体的web_search查询" in message


def test_rank_candidates_for_alpha_uses_business_quality_signal():
    candidates = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "industry": "I1",
                "matched_themes": ["AI"],
                "list_type": "theme_driven",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.1,
                "valuation_quality_score": 70.0,
                "valuation_stretch_score": 30.0,
                "valuation_label": "合理",
            },
            {
                "ts_code": "000002.SZ",
                "name": "B",
                "industry": "I1",
                "matched_themes": ["AI"],
                "list_type": "theme_driven",
                "ma_spread": 0.10,
                "volume_boost": 1.8,
                "theme_strength_score": 1.0,
                "toplist_recency_score": 0.1,
                "valuation_quality_score": 70.0,
                "valuation_stretch_score": 30.0,
                "valuation_label": "合理",
            },
        ]
    )
    audits = [
        trend_agent.AuditResult(ts_code="000001.SZ", name="A", theme="AI", verdict="pass", rationale="ok", sources=[], business_quality_score=80.0, business_quality_label="强", business_quality_summary="改善", business_quality_bullets=["营收改善"], quarters_analyzed=8),
        trend_agent.AuditResult(ts_code="000002.SZ", name="B", theme="AI", verdict="pass", rationale="ok", sources=[], business_quality_score=25.0, business_quality_label="偏弱", business_quality_summary="走弱", business_quality_bullets=["现金流承压"], quarters_analyzed=8),
    ]
    signals = {
        "000001.SZ": {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8},
        "000002.SZ": {"breakout_window_ok": True, "already_breakout": False, "extended_breakout": False, "turnover_mult": 1.8},
    }
    ranked = trend_agent.rank_candidates_for_alpha(candidates, audits=audits, signals=signals, config=StrategyConfig())
    assert ranked.iloc[0]["ts_code"] == "000001.SZ"


def test_render_report_markdown_debug_includes_business_quality_section():
    report = trend_agent.ReportModel(
        title="测试研报",
        generated_at="2026-03-22 12:00:00",
        theme_overviews=[],
        core_table_rows=[],
        theme_table_rows=[],
        stock_sections=[
            trend_agent.ReportStockSection(
                ts_code="000001.SZ",
                name="测试股",
                matched_themes=["AI应用"],
                recommendation="buy",
                recommendation_label="推荐",
                research_depth="standard",
                summary="摘要",
                investment_logic=["逻辑"],
                positive_findings=[],
                growth_catalysts=[],
                technical_analysis=["技术"],
                capital_validation=["资金"],
                trade_plan=["计划"],
                risks=["风险"],
                business_quality_score=55.0,
                business_quality_label="中性",
                business_quality_summary="经营趋势平稳。",
                business_quality_bullets=["营收平稳", "现金流数据有限"],
                quarters_analyzed=6,
                source_urls=[],
            )
        ],
        risks=[],
    )
    md = trend_agent.render_report_markdown_debug(report)
    assert "12季度经营趋势与业务质量" in md
    assert "已分析季度数 6" in md
