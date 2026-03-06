import json

import pandas as pd

import trend_agent
from trend_agent import (
    AuditResult,
    GrowthCatalyst,
    PositiveFinding,
    StrategyConfig,
    ThemeItem,
)


def test_theme_cache_invalidation_by_theme_set(monkeypatch, tmp_path):
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
    cache_path = tmp_path / "qwen_cache.json"

    out_a = trend_agent.qwen_match_themes(
        themes=[ThemeItem(name="主题A", keywords=["a"], summary="", sources=[])],
        candidates=candidates,
        cache_path=cache_path,
        cache_version="2",
    )
    out_b = trend_agent.qwen_match_themes(
        themes=[ThemeItem(name="主题B", keywords=["b"], summary="", sources=[])],
        candidates=candidates,
        cache_path=cache_path,
        cache_version="2",
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

    def fake_match(themes, candidates, cache_path, cache_version="2", config=None):
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

    monkeypatch.setattr(trend_agent, "qwen_match_themes", fake_match)
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

    def fake_match(themes, candidates, cache_path, cache_version="3", config=None):
        out = candidates.copy()
        out["matched_themes"] = [[] for _ in range(len(out))]
        return out

    heuristic_calls = {"n": 0}

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        heuristic_calls["n"] += 1
        out = candidates.copy()
        out["matched_themes"] = [["化工与周期材料"]]
        return out

    monkeypatch.setattr(trend_agent, "qwen_match_themes", fake_match)
    monkeypatch.setattr(trend_agent, "heuristic_match_themes", fake_heuristic)
    cfg = StrategyConfig(toplist_exclusion_mode="penalty", theme_match_policy="conservative")
    themes = [ThemeItem(name="化工与周期材料", keywords=["化工", "玻璃"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)

    row = out[out["ts_code"] == "600059.SH"].iloc[0]
    assert heuristic_calls["n"] == 0
    assert row["matched_themes"] == []
    assert bool(row["off_theme"]) is True
    assert row["filter_tier"] == "OFF_THEME_FALLBACK"


def test_phase2_conservative_no_qwen_match_returns_off_theme(monkeypatch):
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

    def fake_match(themes, candidates, cache_path, cache_version="3", config=None):
        out = candidates.copy()
        out["matched_themes"] = [[] for _ in range(len(out))]
        return out

    heuristic_calls = {"n": 0}

    def fake_heuristic(themes, candidates, existing_col="matched_themes"):
        heuristic_calls["n"] += 1
        out = candidates.copy()
        out["matched_themes"] = [["AI"], ["AI"]]
        return out

    monkeypatch.setattr(trend_agent, "qwen_match_themes", fake_match)
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
    assert merged.count("## 【核心金股】") == 1
    assert "| A(000001.SZ) |" in merged
    assert "| B(000002.SZ) |" in merged


def test_qwen_match_guard_blocks_scope_only_false_positive(monkeypatch, tmp_path):
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
    out = trend_agent.qwen_match_themes(themes, candidates, cache_path=tmp_path / "cache.json", cache_version="3")
    assert out.iloc[0]["matched_themes"] == []


def test_qwen_match_guard_keeps_valid_chemical_match(monkeypatch, tmp_path):
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
    out = trend_agent.qwen_match_themes(themes, candidates, cache_path=tmp_path / "cache.json", cache_version="3")
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


def test_qwen_match_retries_then_succeeds(monkeypatch, tmp_path):
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
        qwen_rate_limit_max_retries=6,
        qwen_rate_limit_base_delay_sec=0.01,
        qwen_rate_limit_max_delay_sec=0.05,
        qwen_request_interval_sec=0.0,
    )
    out = trend_agent.qwen_match_themes(
        themes,
        candidates,
        cache_path=tmp_path / "cache.json",
        cache_version="3",
        config=cfg,
    )
    assert call_count["n"] == 3
    assert out.iloc[0]["matched_themes"] == ["AI应用"]


def test_qwen_match_exhausted_rate_limit_degrades_to_empty(monkeypatch, tmp_path):
    def fake_sleep(_seconds):
        return None

    def fake_invoke(provider, messages, temperature=0.1):
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
        qwen_rate_limit_max_retries=1,
        qwen_rate_limit_base_delay_sec=0.01,
        qwen_rate_limit_max_delay_sec=0.05,
        qwen_request_interval_sec=0.0,
    )
    cache_path = tmp_path / "cache.json"
    out = trend_agent.qwen_match_themes(
        themes,
        candidates,
        cache_path=cache_path,
        cache_version="3",
        config=cfg,
    )
    assert out.iloc[0]["matched_themes"] == []
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert any(v.get("note") == "rate_limited_exhausted" for v in cache_payload.values() if isinstance(v, dict))


def test_qwen_match_respects_configured_batch_size(monkeypatch, tmp_path):
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
    cfg = StrategyConfig(qwen_batch_size=2, qwen_request_interval_sec=0.0)
    out = trend_agent.qwen_match_themes(
        themes,
        candidates,
        cache_path=tmp_path / "cache.json",
        cache_version="3",
        config=cfg,
    )
    assert call_count["n"] == 3
    assert out["matched_themes"].apply(lambda x: x == ["AI应用"]).all()


def test_phase2_quant_filter_survives_qwen_rate_limit(monkeypatch):
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
        qwen_batch_size=1,
        qwen_rate_limit_max_retries=1,
        qwen_rate_limit_base_delay_sec=0.01,
        qwen_rate_limit_max_delay_sec=0.05,
        qwen_request_interval_sec=0.0,
    )
    themes = [ThemeItem(name="AI", keywords=["算力"], summary="", sources=[], validation_status="confirmed")]
    out = trend_agent.phase2_quant_filter(themes, config=cfg)
    assert not out.empty
    assert "matched_themes" in out.columns
