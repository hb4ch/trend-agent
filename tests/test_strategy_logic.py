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
    out_a = trend_agent.qwen_match_themes(
        themes=[ThemeItem(name="主题A", keywords=["a"], summary="", sources=[])],
        candidates=candidates,
    )
    out_b = trend_agent.qwen_match_themes(
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
                    "source_urls": ["https://example.com/report"],
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
                    "source_urls": ["https://example.com/report"],
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
                    "source_urls": [],
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
                    "source_urls": [],
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
                    "source_urls": [],
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
                    "source_urls": [],
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
                    "source_urls": [],
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
    out = trend_agent.qwen_match_themes(themes, candidates)
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
    out = trend_agent.qwen_match_themes(themes, candidates)
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
        config=cfg,
    )
    assert call_count["n"] == 3
    assert out.iloc[0]["matched_themes"] == ["AI应用"]


def test_qwen_match_exhausted_rate_limit_degrades_to_empty(monkeypatch, tmp_path):
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
        qwen_rate_limit_max_retries=1,
        qwen_rate_limit_base_delay_sec=0.01,
        qwen_rate_limit_max_delay_sec=0.05,
        qwen_request_interval_sec=0.0,
    )
    out = trend_agent.qwen_match_themes(
        themes,
        candidates,
        config=cfg,
    )
    assert call_count["n"] == 2
    assert out.iloc[0]["matched_themes"] == []


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
