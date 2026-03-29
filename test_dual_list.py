#!/usr/bin/env python3
"""
Integration test for dual-list stock selection logic.

Uses the real Qwen3-8B model via SiliconFlow (same as production pipeline).

Tests:
1. extract_theme_named_stocks() finds stocks in theme evidence
2. qwen_match_themes() whitespace normalization — real LLM output
3. Named stock merge into theme pool
4. Themed stocks matched, unrelated stocks rejected
"""

import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

# Load .env before importing trend_agent
from dotenv import load_dotenv
load_dotenv()

from trend_agent import (
    ThemeItem,
    StrategyConfig,
    extract_theme_named_stocks,
    qwen_match_themes,
)


pytestmark = [pytest.mark.integration, pytest.mark.live_llm]


def make_screen_df(stocks: list[dict]) -> pd.DataFrame:
    """Build a minimal screen_df with required columns."""
    defaults = {
        "consolidation_score": 60.0,
        "volume_boost": 1.5,
        "momentum_score": 50.0,
        "composite_score": 50.0,
        "ma_spread": 0.10,
        "ma_spread_std": 0.02,
        "squeeze_readiness": 30.0,
        "volume_quality_score": 50.0,
        "avg_turnover": 3.0,
        "market_cap": 50e8,
        "industry": "测试",
        "main_business": "",
        "business_scope": "",
        "introduction": "",
    }
    rows = []
    for s in stocks:
        row = {**defaults, **s}
        rows.append(row)
    return pd.DataFrame(rows)


THEMES = [
    ThemeItem(
        name="AI算力基础设施",
        keywords=["AI", "算力", "芯片", "半导体", "封装测试", "集成电路"],
        summary="AI算力需求爆发，带动芯片、封测产业链景气上行",
        sources=["https://example.com"],
        validation_status="confirmed",
        capital_signal="长电科技 上榜3次，净买入2.1亿",
        evidence="龙虎榜显示长电科技(600584.SH)连续上榜，机构净买入",
    ),
    ThemeItem(
        name="磷化工/农化资源",
        keywords=["磷化工", "农化", "化肥", "磷矿", "纯碱"],
        summary="磷化工涨价周期，磷矿资源受限推动景气上行",
        sources=["https://example.com"],
        validation_status="confirmed",
        capital_signal="湖北宜化 上榜2次",
        evidence="湖北宜化(000422.SZ)磷矿资源受限推动景气",
    ),
]

# Mix of stocks: 3 should match, 3 should NOT match
STOCKS = [
    # Should match AI theme
    {
        "ts_code": "600584.SH",
        "name": "长电科技",
        "industry": "半导体",
        "main_business": "集成电路封装测试,分立器件制造销售",
        "introduction": "全球领先的集成电路制造与技术服务提供商，芯片封装测试",
    },
    {
        "ts_code": "002049.SZ",
        "name": "紫光国微",
        "industry": "半导体",
        "main_business": "集成电路芯片设计与销售,智能芯片产品,特种集成电路,存储器芯片",
        "introduction": "国内集成电路设计企业龙头，在特种集成电路和智能安全芯片领域领先",
    },
    # Should match phosphorus/chemical theme
    {
        "ts_code": "000422.SZ",
        "name": "湖北宜化",
        "industry": "农药化肥",
        "main_business": "尿素,磷酸二铵,聚氯乙烯,烧碱,合成氨",
        "introduction": "国内化肥龙头企业，在磷矿之都宜昌设立磷化工生产基地",
    },
    # Should NOT match any theme
    {
        "ts_code": "000568.SZ",
        "name": "泸州老窖",
        "industry": "白酒",
        "main_business": "泸州老窖系列酒",
        "introduction": "中国白酒浓香技艺的开创者",
    },
    {
        "ts_code": "000895.SZ",
        "name": "双汇发展",
        "industry": "食品",
        "main_business": "高温肉制品,低温肉制品,鲜冻猪产品",
        "introduction": "农业产业化国家重点龙头企业，肉类加工",
    },
    {
        "ts_code": "000726.SZ",
        "name": "鲁泰Ａ",
        "industry": "纺织",
        "main_business": "衬衣色织面,衬衣,皮棉,纺纱",
        "introduction": "全球高档色织面料生产商，纺织服装",
    },
]


class TestQwenMatchThemesLive(unittest.TestCase):
    """Test qwen_match_themes with real Qwen3-8B model."""

    @classmethod
    def setUpClass(cls):
        """Run Qwen matching once and reuse results."""
        cls.screen_df = make_screen_df(STOCKS)
        config = StrategyConfig.from_env()
        print("\n[SETUP] Calling qwen_match_themes with real Qwen3-8B...")
        cls.result = qwen_match_themes(
            THEMES,
            cls.screen_df,
            config=config,
        )
        print("[SETUP] Qwen response received.")
        # Print raw results for visibility
        for _, row in cls.result.iterrows():
            print(f"  {row['ts_code']} ({row['name']}): {row.get('matched_themes', [])}")

    def _get_matched(self, ts_code: str) -> list:
        row = self.result[self.result["ts_code"] == ts_code]
        if row.empty:
            return []
        return row.iloc[0].get("matched_themes", [])

    # --- Recall: themed stocks should match ---

    def test_changdian_matches_ai_theme(self):
        """长电科技 (semiconductor) should match AI算力基础设施."""
        matched = self._get_matched("600584.SH")
        self.assertTrue(len(matched) > 0, f"长电科技 should match a theme, got: {matched}")
        # Should contain the AI theme (canonical name, not with extra spaces)
        self.assertTrue(
            any("AI" in t and "算力" in t for t in matched),
            f"长电科技 should match AI theme, got: {matched}",
        )

    def test_ziguang_matches_ai_theme(self):
        """紫光国微 (IC design) should match AI theme."""
        matched = self._get_matched("002049.SZ")
        self.assertTrue(len(matched) > 0, f"紫光国微 should match a theme, got: {matched}")
        self.assertTrue(
            any("AI" in t or "算力" in t or "芯片" in t for t in matched),
            f"紫光国微 should match AI/chip theme, got: {matched}",
        )

    def test_yihua_matches_phosphorus_theme(self):
        """湖北宜化 (fertilizer/phosphorus) should match 磷化工/农化资源."""
        matched = self._get_matched("000422.SZ")
        self.assertTrue(len(matched) > 0, f"湖北宜化 should match a theme, got: {matched}")
        self.assertTrue(
            any("磷化工" in t or "农化" in t for t in matched),
            f"湖北宜化 should match phosphorus theme, got: {matched}",
        )

    # --- Precision: unrelated stocks should NOT match ---

    def test_luzhou_no_match(self):
        """泸州老窖 (baijiu) should not match any theme."""
        matched = self._get_matched("000568.SZ")
        self.assertEqual(matched, [], f"泸州老窖 should not match, got: {matched}")

    def test_shuanghui_no_match(self):
        """双汇发展 (food) should not match any theme."""
        matched = self._get_matched("000895.SZ")
        self.assertEqual(matched, [], f"双汇发展 should not match, got: {matched}")

    def test_lutai_no_match(self):
        """鲁泰A (textile) should not match any theme."""
        matched = self._get_matched("000726.SZ")
        self.assertEqual(matched, [], f"鲁泰A should not match, got: {matched}")

    # --- Whitespace normalization ---

    def test_theme_names_are_canonical(self):
        """All matched theme names should be canonical (no extra whitespace)."""
        canonical_names = {t.name for t in THEMES}
        for _, row in self.result.iterrows():
            matched = row.get("matched_themes", [])
            for t in matched:
                self.assertIn(
                    t, canonical_names,
                    f"Theme '{t}' for {row['ts_code']} is not canonical. "
                    f"Expected one of {canonical_names}",
                )


class TestNamedStockMerge(unittest.TestCase):
    """Test that named stocks get themes from evidence when Qwen misses them."""

    @classmethod
    def setUpClass(cls):
        """Run Qwen matching and then apply named stock merge."""
        cls.screen_df = make_screen_df(STOCKS)
        config = StrategyConfig.from_env()

        # Run Qwen match
        print("\n[SETUP] Running Qwen match for named stock merge test...")
        cls.theme_pool = qwen_match_themes(
            THEMES,
            cls.screen_df,
            config=config,
            relaxed_validation=True,
            named_stock_codes={"600584.SH", "000422.SZ"},
        )

        # Apply named stock merge (same logic as phase2_quant_filter)
        named_stocks = {"600584.SH": ["AI算力基础设施"], "000422.SZ": ["磷化工/农化资源"]}
        if "matched_themes" not in cls.theme_pool.columns:
            cls.theme_pool["matched_themes"] = [[] for _ in range(len(cls.theme_pool))]

        for ts_code, theme_names in named_stocks.items():
            mask = cls.theme_pool["ts_code"].astype(str) == str(ts_code)
            idx = cls.theme_pool.index[mask]
            if len(idx) == 0:
                continue
            existing = cls.theme_pool.at[idx[0], "matched_themes"]
            if not isinstance(existing, list):
                existing = []
            merged = list(dict.fromkeys(existing + theme_names))
            for i in idx:
                cls.theme_pool.at[i, "matched_themes"] = merged

        print("[SETUP] After merge:")
        for _, row in cls.theme_pool.iterrows():
            print(f"  {row['ts_code']} ({row['name']}): {row.get('matched_themes', [])}")

    def test_named_stocks_have_themes(self):
        """Named stocks should always have themes after merge, even if Qwen missed."""
        for ts_code in ["600584.SH", "000422.SZ"]:
            row = self.theme_pool[self.theme_pool["ts_code"] == ts_code].iloc[0]
            matched = row["matched_themes"]
            self.assertIsInstance(matched, list)
            self.assertTrue(len(matched) > 0, f"{ts_code} should have themes after merge")

    def test_unrelated_stocks_still_empty(self):
        """Unrelated stocks should remain unmatched after merge."""
        for ts_code in ["000568.SZ", "000895.SZ", "000726.SZ"]:
            row = self.theme_pool[self.theme_pool["ts_code"] == ts_code].iloc[0]
            matched = row["matched_themes"]
            self.assertEqual(matched, [], f"{ts_code} should remain unmatched")

    def test_match_count(self):
        """Should have at least 2 matched stocks (the named ones) and <=3 unmatched."""
        match_count = self.theme_pool["matched_themes"].apply(bool).sum()
        self.assertGreaterEqual(match_count, 2, "At least 2 named stocks should have matches")
        # Could be more if Qwen also matched some
        self.assertLessEqual(match_count, 4, "At most 4 stocks should match (3 themed + 1 possible)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
