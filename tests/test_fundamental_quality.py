"""
Unit tests for fundamental_quality.py — lifecycle classifier and dimension scorers.

Covers:
- Pre-revenue biotech (growth/pre_revenue): not penalized for losses
- Mature consumer (mature/steady): quality_score >= 65 with adequate data
- Declining cyclical: low quality_score
- All-NaN data → unknown stage, neutral 50.0
- Edge: exactly at profitability threshold → transitional
- Edge: single quarter data → unknown
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fundamental_quality import (
    classify_lifecycle,
    compute_fundamental_quality,
    _quarterly_cagr,
    _trend_delta,
    _acceleration,
    _volatility,
    _consecutive_positive_quarters,
    _gaussian,
    _linear,
)


# ── Helper unit tests ──────────────────────────────────────────────

def test_quarterly_cagr_growing():
    """Moderate-growing revenue with log-linear regression."""
    series = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0])
    result = _quarterly_cagr(series)
    assert result is not None
    assert result > 0.10  # rough annualized CAGR


def test_quarterly_cagr_nan_input():
    series = pd.Series([np.nan, np.nan, np.nan])
    assert _quarterly_cagr(series) is None


def test_quarterly_cagr_short():
    assert _quarterly_cagr(pd.Series([100.0])) is None


def test_trend_delta_positive():
    series = pd.Series([-35.0, -32.0, -28.0, -24.0, -20.0, -16.0, -10.0, -5.0])
    result = _trend_delta(series)
    assert result is not None
    assert result > 0.4


def test_trend_delta_negative():
    series = pd.Series([20.0, 18.0, 15.0, 12.0, 8.0, 4.0, -2.0, -8.0])
    result = _trend_delta(series)
    assert result is not None
    assert result < -0.4


def test_trend_delta_flat():
    series = pd.Series([10.0, 10.0, 10.0, 10.0])
    result = _trend_delta(series)
    assert result is not None
    assert abs(result) < 0.1


def test_trend_delta_nan():
    # All-NaN with 2+ valid returns None
    assert _trend_delta(pd.Series([np.nan, np.nan])) is None


def test_acceleration_positive():
    # Strong acceleration needs 6+ points
    series = pd.Series([100.0, 102.0, 105.0, 112.0, 125.0, 142.0, 165.0, 195.0])
    result = _acceleration(series)
    assert result is not None
    assert result > 0


def test_acceleration_short_data():
    # Less than 6 points → None
    series = pd.Series([100.0, 115.0, 120.0, 121.0])
    assert _acceleration(series) is None


def test_volatility():
    series = pd.Series([100.0, 110.0, 90.0, 105.0])
    result = _volatility(series)
    assert result is not None
    assert 0.05 < result < 0.2


def test_consecutive_positive_quarters():
    series = pd.Series([-5.0, -2.0, 1.0, 3.0, 5.0, 7.0, 8.0, 10.0])
    assert _consecutive_positive_quarters(series) == 6


def test_consecutive_positive_quarters_none():
    series = pd.Series([-5.0, -2.0, -1.0, -3.0])
    assert _consecutive_positive_quarters(series) == 0


def test_gaussian_peak():
    # _gaussian(x, optimal, width): returns 0-1, peak at x=optimal
    result = _gaussian(0.0, optimal=0.0, width=1.0)
    assert 0.99 <= result <= 1.01


def test_gaussian_far():
    result = _gaussian(3.0, optimal=0.0, width=1.0)
    assert result < 0.05


def test_linear_midpoint():
    result = _linear(0.5, low=0.0, high=1.0, max_score=100.0)
    assert 49 <= result <= 51


def test_linear_invert():
    result_lo = _linear(0.0, low=0.0, high=1.0, max_score=100.0, invert=True)
    assert 99 <= result_lo <= 101


# ── Lifecycle classifier tests ─────────────────────────────────────

def test_classify_pre_revenue_biotech():
    """Pre-revenue biotech: growing revenue, still unprofitable, improving cash."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [1.0, 1.5, 2.0, 2.5, 3.0, 3.8, 4.5, 5.0],
        "gross_margin": [60.0, 62.0, 61.0, 63.0, 65.0, 64.0, 66.0, 67.0],
        "net_income": [-50, -48, -45, -42, -38, -35, -30, -28],
        "n_cashflow_act": [-40, -38, -35, -30, -25, -20, -15, -12],
    })
    result = classify_lifecycle(df, "医药生物")
    assert result["stage"] in ("growth", "transitional")


def test_classify_mature_consumer():
    """Mature consumer: steady revenue, profitable, stable cashflow."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [200, 202, 204, 206, 208, 210, 212, 214],
        "gross_margin": [45.0, 45.2, 45.5, 45.8, 46.0, 46.2, 46.5, 46.8],
        "net_income": [20, 21, 22, 23, 24, 25, 26, 27],
        "n_cashflow_act": [25, 26, 27, 28, 29, 30, 31, 32],
    })
    result = classify_lifecycle(df, "食品饮料")
    assert result["stage"] in ("mature", "growth")


def test_classify_declining_cyclical():
    """Declining cyclical: shrinking revenue, volatile margins, cash turning negative."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [180, 176, 170, 166, 155, 145, 132, 120],
        "gross_margin": [28.0, 27.5, 27.0, 26.0, 24.5, 23.0, 21.5, 20.0],
        "n_cashflow_act": [30, 28, 24, 20, 12, 4, -6, -15],
        "net_income": [20, 18, 15, 12, 8, 4, -2, -8],
    })
    result = classify_lifecycle(df, "钢铁")
    assert result["stage"] in ("declining", "cyclical", "transitional")


def test_classify_all_nan_is_unknown():
    """All-NaN DataFrame should return unknown stage, not crash."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=4, freq="QE"),
        "revenue": [np.nan, np.nan, np.nan, np.nan],
        "gross_margin": [np.nan, np.nan, np.nan, np.nan],
        "net_income": [np.nan, np.nan, np.nan, np.nan],
        "n_cashflow_act": [np.nan, np.nan, np.nan, np.nan],
    })
    result = classify_lifecycle(df, "")
    assert result["stage"] == "unknown"


def test_classify_single_quarter_is_unknown():
    """Single quarter data: not enough to classify."""
    df = pd.DataFrame({
        "end_date": pd.to_datetime(["2023-03-31"]),
        "revenue": [100.0],
        "gross_margin": [20.0],
        "net_income": [5.0],
        "n_cashflow_act": [3.0],
    })
    result = classify_lifecycle(df, "")
    assert result["stage"] == "unknown"


def test_classify_edge_profitability_threshold():
    """Company crossing from loss to profit: should be transitional."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [50, 52, 54, 56, 58, 60, 62, 64],
        "gross_margin": [20.0] * 8,
        "net_income": [-3, -2, -1, 0, 0, 1, 2, 3],
        "n_cashflow_act": [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5],
    })
    result = classify_lifecycle(df, "")
    assert result["stage"] in ("transitional", "growth")
    assert result["confidence"] > 0.5


# ── Dimension scorer tests ─────────────────────────────────────────

def test_compute_quality_pre_revenue_not_penalized():
    """Pre-revenue biotech should get reasonable score despite losses."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [1.0, 1.5, 2.0, 2.5, 3.0, 3.8, 4.5, 5.0],
        "gross_margin": [60.0, 62.0, 61.0, 63.0, 65.0, 64.0, 66.0, 67.0],
        "net_income": [-50, -48, -45, -42, -38, -35, -30, -28],
        "n_cashflow_act": [-40, -38, -35, -30, -25, -20, -15, -12],
    })
    lifecycle = classify_lifecycle(df, "医药生物")
    result = compute_fundamental_quality(df, lifecycle)
    assert result["fundamental_quality_score"] >= 55
    assert result["label"] == "成长关注"
    # Growth quality should be high due to fast revenue growth
    assert result["dimension_scores"]["growth_quality"] >= 70


def test_compute_quality_mature_high_score():
    """Mature consumer with ROE data should score well."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [200, 202, 204, 206, 208, 210, 212, 214],
        "gross_margin": [45.0, 45.2, 45.5, 45.8, 46.0, 46.2, 46.5, 46.8],
        "net_income": [20, 21, 22, 23, 24, 25, 26, 27],
        "n_cashflow_act": [25, 26, 27, 28, 29, 30, 31, 32],
        "roe": [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5],
        "total_assets": [500, 510, 520, 530, 540, 550, 560, 570],
        "total_hldr_eqy": [200, 205, 210, 215, 220, 225, 230, 235],
    })
    lifecycle = classify_lifecycle(df, "食品饮料")
    result = compute_fundamental_quality(df, lifecycle)
    assert result["fundamental_quality_score"] >= 55
    assert result["dimension_scores"]["profitability"] >= 40


def test_compute_quality_declining_low_score():
    """Declining cyclical should score low."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [180, 176, 170, 166, 155, 145, 132, 120],
        "gross_margin": [28.0, 27.5, 27.0, 26.0, 24.5, 23.0, 21.5, 20.0],
        "n_cashflow_act": [30, 28, 24, 20, 12, 4, -6, -15],
        "net_income": [20, 18, 15, 12, 8, 4, -2, -8],
    })
    lifecycle = classify_lifecycle(df, "钢铁")
    result = compute_fundamental_quality(df, lifecycle)
    assert result["fundamental_quality_score"] < 55


def test_compute_quality_all_nan_neutral():
    """All-NaN data: unknown stage, neutral 50.0."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=4, freq="QE"),
        "revenue": [np.nan, np.nan, np.nan, np.nan],
        "gross_margin": [np.nan, np.nan, np.nan, np.nan],
        "net_income": [np.nan, np.nan, np.nan, np.nan],
        "n_cashflow_act": [np.nan, np.nan, np.nan, np.nan],
    })
    lifecycle = classify_lifecycle(df, "")
    result = compute_fundamental_quality(df, lifecycle)
    assert result["fundamental_quality_score"] == 50.0
    assert result["label"] == "中性"


def test_compute_quality_returns_all_keys():
    """All expected keys are present in result."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=4, freq="QE"),
        "revenue": [100, 105, 108, 112],
        "gross_margin": [20.0, 20.5, 21.0, 21.5],
        "net_income": [5, 6, 7, 8],
        "n_cashflow_act": [3, 4, 5, 6],
    })
    lifecycle = classify_lifecycle(df, "")
    result = compute_fundamental_quality(df, lifecycle)
    assert "fundamental_quality_score" in result
    assert "label" in result
    assert "summary" in result
    assert "bullets" in result
    assert "dimension_scores" in result
    assert set(result["dimension_scores"].keys()) == {
        "profitability", "growth_quality", "financial_health", "earnings_quality",
    }
    assert 0 <= result["fundamental_quality_score"] <= 100
    assert all(0 <= v <= 100 for v in result["dimension_scores"].values())


def test_bullets_not_empty():
    """Bullets should be non-empty for valid data."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=4, freq="QE"),
        "revenue": [100, 105, 108, 112],
        "gross_margin": [20.0, 20.5, 21.0, 21.5],
        "net_income": [5, 6, 7, 8],
        "n_cashflow_act": [3, 4, 5, 6],
    })
    lifecycle = classify_lifecycle(df, "电子")
    result = compute_fundamental_quality(df, lifecycle)
    assert len(result["bullets"]) >= 2


def test_label_semantics_mature_high_quality():
    """High-quality mature company gets a respectable label."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [500, 510, 520, 530, 540, 550, 560, 570],
        "gross_margin": [55.0, 55.5, 56.0, 56.5, 57.0, 57.5, 58.0, 58.5],
        "net_income": [80, 82, 84, 86, 88, 90, 92, 94],
        "n_cashflow_act": [90, 91, 92, 93, 94, 95, 96, 97],
        "roe": [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5],
        "total_assets": [800, 820, 840, 860, 880, 900, 920, 940],
        "total_hldr_eqy": [500, 510, 520, 530, 540, 550, 560, 570],
    })
    lifecycle = classify_lifecycle(df, "食品饮料")
    result = compute_fundamental_quality(df, lifecycle)
    assert result["label"] in ("优质", "稳健", "成长关注", "关注")


def test_summary_contains_lifecycle_info():
    """Summary should mention the lifecycle stage in Chinese."""
    df = pd.DataFrame({
        "end_date": pd.date_range("2023-03-31", periods=8, freq="QE"),
        "revenue": [500, 510, 520, 530, 540, 550, 560, 570],
        "gross_margin": [55.0, 55.5, 56.0, 56.5, 57.0, 57.5, 58.0, 58.5],
        "net_income": [80, 82, 84, 86, 88, 90, 92, 94],
        "n_cashflow_act": [90, 91, 92, 93, 94, 95, 96, 97],
    })
    lifecycle = classify_lifecycle(df, "食品饮料")
    result = compute_fundamental_quality(df, lifecycle)
    # Summary should contain a bracket-tagged lifecycle indicator
    assert "【" in result["summary"] and "】" in result["summary"]
