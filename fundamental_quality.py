"""
Lifecycle-aware fundamental quality scoring for Contrarian Agent.

Separates "is this a good business?" from "is it currently profitable?"
by routing growth-stage companies through a different scoring rubric
that rewards trajectory and sustainability, not current earnings level.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Lifecycle classification ──────────────────────────────────────────────

def _sector_from_industry(industry: str) -> str:
    """Map Chinese A-share industry name to sector bucket."""
    ind = industry.lower()
    if any(k in ind for k in ["电子", "计算机", "软件", "通信", "半导体", "芯片", "互联网", "人工智能"]):
        return "tech"
    if any(k in ind for k in ["医药", "生物", "制药", "医疗", "疫苗", "基因"]):
        return "biotech"
    if any(k in ind for k in ["银行", "保险", "证券", "金融", "信托"]):
        return "financial"
    if any(k in ind for k in ["食品", "饮料", "白酒", "家电", "服装", "零售", "消费品"]):
        return "consumer"
    if any(k in ind for k in ["钢铁", "煤炭", "有色", "化工", "石油", "水泥"]):
        return "cyclical"
    if any(k in ind for k in ["电力", "水务", "燃气", "环保", "新能源"]):
        return "defensive"
    if any(k in ind for k in ["汽车", "机械", "军工", "航空", "船舶", "电气", "设备"]):
        return "industrial"
    return "other"


def _pick_metric(df: pd.DataFrame, aliases: List[str]) -> pd.Series:
    """Pick the first available numeric column among aliases."""
    for col in aliases:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= 2:
                return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def classify_lifecycle(
    df: pd.DataFrame,
    industry: str = "",
) -> Dict[str, Any]:
    """Classify a stock's lifecycle stage from quarterly financial data.

    Returns dict with keys: stage, subflavor, confidence, signals
    Degrades gracefully: when data is sparse, confidence drops and
    classification relies more on industry priors.
    """
    sector = _sector_from_industry(industry) if industry else "other"
    quarter_count = int(len(df)) if df is not None else 0

    if df is None or df.empty or quarter_count < 2:
        return {
            "stage": "unknown",
            "subflavor": "",
            "confidence": 0.0,
            "signals": {"quarter_count": quarter_count, "sector": sector},
        }

    # Extract metrics
    revenue = _pick_metric(df, ["revenue", "total_revenue", "total_operat_income",
                                  "operating_revenue", "operate_income"])
    net_income = _pick_metric(df, ["net_income", "n_income", "n_income_attr_p",
                                     "profit_to_gr", "netprofit"])
    gross_margin = _pick_metric(df, ["gross_margin", "grossprofit_margin",
                                       "gross_margin_rate", "gp_margin"])
    op_cashflow = _pick_metric(df, ["operate_cashflow", "n_cashflow_act",
                                      "net_cash_flows_oper_act", "oper_cash_flow"])

    signals: Dict[str, Any] = {
        "quarter_count": quarter_count,
        "sector": sector,
    }

    # Compute revenue CAGR (annualized from quarterly trend)
    rev_cagr = _quarterly_cagr(revenue)
    signals["revenue_cagr_8q"] = rev_cagr

    # Profit age: consecutive quarters with positive net income
    profit_age = _consecutive_positive_quarters(net_income)
    signals["profit_age_quarters"] = profit_age

    # Cash profile
    cash_trend = _trend_delta(op_cashflow)
    signals["cash_trend"] = cash_trend

    # Net margin / loss signals
    ni_recent = net_income.dropna().tail(4)
    is_currently_profitable = bool((ni_recent > 0).any()) if len(ni_recent) > 0 else None
    signals["is_currently_profitable"] = is_currently_profitable

    ni_trend = _trend_delta(net_income)
    signals["net_income_trend"] = ni_trend

    # Gross margin trend
    gm_trend = _trend_delta(gross_margin)
    signals["gross_margin_trend"] = gm_trend

    # Revenue volatility (coefficient of variation over quarters)
    rev_vol = _volatility(revenue)
    signals["revenue_volatility"] = rev_vol

    # If all key metrics are unavailable, bail early
    if rev_cagr is None and ni_trend is None and cash_trend is None:
        return {
            "stage": "unknown",
            "subflavor": "",
            "confidence": max(0.1, quarter_count / 24.0),
            "signals": signals,
        }

    # ── Decision tree ──
    confidence = min(1.0, quarter_count / 12.0)

    # Growth signals
    is_hypergrowth = rev_cagr is not None and rev_cagr > 0.25
    is_expansion = rev_cagr is not None and 0.10 < rev_cagr <= 0.25
    is_early_stage = rev_cagr is not None and rev_cagr > 0.05 and profit_age < 8

    # Pre-revenue / pre-profit
    is_pre_profit = (
        is_currently_profitable is not True
        or (profit_age < 4 and ni_recent.sum() <= 0 if len(ni_recent) > 0 else False)
    )
    is_revenue_early = revenue.notna().sum() >= 2 and revenue.dropna().max() < 1e8  # < 100M
    # Only use revenue_early if we have >= 3 data points
    if revenue.notna().sum() < 3:
        is_revenue_early = False

    # Maturity signals
    is_steady = rev_cagr is not None and 0.03 < rev_cagr <= 0.15 and profit_age >= 8
    is_cash_cow = (
        rev_cagr is not None and rev_cagr <= 0.05
        and profit_age >= 4
        and is_currently_profitable is True
    )

    # Decline/cyclical signals
    is_declining = rev_cagr is not None and rev_cagr < -0.03
    is_volatile = rev_vol is not None and rev_vol > 0.30

    # ── Classification ──
    if is_hypergrowth and is_pre_profit:
        stage, subflavor = "growth", "hypergrowth"
        confidence = min(1.0, confidence + 0.1)
    elif is_revenue_early and is_pre_profit:
        stage, subflavor = "growth", "pre_revenue"
        confidence = min(1.0, confidence + 0.05)
    elif (is_expansion or is_early_stage) and profit_age < 8:
        stage, subflavor = "growth", "expansion"
        confidence = min(1.0, confidence + 0.05)
    elif is_cash_cow:
        stage, subflavor = "mature", "cash_cow"
        confidence = min(1.0, confidence + 0.05)
    elif is_steady:
        stage, subflavor = "mature", "steady"
    elif is_declining and is_volatile:
        stage, subflavor = "cyclical", ""
    elif is_declining:
        stage, subflavor = "declining", ""
    elif is_volatile:
        stage, subflavor = "cyclical", ""
    else:
        stage, subflavor = "transitional", ""

    # Industry prior override for low-confidence cases
    if quarter_count < 4 and sector in ("biotech", "tech"):
        if not is_currently_profitable or is_currently_profitable is None:
            stage, subflavor = "growth", "pre_revenue" if is_revenue_early else "expansion"
            confidence = max(0.2, confidence * 0.7)
    elif quarter_count < 4 and sector == "financial":
        stage, subflavor = "mature", "steady"
        confidence = max(0.2, confidence * 0.7)

    return {
        "stage": stage,
        "subflavor": subflavor,
        "confidence": round(confidence, 2),
        "signals": signals,
    }


# ── Dimension scoring ─────────────────────────────────────────────────────

def compute_fundamental_quality(
    df: pd.DataFrame,
    lifecycle: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute lifecycle-aware fundamental quality score (0-100).

    Returns dict with:
      - fundamental_quality_score: 0-100
      - dimension_scores: {profitability, growth_quality, financial_health, earnings_quality}
      - label: Chinese label
      - bullets: Chinese bullet points
      - summary: one-line Chinese summary
      - data_completeness: 0.0-1.0
    """
    stage = lifecycle.get("stage", "unknown")

    # Extract all available metrics
    revenue = _pick_metric(df, ["revenue", "total_revenue", "total_operat_income",
                                  "operating_revenue", "operate_income"])
    net_income = _pick_metric(df, ["net_income", "n_income", "n_income_attr_p",
                                     "profit_to_gr", "netprofit"])
    gross_margin = _pick_metric(df, ["gross_margin", "grossprofit_margin",
                                       "gross_margin_rate", "gp_margin"])
    op_cashflow = _pick_metric(df, ["operate_cashflow", "n_cashflow_act",
                                      "net_cash_flows_oper_act", "oper_cash_flow"])
    roe = _pick_metric(df, ["roe", "roe_dt"])
    debt_ratio = _pick_metric(df, ["debt_to_assets"])
    current_ratio = _pick_metric(df, ["current_ratio"])
    total_assets = _pick_metric(df, ["total_assets"])
    total_liab = _pick_metric(df, ["total_liab", "total_liability"])
    equity = _pick_metric(df, ["total_hldr_eqy", "total_hldr_eqy_exc_min_int",
                                 "total_equity"])

    # Compute derived metrics when direct columns are missing
    if debt_ratio.isna().all() and not total_liab.isna().all() and not total_assets.isna().all():
        debt_ratio = (total_liab / total_assets.replace(0, np.nan)) * 100.0
    if roe.isna().all() and not net_income.isna().all() and not equity.isna().all():
        roe = (net_income / equity.replace(0, np.nan)) * 100.0

    # Data completeness
    metric_count = sum(1 for s in [revenue, net_income, gross_margin, op_cashflow,
                                    roe, debt_ratio, current_ratio]
                       if s.notna().sum() >= 2)
    data_completeness = min(1.0, metric_count / 5.0)

    # Score each dimension
    if stage == "growth":
        profit_score = _score_profitability_growth(lifecycle, net_income, gross_margin)
        growth_score = _score_growth_quality_growth(df, lifecycle, revenue, net_income, op_cashflow)
        health_score = _score_financial_health_growth(lifecycle, op_cashflow, debt_ratio, total_assets)
        earnings_score = _score_earnings_quality_universal(net_income, op_cashflow, total_assets)
        weights = {"profitability": 0.05, "growth_quality": 0.50,
                   "financial_health": 0.30, "earnings_quality": 0.15}
    elif stage == "mature":
        profit_score = _score_profitability_mature(roe, gross_margin, net_income)
        growth_score = _score_growth_quality_mature(revenue)
        health_score = _score_financial_health_mature(debt_ratio, current_ratio, op_cashflow)
        earnings_score = _score_earnings_quality_universal(net_income, op_cashflow, total_assets)
        weights = {"profitability": 0.30, "growth_quality": 0.20,
                   "financial_health": 0.30, "earnings_quality": 0.20}
    elif stage in ("cyclical", "declining"):
        profit_score = _score_profitability_mature(roe, gross_margin, net_income)
        growth_score = _score_growth_quality_mature(revenue)
        health_score = _score_financial_health_mature(debt_ratio, current_ratio, op_cashflow)
        earnings_score = _score_earnings_quality_universal(net_income, op_cashflow, total_assets)
        weights = {"profitability": 0.15, "growth_quality": 0.15,
                   "financial_health": 0.40, "earnings_quality": 0.30}
    else:  # transitional, unknown
        profit_score = _score_profitability_mature(roe, gross_margin, net_income)
        growth_score = _score_growth_quality_mature(revenue)
        health_score = _score_financial_health_mature(debt_ratio, current_ratio, op_cashflow)
        earnings_score = _score_earnings_quality_universal(net_income, op_cashflow, total_assets)
        weights = {"profitability": 0.25, "growth_quality": 0.25,
                   "financial_health": 0.25, "earnings_quality": 0.25}

    dimension_scores = {
        "profitability": round(profit_score, 1),
        "growth_quality": round(growth_score, 1),
        "financial_health": round(health_score, 1),
        "earnings_quality": round(earnings_score, 1),
    }

    weighted = sum(w * dimension_scores[k] for k, w in weights.items())
    score = max(0.0, min(100.0, weighted))

    # Label
    subflavor = lifecycle.get("subflavor", "")
    if stage == "growth":
        if subflavor == "pre_revenue":
            label = "早期成长" if score >= 55 else "早期关注"
        else:
            label = "高成长" if score >= 65 else "成长关注" if score >= 40 else "成长风险"
    elif stage == "mature":
        label = "优质" if score >= 70 else "稳健" if score >= 50 else "关注" if score >= 35 else "偏弱"
    elif stage in ("cyclical", "declining"):
        label = "周期关注" if score >= 50 else "周期风险"
    else:
        label = "中性" if score >= 40 else "待观察"

    # Bullets
    bullets = _build_bullets(lifecycle, dimension_scores, revenue, net_income,
                              gross_margin, op_cashflow, roe, debt_ratio, data_completeness)

    # One-line summary
    summary = _build_summary(stage, score, revenue, net_income, op_cashflow)

    return {
        "fundamental_quality_score": round(score, 1),
        "dimension_scores": dimension_scores,
        "label": label,
        "bullets": bullets,
        "summary": summary,
        "data_completeness": round(data_completeness, 2),
    }


# ── Dimension scorers: Mature stage ───────────────────────────────────────

def _score_profitability_mature(
    roe: pd.Series, gross_margin: pd.Series, net_income: pd.Series
) -> float:
    """Score 0-100 for mature-stage profitability."""
    score = 50.0
    available = 0

    # ROE: map 0%-25% to 0-60 points
    roe_vals = roe.dropna()
    if len(roe_vals) >= 2:
        avg_roe = float(roe_vals.tail(4).mean())
        roe_score = min(60.0, max(0.0, avg_roe / 25.0 * 60.0))
        roe_trend = _trend_delta(roe)
        if roe_trend is not None and roe_trend > 0:
            roe_score = min(60.0, roe_score + 10.0)
        score = roe_score
        available += 1

    # Gross margin: Gaussian around 30%
    gm_vals = gross_margin.dropna()
    if len(gm_vals) >= 2:
        avg_gm = float(gm_vals.tail(4).mean())
        gm_score = min(40.0, 40.0 / (1.0 + np.exp(-(avg_gm - 20.0) / 5.0)))
        if available > 0:
            score = score * 0.7 + gm_score * 0.3
        else:
            score = gm_score
        available += 1

    # Net income trend
    ni_trend = _trend_delta(net_income)
    if ni_trend is not None:
        ni_bonus = max(-10.0, min(10.0, ni_trend * 20.0))
        score = min(100.0, max(0.0, score + ni_bonus))

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


def _score_growth_quality_mature(
    revenue: pd.Series
) -> float:
    """Score 0-100 for mature-stage growth quality."""
    score = 50.0
    available = 0

    rev_cagr = _quarterly_cagr(revenue)
    if rev_cagr is not None:
        # Mature: moderate growth is good, negative is bad, excessive is neutral
        if 0.03 < rev_cagr <= 0.15:
            growth_score = 50.0 + rev_cagr * 200.0  # 50-80
        elif rev_cagr > 0.15:
            growth_score = 60.0  # hypergrowth in mature → suspicious
        elif rev_cagr >= 0:
            growth_score = 50.0 + rev_cagr * 100.0  # 50-53
        else:
            growth_score = 50.0 + rev_cagr * 150.0  # below 50 for negative
        score = max(10.0, min(90.0, growth_score))
        available += 1

    # Revenue acceleration / deceleration
    rev_accel = _acceleration(revenue)
    if rev_accel is not None:
        accel_bonus = max(-8.0, min(8.0, rev_accel * 15.0))
        score = min(100.0, max(0.0, score + accel_bonus))

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


def _score_financial_health_mature(
    debt_ratio: pd.Series, current_ratio: pd.Series,
    op_cashflow: pd.Series
) -> float:
    """Score 0-100 for mature-stage financial health."""
    score = 50.0
    available = 0

    # Debt ratio: lower is better, 20-70% range
    dr_vals = debt_ratio.dropna()
    if len(dr_vals) >= 2:
        avg_dr = float(dr_vals.tail(4).mean())
        if avg_dr < 20:
            dr_score = 40.0
        elif avg_dr < 50:
            dr_score = 40.0 - (avg_dr - 20) * 0.4  # 40 → 28
        elif avg_dr < 70:
            dr_score = 28.0 - (avg_dr - 50) * 1.0  # 28 → 8
        else:
            dr_score = max(0.0, 8.0 - (avg_dr - 70) * 0.3)
        score = score * 0.5 + dr_score * 0.5
        available += 1

    # Current ratio: 1.0-3.0 is healthy
    cr_vals = current_ratio.dropna()
    if len(cr_vals) >= 2:
        avg_cr = float(cr_vals.tail(4).mean())
        cr_score = _gaussian(avg_cr, optimal=2.0, width=0.8) * 30.0
        score = score * 0.6 + cr_score * 0.4
        available += 1

    # OpCF stability
    cf_vals = op_cashflow.dropna()
    if len(cf_vals) >= 3:
        cf_pos_ratio = (cf_vals > 0).mean()
        cf_score = cf_pos_ratio * 20.0
        score = min(100.0, score + cf_score)
        available += 1

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


# ── Dimension scorers: Growth stage ────────────────────────────────────────

def _score_profitability_growth(
    lifecycle: Dict[str, Any],
    net_income: pd.Series, gross_margin: pd.Series
) -> float:
    """Score 0-100 for growth-stage profitability. Does NOT penalize losses."""
    subflavor = lifecycle.get("subflavor", "")
    score = 50.0  # neutral baseline for growth

    # Loss narrowing = positive signal
    ni_trend = _trend_delta(net_income)
    if ni_trend is not None and ni_trend > 0:
        score += min(15.0, ni_trend * 30.0)

    # Gross margin trend
    gm_trend = _trend_delta(gross_margin)
    if gm_trend is not None and gm_trend > 0:
        score += min(10.0, gm_trend * 20.0)

    # Pre-revenue: only margin trend matters
    if subflavor == "pre_revenue":
        score = min(70.0, max(35.0, score))

    return min(100.0, max(0.0, score))


def _score_growth_quality_growth(
    df: pd.DataFrame, lifecycle: Dict[str, Any],
    revenue: pd.Series, net_income: pd.Series, op_cashflow: pd.Series
) -> float:
    """Score 0-100 for growth-stage trajectory quality."""
    subflavor = lifecycle.get("subflavor", "")
    score = 50.0
    available = 0

    rev_cagr = _quarterly_cagr(revenue)
    if rev_cagr is not None:
        if subflavor == "pre_revenue":
            # Pre-revenue: any positive revenue trend is good
            growth_score = 50.0 + min(40.0, rev_cagr * 100.0) if rev_cagr > 0 else 50.0
        elif subflavor == "hypergrowth":
            # Hypergrowth: reward the growth rate directly
            growth_score = 50.0 + min(45.0, rev_cagr * 120.0)
        else:
            # Expansion: reward 10-25% CAGR
            growth_score = 50.0 + min(40.0, rev_cagr * 160.0)
        score = max(20.0, min(95.0, growth_score))
        available += 1

    # Revenue acceleration
    rev_accel = _acceleration(revenue)
    if rev_accel is not None:
        accel_bonus = max(-5.0, min(15.0, rev_accel * 25.0)) if rev_accel > 0 else max(-10.0, rev_accel * 20.0)
        score = min(100.0, max(0.0, score + accel_bonus))

    # Pre-revenue: R&D proxy from opex if available
    if subflavor == "pre_revenue":
        opex = _pick_metric(df, ["oper_cost", "total_cogs", "sell_exp"])
        if opex.notna().sum() >= 2:
            opex_trend = _trend_delta(opex)
            if opex_trend is not None and opex_trend > 0.05:
                score = min(100.0, score + 8.0)  # increasing investment = progress signal
            elif opex_trend is not None and opex_trend < -0.05:
                score = max(0.0, score - 8.0)  # cutting investment → concern

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


def _score_financial_health_growth(
    lifecycle: Dict[str, Any],
    op_cashflow: pd.Series, debt_ratio: pd.Series,
    total_assets: pd.Series
) -> float:
    """Score 0-100 for growth-stage financial health. Focuses on cash runway."""
    subflavor = lifecycle.get("subflavor", "")
    score = 50.0
    available = 0

    # Cash runway for pre-revenue / high-burn growth
    cf_vals = op_cashflow.dropna()
    if len(cf_vals) >= 3 and subflavor in ("pre_revenue", "hypergrowth"):
        avg_burn = float(cf_vals.tail(4).mean())
        if avg_burn < 0:  # burning cash
            # Estimate cash from assets (very rough proxy)
            asset_vals = total_assets.dropna()
            if len(asset_vals) >= 1:
                latest_assets = float(asset_vals.iloc[-1])
                cash_proxy = latest_assets * 0.15  # assume ~15% of assets = cash
                burn_rate = abs(avg_burn)
                runway_quarters = cash_proxy / max(burn_rate, 1.0)
                runway_score = min(40.0, runway_quarters / 4.0 * 10.0)  # 0 at 0Q, 40 at 16Q+
                score = score * 0.5 + runway_score * 0.5
                available += 1

            # Burn rate trend (improving = less negative)
            burn_trend = _trend_delta(op_cashflow)
            if burn_trend is not None and burn_trend > 0:
                score = min(100.0, score + 12.0)
            elif burn_trend is not None and burn_trend < -0.1:
                score = max(0.0, score - 10.0)

    # Debt ratio (less important for growth, but still informative)
    dr_vals = debt_ratio.dropna()
    if len(dr_vals) >= 2:
        avg_dr = float(dr_vals.tail(4).mean())
        if avg_dr < 40:
            dr_score = 30.0
        elif avg_dr < 60:
            dr_score = 20.0
        else:
            dr_score = max(0.0, 20.0 - (avg_dr - 60) * 0.5)
        score = score * 0.6 + dr_score * 0.4
        available += 1

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


# ── Earnings quality (universal) ──────────────────────────────────────────

def _score_earnings_quality_universal(
    net_income: pd.Series,
    op_cashflow: pd.Series, total_assets: pd.Series
) -> float:
    """Score 0-100 for earnings quality. Works for all lifecycle stages."""
    score = 50.0
    available = 0

    # OpCF / Net Income ratio (consistency check)
    ni_vals = net_income.dropna()
    cf_vals = op_cashflow.dropna()
    common_idx = ni_vals.index.intersection(cf_vals.index)
    if len(common_idx) >= 3:
        ni_common = ni_vals.loc[common_idx]
        cf_common = cf_vals.loc[common_idx]
        ratios = cf_common / ni_common.replace(0, np.nan)
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratios) >= 2:
            avg_ratio = float(ratios.tail(4).mean())
            # > 1.0 = cash flow exceeds earnings (high quality)
            # 0.5-1.0 = reasonable
            # < 0.5 or negative = low quality
            if avg_ratio > 1.0:
                cf_score = min(30.0, 15.0 + (avg_ratio - 1.0) * 5.0)
            elif avg_ratio > 0.5:
                cf_score = 15.0 * (avg_ratio - 0.5) / 0.5
            else:
                cf_score = 0.0
            score = score * 0.5 + cf_score * 0.5
            available += 1

    # Accruals proxy
    if len(common_idx) >= 3 and not total_assets.isna().all():
        ni_common = ni_vals.loc[common_idx]
        cf_common = cf_vals.loc[common_idx]
        assets_common = total_assets.loc[total_assets.index.intersection(common_idx)]
        if len(assets_common) >= 2:
            accruals = (ni_common - cf_common) / assets_common.replace(0, np.nan)
            accruals = accruals.replace([np.inf, -np.inf], np.nan).dropna()
            if len(accruals) >= 2:
                avg_accrual = float(accruals.tail(4).mean())
                accrual_trend = _trend_delta(pd.Series(accruals.values, name="accruals"))
                # Low/declining accruals = high quality
                if abs(avg_accrual) < 0.03:
                    accrual_score = 20.0
                elif abs(avg_accrual) < 0.06:
                    accrual_score = 12.0
                else:
                    accrual_score = 5.0
                if accrual_trend is not None and accrual_trend < 0:
                    accrual_score += 5.0  # accruals declining = good
                score = min(100.0, score + accrual_score)
                available += 1

    if available == 0:
        return 50.0
    return min(100.0, max(0.0, score))


# ── Helpers ────────────────────────────────────────────────────────────────

def _trend_delta(series: pd.Series) -> Optional[float]:
    """Recent vs prior half ratio. Positive = improving."""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < 2:
        return None
    n = max(2, len(valid) // 2)
    recent = float(valid.tail(n).mean())
    prior = float(valid.head(n).mean())
    scale = max(abs(prior), abs(recent), 1e-9)
    if scale == 0:
        return 0.0
    return float((recent - prior) / scale)


def _quarterly_cagr(series: pd.Series) -> Optional[float]:
    """Estimate annualized revenue CAGR from quarterly data."""
    valid = series.dropna()
    if len(valid) < 4:
        return None
    vals = valid.values
    quarters = len(vals)
    if quarters < 2:
        return None
    # Log-linear regression
    x = np.arange(quarters)
    y = np.log(np.maximum(vals, 1.0))
    if np.isnan(y).all():
        return None
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return None
    try:
        slope = np.polyfit(x[mask], y[mask], 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return None
    # Annualize: quarterly slope → annual
    annual_cagr = np.exp(slope * 4) - 1
    if not np.isfinite(annual_cagr):
        return None
    return float(max(-0.50, min(2.0, annual_cagr)))


def _acceleration(series: pd.Series) -> Optional[float]:
    """2nd derivative: is growth accelerating or decelerating?"""
    valid = series.dropna()
    if len(valid) < 6:
        return None
    n = len(valid)
    recent_half = valid.iloc[n//2:]
    early_half = valid.iloc[:n//2]
    recent_growth = (recent_half.iloc[-1] - recent_half.iloc[0]) / max(abs(recent_half.iloc[0]), 1e-9)
    early_growth = (early_half.iloc[-1] - early_half.iloc[0]) / max(abs(early_half.iloc[0]), 1e-9)
    return float(recent_growth - early_growth)


def _volatility(series: pd.Series) -> Optional[float]:
    """Coefficient of variation."""
    valid = series.dropna()
    if len(valid) < 3:
        return None
    mean = float(valid.mean())
    if mean == 0:
        return None
    return float(valid.std() / abs(mean))


def _consecutive_positive_quarters(series: pd.Series) -> int:
    """Count consecutive quarters with positive values from the end."""
    valid = series.dropna()
    if valid.empty:
        return 0
    count = 0
    for val in reversed(valid.values):
        if val > 0:
            count += 1
        else:
            break
    return count


def _gaussian(x: float, optimal: float, width: float) -> float:
    """Gaussian scoring function. Returns 0.0-1.0, peaks at optimal."""
    return float(np.exp(-0.5 * ((x - optimal) / width) ** 2))


def _linear(x: float, low: float, high: float, max_score: float, invert: bool = False) -> float:
    """Linear interpolation between low and high, capped at max_score."""
    ratio = (x - low) / (high - low) if high != low else 0.5
    ratio = max(0.0, min(1.0, ratio))
    if invert:
        ratio = 1.0 - ratio
    return ratio * max_score


# ── Bullets & Summary ──────────────────────────────────────────────────────

def _build_bullets(
    lifecycle: Dict[str, Any],
    dim_scores: Dict[str, float],
    revenue: pd.Series, net_income: pd.Series,
    gross_margin: pd.Series, op_cashflow: pd.Series,
    roe: pd.Series, debt_ratio: pd.Series,
    data_completeness: float,
) -> List[str]:
    """Build Chinese bullet points for the quality report."""
    bullets: List[str] = []
    stage = lifecycle.get("stage", "unknown")
    subflavor = lifecycle.get("subflavor", "")

    # Revenue
    rev_cagr = _quarterly_cagr(revenue)
    if rev_cagr is not None:
        direction = "快速增长" if rev_cagr > 0.15 else "稳健增长" if rev_cagr > 0.05 else "缓慢增长" if rev_cagr > 0 else "下滑"
        bullets.append(f"营收{direction}（年化{rev_cagr * 100:.1f}%），基于{revenue.notna().sum()}个季度数据。")

    # Net income / profitability
    ni_trend = _trend_delta(net_income)
    if ni_trend is not None:
        ni_recent = net_income.dropna().tail(4)
        is_profitable = float(ni_recent.mean()) > 0 if len(ni_recent) > 0 else False
        if stage == "growth" and not is_profitable:
            if ni_trend > 0:
                bullets.append("公司尚处投入期，但亏损幅度收窄，经营杠杆逐步显现。")
            else:
                bullets.append("公司尚处投入期，盈利路径有待兑现，关注现金消耗节奏。")
        else:
            ni_label = "改善" if ni_trend > 0.05 else "承压" if ni_trend < -0.05 else "平稳"
            bullets.append(f"净利润趋势{ni_label}。")

    # Gross margin
    gm_trend = _trend_delta(gross_margin)
    if gm_trend is not None:
        gm_label = "改善" if gm_trend > 0.03 else "承压" if gm_trend < -0.03 else "稳定"
        bullets.append(f"毛利率{gm_label}，反映产品竞争力和成本控制能力。")

    # Cash flow
    cf_trend = _trend_delta(op_cashflow)
    if cf_trend is not None:
        cf_label = "改善" if cf_trend > 0.05 else "走弱" if cf_trend < -0.05 else "稳定"
        bullets.append(f"经营现金流{cf_label}。")

    # Financial health
    dr_vals = debt_ratio.dropna()
    if len(dr_vals) >= 2:
        avg_dr = float(dr_vals.tail(4).mean())
        if avg_dr > 60:
            bullets.append(f"资产负债率偏高（{avg_dr:.0f}%），关注偿债压力。")
        elif avg_dr < 30:
            bullets.append(f"资产负债率低（{avg_dr:.0f}%），财务结构稳健。")

    # Lifecycle note
    if stage == "growth" and subflavor == "pre_revenue":
        bullets.append("企业处于早期成长阶段，评分侧重成长潜力和现金可持续性，不因当前未盈利而直接否定。")

    # Data completeness note
    if data_completeness < 0.5:
        bullets.append(f"财务数据覆盖度有限（{data_completeness:.0%}），部分指标基于有限样本估计。")

    return bullets[:5]


def _build_summary(
    stage: str, score: float,
    revenue: pd.Series, net_income: pd.Series, op_cashflow: pd.Series
) -> str:
    """One-line Chinese summary of fundamental quality."""
    stage_labels = {
        "growth": "成长期", "mature": "成熟期",
        "cyclical": "周期型", "declining": "下滑期",
        "transitional": "转型期", "unknown": "数据有限",
    }
    stage_cn = stage_labels.get(stage, stage)

    rev_cagr = _quarterly_cagr(revenue)
    rev_str = f"营收CAGR {rev_cagr*100:.1f}%" if rev_cagr is not None else "营收趋势未知"

    ni_trend = _trend_delta(net_income)
    ni_str = "盈利改善" if (ni_trend is not None and ni_trend > 0) else "盈利承压" if (ni_trend is not None and ni_trend <= 0) else "盈利趋势未知"

    cf_trend = _trend_delta(op_cashflow)
    cf_str = "现金流改善" if (cf_trend is not None and cf_trend > 0) else "现金流承压" if (cf_trend is not None and cf_trend <= 0) else "现金流趋势未知"

    return f"【{stage_cn}】{rev_str}，{ni_str}，{cf_str}。综合质量得分{score:.0f}/100。"
