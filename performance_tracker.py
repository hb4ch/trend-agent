"""Track performance of past agent stock picks — final refined version."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
TICKS_DIR = Path("data/stock_ticks")

BUYABLE_OLD = {"推荐", "强烈推荐"}
WATCH_OLD = {"观察", "回避"}
BUYABLE_NEW = {"可配置", "strong_buy", "buy"}
WATCH_NEW = {"等回踩", "watch"}


def parse_report_date(filename: str) -> str:
    m = re.search(r"report_(\d{8})_\d{6}\.md", filename)
    return m.group(1) if m else "unknown"


def extract_stocks_from_md(path: Path) -> list[dict]:
    text = path.read_text()
    date_str = parse_report_date(path.name)
    report_date = datetime.strptime(date_str, "%Y%m%d")

    stocks = []
    sections = re.split(r"\n(?=### \S+ \d{6}\.[A-Z]+)", text)

    for section in sections:
        header_match = re.match(r"###\s+(.+?)\s+(\d{6}\.[A-Z]+)", section)
        if not header_match:
            continue

        name = header_match.group(1).strip()
        code = header_match.group(2).strip()

        conclusion = ""
        conc_match = re.search(r"结论[：:]\s*(.+?)(?:\n|$)", section)
        if conc_match:
            conclusion = conc_match.group(1).strip()

        if conclusion in BUYABLE_OLD or conclusion in BUYABLE_NEW:
            action = "buy"
        elif conclusion in WATCH_OLD or conclusion in WATCH_NEW:
            action = "watch"
        else:
            action = "unknown"

        # System type
        if conclusion in BUYABLE_OLD or conclusion in WATCH_OLD:
            system = "old"
        else:
            system = "new"

        # Extract quality and position info
        quality = None
        qm = re.search(r"综合质量得分[：:]*\s*(\d+)", section)
        if not qm:
            qm = re.search(r"评分[：:]*\s*(\d+\.?\d*)", section)
        if qm:
            quality = float(qm.group(1))

        position = None
        pm = re.search(r"(\d+)%分位", section)
        if pm:
            position = int(pm.group(1))

        stocks.append({
            "name": name, "code": code, "date": date_str,
            "report_date": report_date, "conclusion": conclusion,
            "action": action, "system": system,
            "quality": quality, "position": position,
        })

    return stocks


def get_nearest_price(code: str, target_date: datetime) -> tuple[float | None, str | None]:
    tick_path = TICKS_DIR / f"{code}.parquet"
    if not tick_path.exists():
        return None, None

    df = pd.read_parquet(tick_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    mask = df["trade_date"] <= target_date
    if not mask.any():
        return None, None

    row = df[mask].iloc[-1]
    return float(row["close"]), row["trade_date"].strftime("%Y-%m-%d")


def get_latest_trade_date(code: str) -> datetime:
    """Get the most recent trade date for a stock."""
    tick_path = TICKS_DIR / f"{code}.parquet"
    if not tick_path.exists():
        return pd.to_datetime("2026-05-22")  # fallback
    df = pd.read_parquet(tick_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df["trade_date"].max()


def compute_returns(stocks: list[dict]) -> pd.DataFrame:
    rows = []

    for s in stocks:
        code = s["code"]
        entry_dt = s["report_date"]
        entry_price, actual_entry_date = get_nearest_price(code, entry_dt)
        if entry_price is None:
            continue

        # Use actual latest available trade date for this stock
        latest_dt = get_latest_trade_date(code)
        latest_price, actual_latest_date = get_nearest_price(code, latest_dt)
        if latest_price is None:
            continue

        if actual_entry_date == actual_latest_date:
            continue

        holding_days = (pd.to_datetime(actual_latest_date) - pd.to_datetime(actual_entry_date)).days
        total_return = (latest_price / entry_price - 1) * 100

        fwd_returns = {}
        for days in [1, 3, 5, 10, 20]:
            future_date = entry_dt + timedelta(days=days)
            fp, _ = get_nearest_price(code, future_date)
            if fp:
                fwd_returns[f"d{days}"] = round((fp / entry_price - 1) * 100, 2)
            else:
                fwd_returns[f"d{days}"] = None

        rows.append({
            "code": code,
            "name": s["name"],
            "report_date": s["date"],
            "entry_date": actual_entry_date,
            "entry_price": round(entry_price, 2),
            "latest_date": actual_latest_date,
            "latest_price": round(latest_price, 2),
            "holding_days": holding_days,
            "total_return_pct": round(total_return, 2),
            "action": s["action"],
            "conclusion": s["conclusion"],
            "system": s["system"],
            "quality": s.get("quality"),
            "position_pct": s.get("position"),
            **fwd_returns,
        })

    return pd.DataFrame(rows)


def print_summary(label, df):
    if df.empty:
        return
    print(f"\n--- {label} ({len(df)} picks) ---")
    print(f"  Avg return: {df['total_return_pct'].mean():+.2f}%")
    print(f"  Median return: {df['total_return_pct'].median():+.2f}%")
    print(f"  Win rate: {(df['total_return_pct'] > 0).mean()*100:.1f}%")
    print(f"  Best: {df['total_return_pct'].max():+.2f}%  Worst: {df['total_return_pct'].min():+.2f}%")

    for action in ["buy", "watch"]:
        sub = df[df["action"] == action]
        if sub.empty:
            continue
        print(f"  [{action}] {len(sub)} picks, avg {sub['total_return_pct'].mean():+.2f}%, "
              f"median {sub['total_return_pct'].median():+.2f}%, "
              f"win {(sub['total_return_pct'] > 0).mean()*100:.0f}%")


def main():
    reports = sorted(REPORTS_DIR.glob("report_20260[56]*.md"))
    print(f"Analyzing {len(reports)} reports from May-Jun 2026\n")

    all_stocks = []
    for rp in reports:
        stocks = extract_stocks_from_md(rp)
        all_stocks.extend(stocks)

    # Deduplicate same stock+date
    seen = set()
    unique = []
    for s in all_stocks:
        key = (s["code"], s["date"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    # Conclusion breakdown
    conc_counts = defaultdict(int)
    for s in unique:
        conc_counts[s["conclusion"]] += 1
    print("Conclusions breakdown:")
    for k, v in sorted(conc_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Compute returns
    df = compute_returns(unique)
    if df.empty:
        print("\nNo returns to compute")
        return

    # Fix date comparison: report_date is YYYYMMDD string
    recent_mask = df["report_date"] >= "20260510"
    recent = df[recent_mask]
    older = df[~recent_mask]

    # Also separate by system
    old_sys = df[df["system"] == "old"]
    new_sys = df[df["system"] == "new"]

    print(f"\n{'='*65}")
    print(f"Overall: {len(df)} picks with forward returns")
    print(f"  Recent (May 10-23): {len(recent)} picks")
    print(f"  Older (May 1-9): {len(older)} picks")
    print(f"  Old system (观察/推荐/回避): {len(old_sys)} picks")
    print(f"  New system (等回踩/可配置): {len(new_sys)} picks")
    print(f"{'='*65}")

    print_summary("OVERALL", df)
    print_summary("RECENT (May 10-23)", recent)
    print_summary("OLDER (May 1-9)", older)
    print_summary("OLD SYSTEM (观察/推荐/回避/强烈推荐)", old_sys)
    print_summary("NEW SYSTEM (等回踩/可配置)", new_sys)

    # First-appearance analysis: only count each stock's first recommendation
    df_sorted = df.sort_values("report_date")
    first_appearance = df_sorted.drop_duplicates(subset="code", keep="first")
    print(f"\n{'='*65}")
    print(f"FIRST-APPEARANCE-ONLY ANALYSIS ({len(first_appearance)} unique stocks)")
    print(f"{'='*65}")
    print_summary("First appearance only", first_appearance)

    # Frequently recommended stocks performance
    print(f"\n{'='*65}")
    print(f"FREQUENTLY RECOMMENDED STOCKS (appeared ≥3 times)")
    print(f"{'='*65}")
    stock_counts = df.groupby("code").size().sort_values(ascending=False)
    for code, count in stock_counts[stock_counts >= 3].items():
        sub = df[df["code"] == code]
        name = sub["name"].iloc[0]
        first = sub.iloc[0]
        last = sub.iloc[-1]
        print(f"  {code} {name:<8s}: {count}x, first {first['report_date']} ({first['conclusion']}), "
              f"return from first: {first['total_return_pct']:+.2f}%")

    # Top/bottom performers
    print(f"\n{'='*65}")
    print(f"TOP 10 WINNERS")
    print(f"{'='*65}")
    for _, row in df.nlargest(10, "total_return_pct").iterrows():
        print(f"  {row['total_return_pct']:+6.1f}% {row['code']} {row['name']:<8s} | "
              f"{row['report_date']} → {row['latest_date']} ({row['holding_days']:2d}d) | "
              f"entry {row['entry_price']:.2f} → {row['latest_price']:.2f} | "
              f"{row['conclusion']}")

    print(f"\n{'='*65}")
    print(f"TOP 10 LOSERS")
    print(f"{'='*65}")
    for _, row in df.nsmallest(10, "total_return_pct").iterrows():
        print(f"  {row['total_return_pct']:+6.1f}% {row['code']} {row['name']:<8s} | "
              f"{row['report_date']} → {row['latest_date']} ({row['holding_days']:2d}d) | "
              f"entry {row['entry_price']:.2f} → {row['latest_price']:.2f} | "
              f"{row['conclusion']}")

    # Performance by report date
    print(f"\n{'='*65}")
    print(f"BY REPORT DATE")
    print(f"{'='*65}")
    for date in sorted(df["report_date"].unique()):
        sub = df[df["report_date"] == date]
        conc_mix = sub["conclusion"].value_counts().to_dict()
        mix_str = ", ".join(f"{k}:{v}" for k, v in sorted(conc_mix.items(), key=lambda x: -x[1]))
        print(f"  {date}: {len(sub):2d} picks, avg {sub['total_return_pct'].mean():+5.2f}%, "
              f"win {(sub['total_return_pct'] > 0).mean()*100:3.0f}%  [{mix_str}]")

    # Forward returns
    print(f"\n{'='*65}")
    print(f"FORWARD RETURNS BY HORIZON")
    print(f"{'='*65}")
    for horizon in ["d1", "d3", "d5", "d10", "d20"]:
        valid = df[df[horizon].notna()]
        if valid.empty:
            continue
        print(f"\n  {horizon}:")
        for action in ["buy", "watch"]:
            sub = valid[valid["action"] == action]
            if sub.empty:
                continue
            print(f"    [{action:5s}] {len(sub):3d} picks, avg {sub[horizon].mean():+5.2f}%, "
                  f"median {sub[horizon].median():+5.2f}%, win {(sub[horizon] > 0).mean()*100:3.0f}%")

    # Save
    out_path = "reports/performance_tracking.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
