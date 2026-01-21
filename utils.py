"""
Utility functions shared across the trend-agent project.

This module contains common helper functions for text processing,
JSON parsing, and formatting that were previously duplicated
across multiple files.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging for the entire project
logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the trend-agent project.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs to console only.
        format_string: Custom format string. If None, uses default.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def truncate(text: str, limit: int = 5000) -> str:
    """
    Truncate text to specified limit with indicator.

    Args:
        text: Text to truncate
        limit: Maximum length before truncation

    Returns:
        Original text if under limit, otherwise truncated text with indicator
    """
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def pretty_print(payload: Any) -> str:
    """
    Format payload as pretty JSON string.

    Args:
        payload: Object to format (dict, list, or other)

    Returns:
        Pretty-printed JSON string for dicts/lists, str() for other types
    """
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, ValueError) as e:
            logger.debug(f"JSON serialization failed: {e}")
            return str(payload)
    return str(payload)


def safe_json_loads(text: str) -> Dict:
    """
    Safely parse JSON from text, handling common edge cases.

    Handles:
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON embedded in other text
    - Malformed JSON (returns empty dict)

    Args:
        text: Text containing JSON

    Returns:
        Parsed dictionary, or empty dict if parsing fails
    """
    import re

    text = (text or "").strip()
    if not text:
        return {}

    # Remove markdown code block wrappers
    if text.startswith("```"):
        text = text.strip("`").strip()

    # Try to find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # Clean up any remaining markdown artifacts
    text = text.strip().strip("```").strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parsing failed: {e}")
        return {}


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.

    Args:
        text: Text to search for URLs

    Returns:
        List of unique URLs with http/https prefix added
    """
    import re

    urls = re.findall(r"(https?://[^\s\]\)\"']+|www\.[^\s\]\)\"']+)", text)
    unique = []
    for url in urls:
        if url.startswith("www."):
            url = "https://" + url
        if url not in unique:
            unique.append(url)
    return unique


def stock_symbol(ts_code: str) -> str:
    """
    Extract stock symbol from ts_code.

    Args:
        ts_code: Stock code in format "000001.SZ"

    Returns:
        Symbol part only, e.g., "000001"
    """
    return str(ts_code or "").split(".")[0]


def normalize_verdict(value: Any) -> str:
    """
    Normalize audit verdict to standard values.

    Args:
        value: Verdict value (str, dict, or other)

    Returns:
        Normalized verdict: "pass", "warn", or "fail"
    """
    text = str(value or "").strip().lower()
    if "fail" in text or "否决" in text:
        return "fail"
    if "warn" in text or "warning" in text or "存疑" in text or "谨慎" in text:
        return "warn"
    if "pass" in text or "通过" in text:
        return "pass"
    if text in {"fail", "warn", "pass"}:
        return text
    return "warn"


def parse_result_date(value: Any) -> Optional[datetime]:
    """
    Parse date from various formats.

    Handles:
    - YYYY-MM-DD
    - YYYY/MM/DD
    - YYYY.MM.DD
    - YYYYMMDD

    Args:
        value: Date value (string or other)

    Returns:
        datetime object or None if parsing fails
    """
    import re

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # Try common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    # Try YYYYMMDD format
    if re.fullmatch(r"\d{8}", text):
        try:
            return datetime.strptime(text, "%Y%m%d")
        except ValueError:
            return None

    return None


def is_recent(dt: Optional[datetime], max_age_days: int) -> bool:
    """
    Check if datetime is within recent days.

    Args:
        dt: Datetime to check (None returns False)
        max_age_days: Maximum age in days

    Returns:
        True if datetime is within max_age_days from now
    """
    if dt is None:
        return False
    from datetime import timedelta

    return dt >= (datetime.now() - timedelta(days=max_age_days))


def trace_append(path: Optional[Path], event: str, payload: Any) -> None:
    """
    Append event to trace file for debugging/audit.

    Args:
        path: Path to trace file (None skips writing)
        event: Event name
        payload: Event payload (must be JSON-serializable)
    """
    if not path:
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")

        with path.open("a", encoding="utf-8") as f:
            trace_entry = {
                "ts": datetime.now().isoformat(),
                "event": event,
                "payload": payload,
            }
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    except (OSError, IOError) as e:
        logger.warning(f"Failed to write trace: {e}")


def create_row_fingerprint(row: Dict[str, Any], fields: List[str]) -> str:
    """
    Create fingerprint hash for data row.

    Useful for caching results based on row content.

    Args:
        row: Data row as dictionary
        fields: Field names to include in fingerprint

    Returns:
        Hexadecimal hash string
    """
    text = " ".join([str(row.get(field, "")) for field in fields])
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{row.get('ts_code','')}::{h[:16]}"


# Module-level convenience functions for backward compatibility
def _truncate(text: str, limit: int = 5000) -> str:
    """Legacy alias for truncate()."""
    return truncate(text, limit)


def _pretty(payload: Any) -> str:
    """Legacy alias for pretty_print()."""
    return pretty_print(payload)


def _debug_print(prefix: str, payload: Any, debug_flag: bool = False) -> None:
    """
    Print debug message if debug flag is enabled.

    Args:
        prefix: Message prefix
        payload: Object to print
        debug_flag: If False, message is not printed
    """
    if debug_flag:
        logger.debug(f"[{prefix}] {truncate(pretty_print(payload), 6000)}")


# =============================================================================
# Dragon Tiger List (龙虎榜) Data Management
# =============================================================================

class DragonTigerList:
    """
    龙虎榜数据管理类

    用于加载和分析龙虎榜数据，识别资金流向和游资偏好。

    Example:
        dtl = DragonTigerList()

        # 识别热门题材
        themes = dtl.identify_hot_themes(days=30)

        # 获取题材资金分析
        analysis = dtl.get_theme_capital_analysis("半导体", days=30)

        # 获取最近上榜股票（用于排除）
        hot_stocks = dtl.get_recent_toplist_stocks(days=60)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DragonTigerList manager.

        Args:
            data_dir: Base directory containing top_list and top_inst folders
        """
        from pathlib import Path

        self.data_dir = Path(data_dir)
        self.top_list_dir = self.data_dir / "top_list"
        self.top_inst_dir = self.data_dir / "top_inst"

        # Cache for stock basic info (for industry mapping)
        self._stock_basic_cache = None

        # Verify directories exist
        if not self.top_list_dir.exists():
            logger.warning(f"top_list directory not found: {self.top_list_dir}")
        if not self.top_inst_dir.exists():
            logger.warning(f"top_inst directory not found: {self.top_inst_dir}")

    def load_recent_toplist(self, days: int = 30) -> Any:
        """
        加载最近N天的龙虎榜数据。

        Args:
            days: Number of days to load

        Returns:
            DataFrame with columns: trade_date, ts_code, name, close, pct_change,
                 turnover_rate, l_sell, l_buy, l_amount, net_amount, net_rate,
                 amount_rate, float_values, reason
        """
        from pathlib import Path
        import pandas as pd
        from datetime import datetime, timedelta

        end_date = datetime.now()
        files = []

        for i in range(days):
            date = end_date - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            file_path = self.top_list_dir / f"{date_str}.parquet"
            if file_path.exists():
                files.append(file_path)

        if not files:
            logger.warning(f"No top_list data found for last {days} days")
            return None

        dfs = []
        for f in sorted(files, reverse=True):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.debug(f"Loaded {len(result)} records from {len(dfs)} files")
            return result

        return None

    def load_recent_topinst(self, days: int = 30) -> Any:
        """
        加载最近N天的龙虎榜机构明细。

        Args:
            days: Number of days to load

        Returns:
            DataFrame with columns: trade_date, ts_code, exalter, buy, buy_rate,
                 sell, sell_rate, net_buy, side, reason
        """
        from pathlib import Path
        import pandas as pd
        from datetime import datetime, timedelta

        end_date = datetime.now()
        files = []

        for i in range(days):
            date = end_date - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            file_path = self.top_inst_dir / f"{date_str}.parquet"
            if file_path.exists():
                files.append(file_path)

        if not files:
            logger.warning(f"No top_inst data found for last {days} days")
            return None

        dfs = []
        for f in sorted(files, reverse=True):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.debug(f"Loaded {len(result)} inst records from {len(dfs)} files")
            return result

        return None

    def _load_stock_basic(self) -> Any:
        """Load stock basic info for industry mapping."""
        import pandas as pd

        if self._stock_basic_cache is not None:
            return self._stock_basic_cache

        basic_path = self.data_dir / "stock_basic" / "stock_basic.parquet"
        if basic_path.exists():
            self._stock_basic_cache = pd.read_parquet(basic_path)
            return self._stock_basic_cache

        logger.warning("stock_basic.parquet not found")
        return None

    def identify_hot_themes(self, days: int = 30) -> Dict[str, Dict]:
        """
        识别游资活跃题材。

        基于龙虎榜数据，按行业/板块聚合，识别资金持续流入的热门题材。

        Args:
            days: 时间窗口（天）

        Returns:
            Dict mapping theme name to stats:
            {
                "半导体": {
                    "hit_count": 45,           # 上榜次数
                    "net_buy": 15.2e9,         # 累计净买入额
                    "hot_stocks": ["中芯国际", ...],  # 热门股票
                    "institution_mix": {     # 资金结构
                        "north": 0.45,        # 北上资金占比
                        "inst": 0.35,         # 机构占比
                        "hot_money": 0.20     # 游资占比
                    },
                    "trend": "accelerating"  # 趋势: accelerating/stable/declining
                },
                ...
            }
        """
        import pandas as pd

        df_list = self.load_recent_toplist(days)
        if df_list is None or df_list.empty:
            return {}

        df_inst = self.load_recent_topinst(days)
        df_basic = self._load_stock_basic()

        # Merge with industry info
        if df_basic is not None:
            df_list = df_list.merge(
                df_basic[["ts_code", "industry"]],
                on="ts_code",
                how="left"
            )

        # Group by industry
        result = {}
        for industry, group in df_list.groupby("industry"):
            if pd.isna(industry):
                continue

            # Basic stats
            hit_count = len(group)
            net_buy = group["net_amount"].sum()

            # Hot stocks (top by net_buy)
            hot_stocks = (
                group.groupby("ts_code")["net_amount"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )

            # Get stock names
            if df_basic is not None:
                name_map = df_basic.set_index("ts_code")["name"].to_dict()
                hot_stocks = [name_map.get(code, code) for code in hot_stocks]

            # Institution mix (from top_inst)
            inst_mix = {"north": 0, "inst": 0, "hot_money": 0}
            if df_inst is not None:
                inst_df = df_inst[df_inst["ts_code"].isin(group["ts_code"])]
                if not inst_df.empty:
                    total_buy = inst_df["buy"].sum()
                    if total_buy > 0:
                        # North money (北上)
                        north_df = inst_df[inst_df["exalter"].str.contains("沪股通|深股通", na=False)]
                        inst_mix["north"] = north_df["buy"].sum() / total_buy

                        # Institutions (机构专用)
                        inst_df_only = inst_df[inst_df["exalter"] == "机构专用"]
                        inst_mix["inst"] = inst_df_only["buy"].sum() / total_buy

                        # Hot money (remaining)
                        inst_mix["hot_money"] = 1 - inst_mix["north"] - inst_mix["inst"]

            # Determine trend (compare first half vs second half)
            group_sorted = group.sort_values("trade_date")
            mid_point = len(group_sorted) // 2
            first_half_net = group_sorted.iloc[:mid_point]["net_amount"].sum()
            second_half_net = group_sorted.iloc[mid_point:]["net_amount"].sum()

            if second_half_net > first_half_net * 1.2:
                trend = "accelerating"
            elif second_half_net < first_half_net * 0.8:
                trend = "declining"
            else:
                trend = "stable"

            result[industry] = {
                "hit_count": hit_count,
                "net_buy": float(net_buy),
                "hot_stocks": hot_stocks,
                "institution_mix": inst_mix,
                "trend": trend,
            }

        # Sort by hit_count and net_buy
        sorted_result = {
            k: v for k, v in sorted(
                result.items(),
                key=lambda x: (x[1]["hit_count"], x[1]["net_buy"]),
                reverse=True
            )
        }

        logger.info(f"Identified {len(sorted_result)} themes from toplist")
        return sorted_result

    def get_theme_capital_analysis(self, theme_name: str, days: int = 30) -> Dict:
        """
        获取题材的资金流向分析。

        Args:
            theme_name: 主题名称（行业名称）
            days: 时间窗口（天）

        Returns:
            Dict with capital flow analysis
        """
        import pandas as pd

        df_list = self.load_recent_toplist(days)
        if df_list is None or df_list.empty:
            return {}

        df_basic = self._load_stock_basic()
        if df_basic is not None:
            df_list = df_list.merge(
                df_basic[["ts_code", "industry", "name"]],
                on="ts_code",
                how="left"
            )

        # Filter by theme (industry)
        theme_df = df_list[df_list["industry"] == theme_name]
        if theme_df.empty:
            logger.warning(f"No data found for theme: {theme_name}")
            return {}

        df_inst = self.load_recent_topinst(days)
        inst_analysis = []
        if df_inst is not None:
            theme_inst = df_inst[df_inst["ts_code"].isin(theme_df["ts_code"])]

            # Top buyers by institution type
            if not theme_inst.empty:
                # North money
                north = theme_inst[theme_inst["exalter"].str.contains("沪股通|深股通", na=False)]
                inst_analysis.append({
                    "type": "北上资金",
                    "net_buy": float(north["net_buy"].sum()),
                    "buy": float(north["buy"].sum()),
                })

                # Institutions
                inst_only = theme_inst[theme_inst["exalter"] == "机构专用"]
                inst_analysis.append({
                    "type": "机构专用",
                    "net_buy": float(inst_only["net_buy"].sum()),
                    "buy": float(inst_only["buy"].sum()),
                })

        # Top stocks
        top_stocks = (
            theme_df.groupby("ts_code")
            .agg({
                "name": "first",
                "net_amount": "sum",
                "net_rate": "mean",
            })
            .sort_values("net_amount", ascending=False)
            .head(10)
            .to_dict("index")
        )

        return {
            "theme_name": theme_name,
            "hit_count": len(theme_df),
            "total_net_buy": float(theme_df["net_amount"].sum()),
            "inst_breakdown": inst_analysis,
            "top_stocks": {
                code: {
                    "name": row["name"],
                    "net_buy": float(row["net_amount"]),
                    "avg_net_rate": float(row["net_rate"]),
                }
                for code, row in top_stocks.items()
            },
        }

    def get_recent_toplist_stocks(self, days: int = 60) -> List[str]:
        """
        获取最近上榜的股票列表（用于排除已大涨股票）。

        Args:
            days: 时间窗口（天）

        Returns:
            List of stock codes that appeared on toplist
        """
        df = self.load_recent_toplist(days)
        if df is None or df.empty:
            return []

        return df["ts_code"].unique().tolist()

    def get_stock_toplist_summary(self, ts_code: str, days: int = 30) -> Dict:
        """
        获取个股的龙虎榜汇总。

        Args:
            ts_code: 股票代码
            days: 时间窗口（天）

        Returns:
            Dict with toplist summary for the stock
        """
        import pandas as pd

        df_list = self.load_recent_toplist(days)
        if df_list is None or df_list.empty:
            return {}

        stock_df = df_list[df_list["ts_code"] == ts_code]
        if stock_df.empty:
            return {"ts_code": ts_code, "hits": []}

        df_inst = self.load_recent_topinst(days)
        hits = []

        for _, row in stock_df.iterrows():
            trade_date = row["trade_date"]
            reason = row["reason"]
            net_amount = row["net_amount"]
            amount_rate = row.get("amount_rate", 0)

            # Get institutional details for this date
            buyers = []
            sellers = []

            if df_inst is not None:
                inst_df = df_inst[
                    (df_inst["ts_code"] == ts_code) &
                    (df_inst["trade_date"] == trade_date)
                ]

                for _, inst_row in inst_df.iterrows():
                    inst_info = {
                        "name": inst_row["exalter"],
                        "buy": float(inst_row["buy"]),
                        "sell": float(inst_row["sell"]),
                        "net_buy": float(inst_row["net_buy"]),
                    }
                    if inst_row["net_buy"] > 0:
                        buyers.append(inst_info)
                    elif inst_row["net_buy"] < 0:
                        sellers.append(inst_info)

            # Sort by net_buy
            buyers.sort(key=lambda x: x["net_buy"], reverse=True)
            sellers.sort(key=lambda x: x["net_buy"])

            hits.append({
                "trade_date": trade_date,
                "reason": reason,
                "net_amount": float(net_amount),
                "amount_rate": float(amount_rate),
                "top_buyers": buyers[:5],
                "top_sellers": sellers[:5],
            })

        # Summary stats
        summary = {
            "ts_code": ts_code,
            "hit_count": len(hits),
            "total_net_buy": float(stock_df["net_amount"].sum()),
            "avg_amount_rate": float(stock_df["amount_rate"].mean()),
            "hits": hits,
        }

        return summary

    def identify_smart_money(self, df: Any) -> Dict[str, List[Dict]]:
        """
        识别"聪明钱"（北上资金、机构、知名游资）。

        Args:
            df: top_inst DataFrame

        Returns:
            Dict mapping smart money type to list of their activities
        """
        import pandas as pd

        if df is None or df.empty:
            return {}

        result = {
            "north": [],      # 北上资金
            "inst": [],       # 机构
            "hot_money": [],  # 游资
        }

        # Known hot money seats
        known_seats = [
            "宁波桑田路",
            "上海溧阳路",
            "作手新一",
            "消闲派",
            "宁波桑田路",
            "绍兴路",
            "湖州",
            "上海",
            "深圳",
        ]

        for _, row in df.iterrows():
            exalter = row["exalter"]
            net_buy = float(row["net_buy"])

            info = {
                "exalter": exalter,
                "ts_code": row["ts_code"],
                "buy": float(row["buy"]),
                "sell": float(row["sell"]),
                "net_buy": net_buy,
            }

            # North money
            if "沪股通" in exalter or "深股通" in exalter:
                result["north"].append(info)

            # Institutions
            elif exalter == "机构专用":
                result["inst"].append(info)

            # Hot money (known seats or large net buy)
            elif any(seat in exalter for seat in known_seats) or abs(net_buy) > 5e7:
                result["hot_money"].append(info)

        # Sort by net_buy
        for key in result:
            result[key].sort(key=lambda x: x["net_buy"], reverse=True)

        return result
