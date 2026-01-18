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
