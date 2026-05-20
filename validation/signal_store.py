from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_SIGNAL_PATH = Path("data/signals/signal_snapshots.parquet")


@dataclass(frozen=True)
class SignalConfig:
    snapshot_path: Path = DEFAULT_SIGNAL_PATH
    signal_date: str | pd.Timestamp | None = None
    score_columns: tuple[str, ...] = (
        "alpha_rank_score",
        "composite_score",
        "technical_selection_score",
        "volume_quality_score",
        "valuation_quality_score",
    )
    metadata_columns: tuple[str, ...] = (
        "ts_code",
        "name",
        "industry",
        "market",
        "exchange",
        "list_type",
        "filter_tier",
        "matched_themes",
    )
    idempotent: bool = True
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def stable_config_hash(config: Any) -> str:
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    text = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_candidates(path: Path) -> pd.DataFrame:
    return read_table(Path(path))


def snapshot_candidates(
    df: pd.DataFrame,
    config: SignalConfig,
    run_id: str | None = None,
    agent_version: str = "unknown",
    config_hash: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Cannot snapshot an empty candidate DataFrame")
    if "ts_code" not in df.columns:
        raise ValueError("Candidate DataFrame must contain ts_code")

    signal_date = _normalize_signal_date(config.signal_date)
    run_id = run_id or f"run_{signal_date.strftime('%Y%m%d')}"
    config_hash = config_hash or stable_config_hash(config)

    rows = _normalize_snapshot_frame(df.copy())
    rows.insert(0, "signal_date", signal_date)
    rows.insert(0, "run_id", str(run_id))
    rows["agent_version"] = str(agent_version)
    rows["config_hash"] = str(config_hash)
    rows["snapshot_created_at"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for key, value in config.extra_metadata.items():
        if key not in rows.columns:
            rows[key] = _normalize_cell(value)

    rows["signal_payload_hash"] = rows.apply(_row_payload_hash, axis=1)
    key_cols = ["run_id", "signal_date", "ts_code"]
    if rows.duplicated(key_cols).any():
        duplicates = rows.loc[rows.duplicated(key_cols, keep=False), key_cols]
        raise ValueError(f"Duplicate snapshot keys in input: {duplicates.to_dict('records')}")

    existing = read_table(config.snapshot_path)
    if existing.empty:
        out = rows
        write_parquet(out, config.snapshot_path)
        return out.reset_index(drop=True)

    existing = _normalize_snapshot_frame(existing)
    existing["signal_date"] = pd.to_datetime(existing["signal_date"]).dt.normalize()
    merged = rows[key_cols + ["signal_payload_hash"]].merge(
        existing[key_cols + ["signal_payload_hash"]],
        on=key_cols,
        how="inner",
        suffixes=("_new", "_existing"),
    )
    conflicts = merged[merged["signal_payload_hash_new"] != merged["signal_payload_hash_existing"]]
    if not conflicts.empty:
        raise ValueError(f"Conflicting immutable snapshot rows: {conflicts[key_cols].to_dict('records')}")

    if not config.idempotent and not merged.empty:
        raise ValueError(f"Snapshot rows already exist: {merged[key_cols].to_dict('records')}")

    overlap_keys = set(_iter_keys(merged, key_cols))
    rows_to_append = rows[~rows.apply(lambda r: tuple(r[c] for c in key_cols) in overlap_keys, axis=1)]
    if rows_to_append.empty:
        return existing.reset_index(drop=True)

    out = pd.concat([existing, rows_to_append], ignore_index=True, sort=False)
    out = out.sort_values(["signal_date", "run_id", "ts_code"]).reset_index(drop=True)
    write_parquet(out, config.snapshot_path)
    return out


def _normalize_signal_date(value: str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d").normalize()
    return pd.to_datetime(value).normalize()


def _normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "signal_date":
            continue
        if out[col].dtype == "object":
            out[col] = out[col].map(_normalize_cell)
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].astype(str)
    return out


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    if pd.isna(value):
        return None
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _row_payload_hash(row: pd.Series) -> str:
    ignored = {"snapshot_created_at", "signal_payload_hash"}
    payload = {str(k): _jsonable(v) for k, v in row.items() if k not in ignored}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_keys(df: pd.DataFrame, columns: Iterable[str]) -> Iterable[tuple[Any, ...]]:
    for _, row in df.iterrows():
        yield tuple(row[col] for col in columns)
