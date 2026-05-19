"""Lightweight JSON/JSONL metric logging utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(value: Any) -> Any:
    """Convert common Python/numpy/torch values into JSON-safe objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    # Optional numpy support without making numpy a dependency of this module.
    if value.__class__.__module__.startswith("numpy"):
        if hasattr(value, "item") and callable(value.item):
            try:
                return to_jsonable(value.item())
            except Exception:
                pass
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return to_jsonable(value.tolist())
            except Exception:
                pass

    # Optional torch support without importing torch at module import time.
    if value.__class__.__module__.startswith("torch"):
        if hasattr(value, "detach") and callable(value.detach):
            try:
                detached = value.detach().cpu()
                if detached.numel() == 1:
                    return to_jsonable(detached.item())
                return to_jsonable(detached.tolist())
            except Exception:
                pass
        if hasattr(value, "item") and callable(value.item):
            try:
                return to_jsonable(value.item())
            except Exception:
                pass

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write a JSON record."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(record), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append a record to a JSONL file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("a", encoding="utf-8") as handle:
        json.dump(to_jsonable(record), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return output_path


def flatten_metrics(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries using dot-separated keys."""
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=full_key))
        else:
            flattened[full_key] = to_jsonable(value)
    return flattened


def _unique_json_path(output_dir: Path, run_name: str) -> Path:
    candidate = output_dir / f"{run_name}.json"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = output_dir / f"{run_name}_{index:03d}.json"
        if not candidate.exists():
            return candidate
        index += 1


def save_metrics(
    record: Mapping[str, Any],
    output_dir: str | Path = "outputs/runs",
    run_name: str | None = None,
) -> Path:
    """Save a run record to an individual JSON file and append metrics.jsonl."""
    directory = ensure_dir(output_dir)
    resolved_run_name = run_name or str(record.get("run_name", "run"))
    safe_run_name = resolved_run_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    json_path = _unique_json_path(directory, safe_run_name)

    jsonable_record = to_jsonable({**dict(record), "run_name": safe_run_name})
    write_json(json_path, jsonable_record)
    append_jsonl(directory / "metrics.jsonl", {**jsonable_record, "metrics_path": json_path})
    return json_path
