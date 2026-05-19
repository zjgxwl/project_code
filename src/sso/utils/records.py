"""Experiment record schema helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import dataset_record_fields
from .experiment import get_timestamp


EXPERIMENT_RECORD_SCHEMA_VERSION = 1


def build_experiment_record(
    *,
    script: str,
    method: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized run record for JSON/JSONL export."""
    config_dict = dict(config)
    model_config = config_dict.get("model", {})
    if not isinstance(model_config, Mapping):
        model_config = {}

    record: dict[str, Any] = {
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "script": script,
        "method": method,
        "model": model_config.get("name", "resnet18"),
        "seed": int(config_dict.get("seed", 42)),
        "timestamp": get_timestamp(),
        **dataset_record_fields(config_dict),
        "metrics": dict(metrics),
    }
    if run_name is not None:
        record["run_name"] = run_name
    if extra:
        record.update(dict(extra))
    validate_experiment_record(record)
    return record


def validate_experiment_record(record: Mapping[str, Any]) -> None:
    """Validate the minimal schema expected by result collection scripts."""
    required = {
        "schema_version",
        "script",
        "method",
        "model",
        "seed",
        "timestamp",
        "dataset",
        "metrics",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Experiment record is missing fields: {missing}")
    if int(record["schema_version"]) != EXPERIMENT_RECORD_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported experiment record schema_version: "
            f"{record['schema_version']}"
        )
    if not isinstance(record["metrics"], Mapping):
        raise ValueError("Experiment record field 'metrics' must be a mapping.")
