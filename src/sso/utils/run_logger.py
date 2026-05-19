"""Structured run-directory logging for thesis experiments."""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .experiment import get_timestamp
from .logging import append_jsonl, ensure_dir, to_jsonable, write_json


def sanitize_run_name(name: str) -> str:
    """Return a filename-safe run name."""
    sanitized = name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "-")
    return sanitized or "run"


def unique_run_dir(output_root: str | Path, run_name: str) -> Path:
    """Create and return a unique run directory under ``output_root``."""
    root = ensure_dir(output_root)
    safe_name = sanitize_run_name(run_name)
    candidate = root / safe_name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    index = 1
    while True:
        candidate = root / f"{safe_name}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def build_environment_record(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collect lightweight runtime environment metadata."""
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": get_timestamp(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:
        record["torch_error"] = str(exc)
    if extra:
        record.update(dict(extra))
    return record


class RunLogger:
    """Write standard logs and artifacts for one experiment run."""

    def __init__(
        self,
        output_root: str | Path,
        run_name: str,
        config: Mapping[str, Any],
        command: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_name = sanitize_run_name(run_name)
        self.run_dir = unique_run_dir(output_root, self.run_name)
        self.artifacts_dir = ensure_dir(self.run_dir / "artifacts")
        self.checkpoints_dir = ensure_dir(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics_epoch.jsonl"
        self.batch_metrics_path = self.run_dir / "metrics_batch.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.metadata = dict(metadata or {})

        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(to_jsonable(config), handle, allow_unicode=True, sort_keys=False)
        with (self.run_dir / "command.txt").open("w", encoding="utf-8") as handle:
            handle.write(" ".join(command or sys.argv))
            handle.write("\n")
        write_json(self.run_dir / "env.json", build_environment_record(self.metadata))

    def log_epoch(self, record: Mapping[str, Any]) -> Path:
        """Append one epoch metric record."""
        return append_jsonl(self.metrics_path, record)

    def log_batch(self, record: Mapping[str, Any]) -> Path:
        """Append one batch metric record."""
        return append_jsonl(self.batch_metrics_path, record)

    def write_summary(self, record: Mapping[str, Any]) -> Path:
        """Write the final summary record and append it to the run index."""
        summary = {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "metrics_epoch_path": self.metrics_path,
            "metrics_batch_path": self.batch_metrics_path,
            **dict(record),
        }
        write_json(self.summary_path, summary)
        append_jsonl(self.run_dir.parent / "runs.jsonl", summary)
        return self.summary_path

    def save_checkpoint(self, state: Mapping[str, Any], name: str) -> Path:
        """Save a torch checkpoint under this run directory."""
        import torch

        path = self.checkpoints_dir / sanitize_run_name(name)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        torch.save(dict(state), path)
        return path
