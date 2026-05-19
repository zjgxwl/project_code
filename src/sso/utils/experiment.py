"""Experiment naming helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def get_timestamp() -> str:
    """Return a compact UTC timestamp for run metadata and filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_optional(name: str, value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{name}-{text}"


def build_run_name(
    method: str,
    scorer: str | None = None,
    model: str | None = None,
    sparsity: float | None = None,
    flops_reduction: float | None = None,
) -> str:
    """Build a readable run name from common experiment dimensions."""
    parts = [method]
    for part in (
        _format_optional("model", model),
        _format_optional("scorer", scorer),
        _format_optional("sparsity", sparsity),
        _format_optional("flops", flops_reduction),
    ):
        if part is not None:
            parts.append(part)
    parts.append(get_timestamp())
    return "_".join(parts)
