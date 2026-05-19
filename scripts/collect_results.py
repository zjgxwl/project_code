"""Collect JSONL experiment metrics into a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.utils.logging import ensure_dir, flatten_metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            records.append(flatten_metrics(loaded))
    if not records:
        raise ValueError(f"Input JSONL is empty: {path}")
    return records


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def first_non_empty(values: list[Any]) -> Any:
    for value in values:
        if value not in {None, ""}:
            return value
    return ""


def aggregate_records(records: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field, "") for field in group_by)
        grouped[key].append(record)

    fieldnames = sorted({field for record in records for field in record})
    output: list[dict[str, Any]] = []
    for key, group_records in grouped.items():
        aggregated: dict[str, Any] = {field: value for field, value in zip(group_by, key, strict=True)}
        aggregated["count"] = len(group_records)
        for field in fieldnames:
            if field in group_by:
                continue
            values = [record.get(field, "") for record in group_records]
            numeric_values = [value for value in values if is_number(value)]
            non_empty_values = [value for value in values if value not in {None, ""}]
            if non_empty_values and len(numeric_values) == len(non_empty_values):
                aggregated[field] = sum(float(value) for value in numeric_values) / len(numeric_values)
            else:
                aggregated[field] = first_non_empty(values)
        output.append(aggregated)
    return output


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({field for record in records for field in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment JSONL metrics into CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to metrics.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV.")
    parser.add_argument("--group-by", nargs="*", default=None, help="Optional fields to group by.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    output_records = aggregate_records(records, args.group_by) if args.group_by else records
    write_csv(args.output, output_records)
    print(f"records_read: {len(records)}")
    print(f"records_written: {len(output_records)}")
    print(f"summary_csv: {args.output}")


if __name__ == "__main__":
    main()
