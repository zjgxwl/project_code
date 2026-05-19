"""Import external or future thesis-result CSV rows into the asset format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.utils import append_jsonl


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import external thesis-result CSV rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thesis"), help="Thesis output directory.")
    parser.add_argument("--chapter", type=str, required=True, choices=["chapter3", "chapter4", "chapter5"], help="Chapter tag.")
    parser.add_argument("--source", type=str, default="external", help="Result source label.")
    parser.add_argument("--method", type=str, default=None, help="Default method if the CSV has no method column.")
    args = parser.parse_args()

    records = read_csv(args.input)
    if not records:
        raise ValueError(f"Input CSV is empty: {args.input}")

    output_path = args.output_dir / "assets" / "external_summary.jsonl"
    for row in records:
        record = {
            "chapter": args.chapter,
            "source": args.source,
            "external": True,
            **row,
        }
        if args.method is not None and not record.get("method"):
            record["method"] = args.method
        append_jsonl(output_path, record)

    print(f"records_imported: {len(records)}")
    print(f"external_summary: {output_path}")


if __name__ == "__main__":
    main()
