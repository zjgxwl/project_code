import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sso.utils.logging import append_jsonl, flatten_metrics, save_metrics, to_jsonable


def test_append_jsonl_writes_multiple_records(tmp_path: Path) -> None:
    output = tmp_path / "metrics.jsonl"
    append_jsonl(output, {"a": 1})
    append_jsonl(output, {"b": 2})

    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_save_metrics_creates_json_jsonl_and_unique_names(tmp_path: Path) -> None:
    first = save_metrics({"metric": 1}, output_dir=tmp_path, run_name="demo")
    second = save_metrics({"metric": 2}, output_dir=tmp_path, run_name="demo")

    assert first.name == "demo.json"
    assert second.name == "demo_001.json"
    assert json.loads(first.read_text(encoding="utf-8"))["metric"] == 1
    assert json.loads(second.read_text(encoding="utf-8"))["metric"] == 2

    jsonl = tmp_path / "metrics.jsonl"
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["metrics_path"].endswith("demo.json")


def test_flatten_metrics_and_to_jsonable_handle_common_types(tmp_path: Path) -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    flattened = flatten_metrics(
        {
            "nested": {"value": 3},
            "path": tmp_path,
            "time": timestamp,
            "tuple": (1, 2),
        }
    )

    assert flattened["nested.value"] == 3
    assert flattened["path"] == str(tmp_path)
    assert flattened["time"] == timestamp.isoformat()
    assert flattened["tuple"] == [1, 2]

    try:
        import torch

        assert to_jsonable(torch.tensor(3.0)) == pytest.approx(3.0)
        assert to_jsonable(torch.tensor([1, 2])) == [1, 2]
    except Exception:
        pass

    try:
        import numpy as np

        assert to_jsonable(np.float32(1.5)) == pytest.approx(1.5)
        assert to_jsonable(np.array([1, 2])) == [1, 2]
    except Exception:
        pass


def test_collect_results_writes_union_field_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "summary.csv"
    append_jsonl(input_path, {"method": "a", "score": 1.0})
    append_jsonl(input_path, {"method": "b", "other": 2.0})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_results.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "records_read: 2" in result.stdout
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert {"method", "score", "other"}.issubset(rows[0].keys())
    assert rows[0]["other"] == ""
    assert rows[1]["score"] == ""


def test_collect_results_group_by_averages_numbers_and_not_bools(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "summary.csv"
    append_jsonl(input_path, {"method": "a", "score": 1.0, "flag": True})
    append_jsonl(input_path, {"method": "a", "score": 3.0, "flag": False})

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_results.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--group-by",
            "method",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["count"] == "2"
    assert float(rows[0]["score"]) == pytest.approx(2.0)
    assert rows[0]["flag"] == "True"


def test_collect_results_rejects_missing_and_empty_input(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    output_path = tmp_path / "summary.csv"
    missing_result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_results.py",
            "--input",
            str(missing),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert missing_result.returncode != 0
    assert "does not exist" in missing_result.stderr

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    empty_result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_results.py",
            "--input",
            str(empty),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert empty_result.returncode != 0
    assert "empty" in empty_result.stderr
