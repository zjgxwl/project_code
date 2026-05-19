import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_thesis_full_profile_is_dry_run_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis.py",
            "chapter5",
            "--profile",
            "full",
            "--output-dir",
            str(tmp_path),
            "--max-runs",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run_count: 2" in result.stdout
    assert "full_profile_is_dry_run_only: true" in result.stdout
    first_record = json.loads(result.stdout.splitlines()[0])
    assert first_record["chapter"] == "chapter5"


def test_run_thesis_quick_chapter3_writes_epoch_logs_and_assets(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis.py",
            "chapter3",
            "--profile",
            "quick",
            "--output-dir",
            str(tmp_path),
            "--max-runs",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "records_written: 1" in result.stdout
    summary_csv = tmp_path / "assets" / "chapter3_summary.csv"
    assert summary_csv.exists()
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["chapter"] == "chapter3"
    run_dir = Path(rows[0]["run_dir"])
    assert (run_dir / "metrics_epoch.jsonl").exists()
    assert (run_dir / "summary.json").exists()


def test_run_thesis_cli_overrides_and_batch_logs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis.py",
            "chapter3",
            "--profile",
            "quick",
            "--output-dir",
            str(tmp_path),
            "--max-runs",
            "1",
            "--epochs",
            "2",
            "--lr",
            "0.02",
            "--batch-size",
            "4",
            "--model",
            "vgg11_bn",
            "--sparsity",
            "0.95",
            "--log-batches",
            "--log-interval",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Epoch [1/2] Batch" in result.stdout
    rows = list(csv.DictReader((tmp_path / "assets" / "chapter3_summary.csv").open("r", encoding="utf-8")))
    run_dir = Path(rows[0]["run_dir"])
    config_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "epochs: 2" in config_text
    assert "lr: 0.02" in config_text
    assert "batch_size: 4" in config_text
    assert "name: vgg11_bn" in config_text
    assert (run_dir / "metrics_batch.jsonl").exists()
    first_epoch = json.loads((run_dir / "metrics_epoch.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_epoch["target_sparsity"] == 0.95
    assert "model_params" in first_epoch


def test_import_thesis_results_and_collect_assets(tmp_path: Path) -> None:
    input_csv = tmp_path / "external.csv"
    input_csv.write_text("dataset,top1\ncifar10,91.2\n", encoding="utf-8")

    import_result = subprocess.run(
        [
            sys.executable,
            "scripts/import_thesis_results.py",
            "--input",
            str(input_csv),
            "--output-dir",
            str(tmp_path),
            "--chapter",
            "chapter3",
            "--method",
            "tost",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "records_imported: 1" in import_result.stdout

    collect_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis.py",
            "collect",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "records_written: 1" in collect_result.stdout
    summary_csv = tmp_path / "assets" / "thesis_summary.csv"
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    assert rows[0]["method"] == "tost"
    assert rows[0]["external"] == "True"
