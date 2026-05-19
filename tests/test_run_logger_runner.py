import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sso.training import fit_model
from sso.utils import RunLogger


def _toy_loader() -> DataLoader:
    inputs = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def test_run_logger_writes_standard_files_and_summary(tmp_path: Path) -> None:
    logger = RunLogger(
        tmp_path,
        run_name="demo run",
        config={"training": {"epochs": 1}},
        command=["python", "demo.py"],
        metadata={"chapter": "chapter3"},
    )

    logger.log_epoch({"epoch": 1, "train_loss": 1.0})
    logger.log_batch({"epoch": 1, "batch": 1, "batch_loss": 1.1})
    logger.write_summary({"val_acc": 0.5})

    assert (logger.run_dir / "config.yaml").exists()
    assert (logger.run_dir / "command.txt").read_text(encoding="utf-8").strip() == "python demo.py"
    assert (logger.run_dir / "env.json").exists()
    assert json.loads((logger.run_dir / "summary.json").read_text(encoding="utf-8"))["val_acc"] == pytest.approx(0.5)
    assert len((logger.run_dir / "metrics_epoch.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((logger.run_dir / "metrics_batch.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (tmp_path / "runs.jsonl").exists()


def test_fit_model_logs_each_epoch_and_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    loader = _toy_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = RunLogger(tmp_path, "fit", config={}, command=["pytest"])

    result = fit_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        logger=logger,
        checkpoint_interval=1,
        base_metrics={"model": "linear", "dataset": "toy", "actual_sparsity": 0.0},
        log_batches=True,
        log_interval=1,
    )

    assert result.epochs == 2
    lines = (logger.run_dir / "metrics_epoch.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    batch_lines = (logger.run_dir / "metrics_batch.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(batch_lines) == 4
    assert json.loads(lines[0])["model"] == "linear"
    assert (logger.checkpoints_dir / "epoch_0001.pt").exists()
    assert (logger.checkpoints_dir / "epoch_0002.pt").exists()
