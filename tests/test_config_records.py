import pytest

from sso.utils import build_experiment_record, validate_experiment_config, validate_experiment_record


def _config() -> dict:
    return {
        "seed": 7,
        "dataset": {
            "name": "cifar10",
            "data_dir": "outputs/thesis_real_data",
            "num_classes": 10,
        },
        "model": {"name": "resnet18", "num_classes": 10},
        "training": {"epochs": 1},
        "pruning": {"sparsity": 0.9},
    }


def test_validate_experiment_config_accepts_common_debug_config() -> None:
    config = _config()

    assert validate_experiment_config(config) is config


def test_validate_experiment_config_rejects_bad_ranges() -> None:
    config = _config()
    config["pruning"]["sparsity"] = 1.0

    with pytest.raises(ValueError, match="pruning.sparsity"):
        validate_experiment_config(config)


def test_build_experiment_record_adds_standard_fields() -> None:
    record = build_experiment_record(
        script="train_tspr.py",
        method="tspr",
        config=_config(),
        metrics={"val_acc": 0.25},
        run_name="demo",
        extra={"scorer": "snip"},
    )

    validate_experiment_record(record)
    assert record["schema_version"] == 1
    assert record["script"] == "train_tspr.py"
    assert record["method"] == "tspr"
    assert record["model"] == "resnet18"
    assert record["dataset"] == "cifar10"
    assert record["seed"] == 7
    assert record["metrics"]["val_acc"] == pytest.approx(0.25)
    assert record["scorer"] == "snip"
