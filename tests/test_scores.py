import copy

import pytest
import torch
from torch import nn

from sso.datasets import build_dataloaders
from sso.models import build_model
from sso.pruning import (
    build_score_dict,
    iter_prunable_named_parameters,
    magnitude_score,
    snip_score,
    synflow_score,
)


def _config() -> dict:
    return {
        "seed": 42,
        "dataset": {
            "name": "cifar10",
            "use_fake_data": True,
            "num_classes": 10,
            "image_size": 32,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {"name": "resnet18", "num_classes": 10},
        "pruning": {"scorer": "magnitude", "score_batches": 1},
    }


def _assert_score_shapes_and_nonnegative(
    model: nn.Module,
    score_dict: dict[str, torch.Tensor],
) -> None:
    prunable = dict(iter_prunable_named_parameters(model))
    assert set(score_dict) == set(prunable)
    for name, parameter in prunable.items():
        score = score_dict[name]
        assert score.shape == parameter.shape
        assert score.device == parameter.device
        assert torch.all(score >= 0)


def _state_clone(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def _assert_state_unchanged(
    model: nn.Module,
    original_state: dict[str, torch.Tensor],
) -> None:
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def _assert_no_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        assert parameter.grad is None


def test_magnitude_score_keys_shapes_and_values() -> None:
    model = build_model(_config())
    score_dict = magnitude_score(model)
    _assert_score_shapes_and_nonnegative(model, score_dict)


def test_snip_score_keys_shapes_nonnegative_and_restores_model() -> None:
    config = _config()
    model = build_model(config)
    model.eval()
    was_training = model.training
    original_state = _state_clone(model)
    train_loader, _ = build_dataloaders(config)

    score_dict = snip_score(
        model,
        dataloader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        max_batches=2,
    )

    _assert_score_shapes_and_nonnegative(model, score_dict)
    _assert_state_unchanged(model, original_state)
    _assert_no_gradients(model)
    assert model.training == was_training


def test_synflow_score_keys_shapes_nonnegative_and_restores_model() -> None:
    config = _config()
    model = build_model(config)
    model.train()
    was_training = model.training
    original_state = _state_clone(model)

    score_dict = synflow_score(
        model,
        input_shape=(1, 3, 32, 32),
        device=torch.device("cpu"),
    )

    _assert_score_shapes_and_nonnegative(model, score_dict)
    _assert_state_unchanged(model, original_state)
    _assert_no_gradients(model)
    assert model.training == was_training


def test_build_score_dict_selects_supported_scorers_without_mutating_config() -> None:
    base_config = _config()

    for scorer in ("magnitude", "snip", "synflow"):
        config = copy.deepcopy(base_config)
        original_config = copy.deepcopy(config)
        config["pruning"]["scorer"] = scorer
        model = build_model(config)
        train_loader, _ = build_dataloaders(config)
        score_dict = build_score_dict(
            model,
            config,
            dataloader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )

        _assert_score_shapes_and_nonnegative(model, score_dict)
        assert config == {**original_config, "pruning": {**original_config["pruning"], "scorer": scorer}}


def test_build_score_dict_rejects_unimplemented_and_unknown_scorers() -> None:
    config = _config()
    model = build_model(config)

    with pytest.raises(NotImplementedError):
        build_score_dict(model, config, scorer="grasp")

    with pytest.raises(NotImplementedError):
        build_score_dict(model, config, scorer="ep")

    with pytest.raises(ValueError):
        build_score_dict(model, config, scorer="unknown")
