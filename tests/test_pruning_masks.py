import pytest
import torch

from sso.models import build_model
from sso.pruning import (
    apply_mask_to_model,
    compute_sparsity,
    global_topk_mask,
    iter_prunable_named_parameters,
    magnitude_score,
    masked_state_dict,
)


def _config() -> dict:
    return {
        "dataset": {"num_classes": 10},
        "model": {"name": "resnet18", "num_classes": 10},
    }


def test_prunable_parameter_enumeration_defaults_to_conv_linear_weights() -> None:
    model = build_model(_config())
    named_parameters = dict(model.named_parameters())
    prunable = dict(iter_prunable_named_parameters(model))

    assert prunable
    assert set(prunable).issubset(named_parameters)
    assert "conv1.weight" in prunable
    assert "fc.weight" in prunable
    assert "fc.bias" not in prunable
    assert all(name.endswith(".weight") or name == "weight" for name in prunable)
    assert not any("bn" in name.lower() for name in prunable)


def test_magnitude_score_keys_match_prunable_parameters() -> None:
    model = build_model(_config())
    prunable_keys = set(dict(iter_prunable_named_parameters(model)))
    score_dict = magnitude_score(model)

    assert set(score_dict) == prunable_keys
    for name, parameter in iter_prunable_named_parameters(model):
        assert score_dict[name].shape == parameter.shape
        assert score_dict[name].device == parameter.device
        assert score_dict[name].dtype == parameter.dtype


def test_global_topk_mask_shapes_and_sparsity() -> None:
    model = build_model(_config())
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=0.5)

    assert set(mask_dict) == set(score_dict)
    for name, score in score_dict.items():
        mask = mask_dict[name]
        assert mask.shape == score.shape
        assert mask.device == score.device
        assert mask.dtype == score.dtype
        assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    assert compute_sparsity(mask_dict) == pytest.approx(0.5, abs=1e-6)


def test_global_topk_mask_rejects_invalid_sparsity() -> None:
    scores = {"weight": torch.ones(4)}

    with pytest.raises(ValueError):
        global_topk_mask(scores, sparsity=-0.1)

    with pytest.raises(ValueError):
        global_topk_mask(scores, sparsity=1.1)


def test_global_topk_mask_boundaries() -> None:
    scores = {"weight": torch.arange(4, dtype=torch.float32)}

    all_kept = global_topk_mask(scores, sparsity=0.0)
    all_pruned = global_topk_mask(scores, sparsity=1.0)

    assert torch.equal(all_kept["weight"], torch.ones_like(scores["weight"]))
    assert torch.equal(all_pruned["weight"], torch.zeros_like(scores["weight"]))


def test_apply_mask_to_model_zeros_masked_weights() -> None:
    model = build_model(_config())
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=0.5)

    apply_mask_to_model(model, mask_dict)
    named_parameters = dict(model.named_parameters())

    for name, mask in mask_dict.items():
        parameter = named_parameters[name]
        assert torch.count_nonzero(parameter[mask.to(dtype=torch.bool) == 0]).item() == 0


def test_apply_mask_to_model_rejects_unknown_key() -> None:
    model = build_model(_config())

    with pytest.raises(ValueError):
        apply_mask_to_model(model, {"missing.weight": torch.ones(1)})


def test_apply_mask_to_model_rejects_shape_mismatch() -> None:
    model = build_model(_config())

    with pytest.raises(ValueError):
        apply_mask_to_model(model, {"conv1.weight": torch.ones(1)})


def test_masked_state_dict_masks_clone_without_changing_model() -> None:
    model = build_model(_config())
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=0.5)

    masked = masked_state_dict(model, mask_dict)

    assert set(masked) == set(model.state_dict())
    for name, original in original_state.items():
        assert torch.equal(model.state_dict()[name], original)

    for name, mask in mask_dict.items():
        expected = original_state[name] * mask.to(dtype=original_state[name].dtype)
        assert torch.equal(masked[name], expected)
