import pytest
import torch
from torch import nn

from sso.datasets import build_dataloaders
from sso.methods import TCSMMethod, TCSMOutput
from sso.models import build_model
from sso.pruning import build_score_dict, compute_sparsity, iter_prunable_named_parameters


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
        "pruning": {"scorer": "magnitude", "sparsity": 0.9, "score_batches": 1},
    }


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


def _assert_key_shape_match(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> None:
    assert set(candidate) == set(reference)
    for name, tensor in reference.items():
        assert candidate[name].shape == tensor.shape


def test_tcsm_run_outputs_stable_scores_masks_and_metrics() -> None:
    torch.manual_seed(42)
    config = _config()
    device = torch.device("cpu")
    train_loader, _ = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    model = build_model(config).to(device)
    original_state = _state_clone(model)

    base_score_dict = build_score_dict(
        model,
        config,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        scorer="magnitude",
    )
    original_base_score = {
        name: score.detach().clone()
        for name, score in base_score_dict.items()
    }

    tcsm = TCSMMethod(
        model=model,
        base_score_dict=base_score_dict,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        sparsity=0.9,
        calibration_batches=1,
        sign_batches=1,
    )
    output = tcsm.run()

    assert isinstance(output, TCSMOutput)
    prunable = dict(iter_prunable_named_parameters(model))
    _assert_key_shape_match(prunable, output.calibration_score_dict)
    _assert_key_shape_match(prunable, output.sign_consistency_dict)
    _assert_key_shape_match(prunable, output.stable_score_dict)
    _assert_key_shape_match(prunable, output.stable_mask_dict)
    _assert_key_shape_match(prunable, output.stable_omega_dict)

    for score in output.sign_consistency_dict.values():
        assert torch.all(score >= 0)
        assert torch.all(score <= 1)
    for score in output.stable_score_dict.values():
        assert torch.all(score >= 0)
    for omega in output.stable_omega_dict.values():
        assert torch.all(torch.isfinite(omega))
        assert torch.all(omega > 0)

    assert compute_sparsity(output.stable_mask_dict) == pytest.approx(0.9, abs=1e-6)
    assert 0.0 <= output.metrics["base_stable_jaccard"] <= 1.0
    assert output.metrics["sparsity"] == pytest.approx(0.9, abs=1e-6)
    assert output.metrics["mean_stable_score"] >= 0.0
    assert output.metrics["mean_stable_omega"] > 0.0

    _assert_state_unchanged(model, original_state)
    for name, score in original_base_score.items():
        assert torch.equal(base_score_dict[name], score)
