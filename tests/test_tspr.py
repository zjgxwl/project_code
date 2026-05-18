import pytest
import torch
from torch import nn

from sso.methods import TSPRMethod
from sso.models import build_model
from sso.pruning import global_topk_mask, magnitude_score


def _config() -> dict:
    return {
        "dataset": {"num_classes": 10},
        "model": {"name": "resnet18", "num_classes": 10},
    }


def _first_mask_with_pruned_and_kept(mask_dict: dict[str, torch.Tensor]) -> str:
    for name, mask in mask_dict.items():
        if torch.any(mask == 0) and torch.any(mask != 0):
            return name
    raise AssertionError("Expected a mask with both pruned and kept positions.")


def test_tspr_regularization_and_export_behavior() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = build_model(_config()).to(device)
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=0.9)
    original_masks = {name: mask.detach().clone() for name, mask in mask_dict.items()}

    target_name = _first_mask_with_pruned_and_kept(mask_dict)
    named_parameters = dict(model.named_parameters())
    target_param = named_parameters[target_name]
    pruned_positions = mask_dict[target_name].to(dtype=torch.bool) == 0

    method = TSPRMethod(model, mask_dict, score_dict, lambda0=1e-4)

    with torch.no_grad():
        target_param[pruned_positions] = 0.123
    before_train_param = target_param.detach().clone()

    method.before_train()
    assert torch.equal(target_param.detach(), before_train_param)

    reg_loss = method.regularization_loss(epoch=0, total_epochs=1)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.device == device
    assert reg_loss.item() >= 0.0

    model.zero_grad(set_to_none=True)
    reg_loss.backward()
    grad = target_param.grad
    assert grad is not None
    kept_positions = mask_dict[target_name].to(dtype=torch.bool)
    assert torch.count_nonzero(grad[pruned_positions]).item() > 0
    assert torch.count_nonzero(grad[kept_positions]).item() == 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert torch.count_nonzero(target_param.detach()[pruned_positions]).item() > 0

    model_state_before_export = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    exported = method.export_state_dict()
    assert torch.count_nonzero(exported[target_name][pruned_positions]).item() == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, model_state_before_export[name])

    for name, original_mask in original_masks.items():
        assert torch.equal(method.fixed_mask_dict[name], original_mask)

    assert method.lambda_at(0, 10) == pytest.approx(1e-4)
    assert method.lambda_at(10, 10) == pytest.approx(0.0, abs=1e-12)


def test_tspr_regularization_loss_handles_no_pruned_positions() -> None:
    model = build_model(_config())
    score_dict = magnitude_score(model)
    mask_dict = {name: torch.ones_like(score) for name, score in score_dict.items()}
    method = TSPRMethod(model, mask_dict, score_dict, lambda0=1e-4)

    reg_loss = method.regularization_loss(epoch=0, total_epochs=1)

    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.device == next(model.parameters()).device
    assert reg_loss.item() == 0.0
