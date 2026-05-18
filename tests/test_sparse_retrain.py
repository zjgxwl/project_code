import pytest
import torch
from torch import nn

from sso.methods import StandardSparseRetrainingMethod
from sso.models import build_model
from sso.pruning import compute_sparsity, global_topk_mask, magnitude_score


def _config() -> dict:
    return {
        "dataset": {"num_classes": 10},
        "model": {"name": "resnet18", "num_classes": 10},
    }


def _assert_masked_values_are_zero(model: nn.Module, mask_dict: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    for name, mask in mask_dict.items():
        parameter = named_parameters[name]
        pruned_positions = mask.to(device=parameter.device, dtype=torch.bool) == 0
        assert torch.count_nonzero(parameter.detach()[pruned_positions]).item() == 0


def _assert_masked_gradients_are_zero(model: nn.Module, mask_dict: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    for name, mask in mask_dict.items():
        parameter = named_parameters[name]
        if parameter.grad is None:
            continue
        pruned_positions = mask.to(device=parameter.grad.device, dtype=torch.bool) == 0
        assert torch.count_nonzero(parameter.grad.detach()[pruned_positions]).item() == 0


def test_standard_sparse_retraining_keeps_weights_and_gradients_masked() -> None:
    torch.manual_seed(42)
    target_sparsity = 0.9
    model = build_model(_config())
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=target_sparsity)
    original_mask_dict = {name: mask.detach().clone() for name, mask in mask_dict.items()}

    method = StandardSparseRetrainingMethod(model, mask_dict)
    initial_method_masks = method.mask_state_dict()

    method.before_train()
    _assert_masked_values_are_zero(model, mask_dict)

    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()
    method.mask_gradients()
    _assert_masked_gradients_are_zero(model, mask_dict)

    optimizer.step()
    method.after_optimizer_step()
    _assert_masked_values_are_zero(model, mask_dict)

    for name, original_mask in original_mask_dict.items():
        assert torch.equal(mask_dict[name], original_mask)
        assert torch.equal(method.mask_state_dict()[name], initial_method_masks[name])

    assert compute_sparsity(mask_dict) == pytest.approx(target_sparsity, abs=1e-6)

    exported = method.export_state_dict()
    assert set(exported) == set(model.state_dict())
    for name, mask in mask_dict.items():
        expected = model.state_dict()[name] * mask.to(dtype=model.state_dict()[name].dtype)
        assert torch.equal(exported[name], expected)
