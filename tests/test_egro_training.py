import pytest
import torch
from torch import nn

from sso.datasets import build_dataloaders
from sso.methods import EGROMethod, EGROTrainingMethod, TCSMMethod
from sso.models import build_model
from sso.pruning import build_score_dict


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
        "tcsm": {
            "alpha": 0.5,
            "beta": 0.5,
            "eta": 0.05,
            "delta": 1e-12,
            "calibration_batches": 1,
            "sign_batches": 1,
        },
        "egro": {
            "lambda0": 1e-4,
            "flops_reduction": 0.5,
            "min_keep_ratio": 0.2,
            "safety_beta": 0.0,
            "eta_g": 0.05,
            "delta": 1e-12,
            "input_shape": [1, 3, 32, 32],
        },
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


def _build_egro_stage1(model: nn.Module, config: dict, device: torch.device):
    train_loader, _ = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    base_score_dict = build_score_dict(
        model,
        config,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        scorer="magnitude",
    )
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
    tcsm_output = tcsm.run()
    egro_config = config["egro"]
    egro = EGROMethod(
        model=model,
        stable_score_dict=tcsm_output.stable_score_dict,
        input_shape=egro_config["input_shape"],
        flops_reduction=egro_config["flops_reduction"],
        min_keep_ratio=egro_config["min_keep_ratio"],
        safety_beta=egro_config["safety_beta"],
        eta_g=egro_config["eta_g"],
        delta=egro_config["delta"],
    )
    return egro.run()


def _first_dropped_group(output):
    for group in output.groups:
        if output.group_mask[group.group_id] == 0:
            return group
    raise AssertionError("Expected at least one dropped group.")


def test_egro_training_regularization_and_group_masked_export() -> None:
    torch.manual_seed(42)
    config = _config()
    device = torch.device("cpu")
    model = build_model(config).to(device)
    egro_output = _build_egro_stage1(model, config, device)
    original_group_mask = dict(egro_output.group_mask)

    method = EGROTrainingMethod(
        model=model,
        groups=egro_output.groups,
        group_mask=egro_output.group_mask,
        group_omega=egro_output.group_omega,
        lambda0=1e-4,
        delta=1e-12,
    )

    state_before_train = _state_clone(model)
    method.before_train()
    _assert_state_unchanged(model, state_before_train)

    dropped_group = _first_dropped_group(egro_output)
    modules = dict(model.named_modules())
    dropped_module = modules[dropped_group.layer_name]
    assert isinstance(dropped_module, nn.Conv2d)
    with torch.no_grad():
        dropped_module.weight[dropped_group.out_channel].add_(0.123)

    reg_loss = method.regularization_loss(epoch=0, total_epochs=1)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.device == device
    assert reg_loss.item() >= 0.0

    model.zero_grad(set_to_none=True)
    reg_loss.backward()
    grad = dropped_module.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[dropped_group.out_channel]).item() > 0
    if dropped_module.bias is not None:
        assert dropped_module.bias.grad is None

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert torch.count_nonzero(dropped_module.weight.detach()[dropped_group.out_channel]).item() > 0

    state_before_export = _state_clone(model)
    exported = method.export_group_masked_state_dict()
    weight_key = f"{dropped_group.layer_name}.weight"
    assert torch.count_nonzero(exported[weight_key][dropped_group.out_channel]).item() == 0
    if dropped_module.bias is not None:
        bias_key = f"{dropped_group.layer_name}.bias"
        assert torch.count_nonzero(exported[bias_key][dropped_group.out_channel]).item() == 0
    _assert_state_unchanged(model, state_before_export)

    assert method.fixed_group_mask == original_group_mask
    assert method.lambda_at(0, 10) == pytest.approx(1e-4)
    assert method.lambda_at(10, 10) == pytest.approx(0.0, abs=1e-12)
