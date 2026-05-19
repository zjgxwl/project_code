import math
from collections import Counter

import torch
from torch import nn

from sso.datasets import build_dataloaders
from sso.methods import EGROMethod, EGROOutput, TCSMMethod
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


def test_egro_stage1_outputs_group_selection_metrics_without_mutation() -> None:
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
    original_stable_score = {
        name: score.detach().clone()
        for name, score in tcsm_output.stable_score_dict.items()
    }

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
    output = egro.run()

    assert isinstance(output, EGROOutput)
    assert output.groups
    assert set(output.group_scores) == {group.group_id for group in output.groups}
    assert set(output.group_mask) == {group.group_id for group in output.groups}
    assert set(output.group_omega) == {group.group_id for group in output.groups}

    for score in output.group_scores.values():
        assert math.isfinite(score)
        assert score >= 0.0
    for mask in output.group_mask.values():
        assert mask in {0, 1}
    for omega in output.group_omega.values():
        assert math.isfinite(omega)
        assert omega > 0.0

    layer_counts = Counter(group.layer_name for group in output.groups)
    kept_counts = Counter(
        group.layer_name
        for group in output.groups
        if output.group_mask[group.group_id] == 1
    )
    min_keep_ratio = float(egro_config["min_keep_ratio"])
    for layer_name, count in layer_counts.items():
        min_keep = max(1, min(count, math.ceil(min_keep_ratio * count)))
        assert kept_counts[layer_name] >= min_keep

    assert 0.0 <= output.metrics["flops_reduction"] <= 1.0
    assert 0.0 <= output.metrics["param_reduction"] <= 1.0
    assert output.metrics["total_groups"] == len(output.groups)
    assert output.metrics["kept_groups"] + output.metrics["dropped_groups"] == len(output.groups)

    _assert_state_unchanged(model, original_state)
    for name, score in original_stable_score.items():
        assert torch.equal(tcsm_output.stable_score_dict[name], score)
