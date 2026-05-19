import copy

import pytest
import torch
from torch import nn

from sso.datasets import build_dataloaders
from sso.export import StructuredExportArtifact, export_slim_artifact, export_vgg_slim_artifact, export_vgg_slim_model
from sso.methods import EGROMethod, TCSMMethod
from sso.models import CifarVGG, build_model
from sso.pruning import build_score_dict


def _config() -> dict:
    return {
        "seed": 42,
        "dataset": {
            "name": "cifar10",
            "data_dir": "outputs/thesis_real_data",
            "download": False,
            "num_classes": 10,
            "image_size": 32,
            "batch_size": 4,
            "num_workers": 0,
            "train_subset_size": 8,
            "val_subset_size": 4,
        },
        "model": {"name": "vgg11_bn", "num_classes": 10},
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


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_egro_output(model: nn.Module, config: dict, device: torch.device):
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


def _conv_bn_pairs(model: nn.Module) -> list[tuple[nn.Conv2d, nn.BatchNorm2d]]:
    layers = list(model.features)
    pairs: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = []
    for index in range(0, len(layers), 3):
        assert isinstance(layers[index], nn.Conv2d)
        assert isinstance(layers[index + 1], nn.BatchNorm2d)
        pairs.append((layers[index], layers[index + 1]))
    return pairs


def test_export_vgg_slim_model_forwards_and_preserves_source_model() -> None:
    torch.manual_seed(42)
    config = _config()
    device = torch.device("cpu")
    model = build_model(config).to(device)
    model.eval()
    original_training = model.training
    original_state = _state_clone(model)
    egro_output = _build_egro_output(model, config, device)
    original_group_mask = copy.deepcopy(egro_output.group_mask)

    slim_model = export_vgg_slim_model(
        model=model,
        group_mask=egro_output.group_mask,
        groups=egro_output.groups,
        num_classes=10,
        input_shape=config["egro"]["input_shape"],
    )

    inputs = torch.randn(4, 3, 32, 32, device=device)
    logits = slim_model(inputs)
    assert logits.shape == (4, 10)
    assert next(slim_model.parameters()).device == device
    assert _count_parameters(slim_model) < _count_parameters(model)

    last_conv_out_channels = None
    for conv, bn in _conv_bn_pairs(slim_model):
        assert conv.out_channels >= 1
        assert bn.num_features == conv.out_channels
        last_conv_out_channels = conv.out_channels
    assert isinstance(slim_model.classifier, nn.Linear)
    assert slim_model.classifier.in_features == last_conv_out_channels

    assert model.training == original_training
    _assert_state_unchanged(model, original_state)
    assert egro_output.group_mask == original_group_mask


def test_export_vgg_slim_artifact_reports_real_structure_metadata() -> None:
    torch.manual_seed(42)
    config = _config()
    device = torch.device("cpu")
    model = build_model(config).to(device)
    original_state = _state_clone(model)
    egro_output = _build_egro_output(model, config, device)

    artifact = export_vgg_slim_artifact(
        model=model,
        group_mask=egro_output.group_mask,
        groups=egro_output.groups,
        num_classes=10,
        input_shape=config["egro"]["input_shape"],
    )

    assert isinstance(artifact, StructuredExportArtifact)
    assert isinstance(artifact.model, CifarVGG)
    metadata = artifact.metadata
    assert metadata["export_type"] == "vgg_slim"
    assert metadata["original_params"] > metadata["slim_params"]
    assert 0.0 < metadata["param_reduction"] < 1.0
    assert metadata["original_conv_flops"] > metadata["slim_conv_flops"]
    assert 0.0 < metadata["flops_reduction"] < 1.0
    assert metadata["layer_keep"]
    for values in metadata["layer_keep"].values():
        assert 0.0 < values["keep_ratio"] <= 1.0
        assert values["kept_channels"] <= values["total_channels"]
    _assert_state_unchanged(model, original_state)


def test_export_slim_artifact_rejects_resnet_until_shortcut_rewrite_exists() -> None:
    config = _config()
    config["model"] = {"name": "resnet18", "num_classes": 10}
    model = build_model(config)

    with pytest.raises(NotImplementedError, match="ResNet shortcut"):
        export_slim_artifact(
            model=model,
            group_mask={},
            groups=[],
            num_classes=10,
            input_shape=(1, 3, 32, 32),
        )


def test_export_vgg_slim_model_rejects_invalid_vgg_structure() -> None:
    model = CifarVGG(channels=[4], num_classes=10)
    model.features = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(4),
    )

    groups = []
    group_mask = {}
    with pytest.raises(ValueError):
        export_vgg_slim_model(
            model=model,
            group_mask=group_mask,
            groups=groups,
            num_classes=10,
            input_shape=(1, 3, 32, 32),
        )
