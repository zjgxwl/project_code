"""Structured slim export for project-local CifarVGG models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from sso.methods import StructureGroup
from sso.models import CifarVGG


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def _assert_state_unchanged(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, state[name]):
            raise RuntimeError(f"export_vgg_slim_model mutated source model state: {name}")


def _validate_vgg_triplets(model: CifarVGG) -> list[tuple[str, nn.Conv2d, nn.BatchNorm2d, nn.ReLU]]:
    if not isinstance(model.features, nn.Sequential):
        raise ValueError("CifarVGG features must be an nn.Sequential.")
    layers = list(model.features)
    if len(layers) == 0 or len(layers) % 3 != 0:
        raise ValueError("CifarVGG features must repeat Conv2d-BatchNorm2d-ReLU triplets.")

    triplets: list[tuple[str, nn.Conv2d, nn.BatchNorm2d, nn.ReLU]] = []
    for index in range(0, len(layers), 3):
        conv = layers[index]
        bn = layers[index + 1]
        relu = layers[index + 2]
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"Expected Conv2d at features.{index}.")
        if not isinstance(bn, nn.BatchNorm2d):
            raise ValueError(f"Expected BatchNorm2d at features.{index + 1}.")
        if not isinstance(relu, nn.ReLU):
            raise ValueError(f"Expected ReLU at features.{index + 2}.")
        if bn.num_features != conv.out_channels:
            raise ValueError(f"BatchNorm2d at features.{index + 1} does not match preceding Conv2d.")
        triplets.append((f"features.{index}", conv, bn, relu))
    return triplets


def _keep_indices_by_layer(
    triplets: Sequence[tuple[str, nn.Conv2d, nn.BatchNorm2d, nn.ReLU]],
    groups: Sequence[StructureGroup],
    group_mask: Mapping[str, int | bool],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    conv_names = {layer_name for layer_name, _conv, _bn, _relu in triplets}
    group_ids_by_layer: dict[str, list[tuple[int, str]]] = {name: [] for name in conv_names}

    for group in list(groups):
        if group.layer_name not in conv_names:
            continue
        if group.group_id not in group_mask:
            raise ValueError(f"Missing group mask for {group.group_id}.")
        group_ids_by_layer[group.layer_name].append((int(group.out_channel), group.group_id))

    keep_indices: dict[str, torch.Tensor] = {}
    for layer_name, conv, _bn, _relu in triplets:
        layer_groups = sorted(group_ids_by_layer[layer_name], key=lambda item: item[0])
        if not layer_groups:
            raise ValueError(f"Missing EGRO groups for Conv2d layer {layer_name}.")
        expected = set(range(int(conv.out_channels)))
        found = {out_channel for out_channel, _group_id in layer_groups}
        if found != expected:
            raise ValueError(f"EGRO groups for {layer_name} do not cover all output channels.")

        kept = [
            out_channel
            for out_channel, group_id in layer_groups
            if int(group_mask[group_id]) == 1
        ]
        if not kept:
            # Stage-1 min_keep_ratio should prevent this. Keep one channel here
            # so Stage-3a export remains defensively valid for malformed masks.
            kept = [layer_groups[0][0]]
        keep_indices[layer_name] = torch.tensor(kept, dtype=torch.long, device=device)
    return keep_indices


def _copy_batch_norm(slim_bn: nn.BatchNorm2d, source_bn: nn.BatchNorm2d, keep: torch.Tensor) -> None:
    slim_bn.weight.data.copy_(source_bn.weight.detach().index_select(0, keep))
    slim_bn.bias.data.copy_(source_bn.bias.detach().index_select(0, keep))
    slim_bn.running_mean.data.copy_(source_bn.running_mean.detach().index_select(0, keep))
    slim_bn.running_var.data.copy_(source_bn.running_var.detach().index_select(0, keep))
    slim_bn.num_batches_tracked.data.copy_(source_bn.num_batches_tracked.detach())


def export_vgg_slim_model(
    model: nn.Module,
    group_mask: Mapping[str, int | bool],
    groups: Sequence[StructureGroup],
    num_classes: int,
    input_shape: tuple[int, ...] | list[int],
) -> CifarVGG:
    """Export a real slim CifarVGG from EGRO Conv2d output-channel groups.

    ``input_shape`` is accepted to keep the export interface explicit for
    Stage-3a smoke tests; the project-local VGG uses adaptive pooling, so the
    final classifier depends only on the last kept channel count.
    """
    if not isinstance(model, CifarVGG):
        raise ValueError("export_vgg_slim_model only supports project-local CifarVGG models.")

    _ = tuple(int(dim) for dim in input_shape)
    was_training = model.training
    source_state = _clone_state(model)
    device = _model_device(model)
    triplets = _validate_vgg_triplets(model)
    keep_indices = _keep_indices_by_layer(triplets, list(groups), dict(group_mask), device)
    slim_channels = [int(keep_indices[layer_name].numel()) for layer_name, _conv, _bn, _relu in triplets]
    slim_model = CifarVGG(channels=slim_channels, num_classes=int(num_classes)).to(device)
    slim_triplets = _validate_vgg_triplets(slim_model)

    try:
        with torch.no_grad():
            previous_keep = torch.arange(triplets[0][1].in_channels, dtype=torch.long, device=device)
            for (layer_name, source_conv, source_bn, _source_relu), (
                _slim_layer_name,
                slim_conv,
                slim_bn,
                _slim_relu,
            ) in zip(triplets, slim_triplets, strict=True):
                output_keep = keep_indices[layer_name]
                conv_weight = source_conv.weight.detach().index_select(0, output_keep)
                conv_weight = conv_weight.index_select(1, previous_keep)
                slim_conv.weight.data.copy_(conv_weight)
                if source_conv.bias is not None and slim_conv.bias is not None:
                    slim_conv.bias.data.copy_(source_conv.bias.detach().index_select(0, output_keep))

                _copy_batch_norm(slim_bn, source_bn, output_keep)
                previous_keep = output_keep

            slim_model.classifier.weight.data.copy_(
                model.classifier.weight.detach().index_select(1, previous_keep)
            )
            if model.classifier.bias is not None and slim_model.classifier.bias is not None:
                slim_model.classifier.bias.data.copy_(model.classifier.bias.detach().clone())
    finally:
        model.train(was_training)
        _assert_state_unchanged(model, source_state)

    slim_model.train(was_training)
    return slim_model
