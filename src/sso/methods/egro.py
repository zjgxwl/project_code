"""Chapter 5 EGRO Stage-1 structural group selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .base import BaseSparseMethod


@dataclass(frozen=True)
class StructureGroup:
    """A Conv2d output-channel group used by EGRO Stage-1."""

    group_id: str
    layer_name: str
    param_name: str
    out_channel: int
    channel_index: int
    num_params: int
    flops: float


@dataclass
class EGROOutput:
    """Outputs produced by the minimal EGRO Stage-1 pipeline."""

    group_scores: dict[str, float]
    group_mask: dict[str, int]
    group_omega: dict[str, float]
    metrics: dict[str, float]
    groups: list[StructureGroup]


class EGROMethod(BaseSparseMethod):
    """Efficient group reorganization Stage-1 structural selection.

    This minimal engineering loop only models Conv2d output-channel groups and
    produces group-level masks. It does not perform dependency-consistent
    structure rewriting, BatchNorm/channel synchronization, model export, or
    group-level retraining. ``safety_beta`` is reserved for future layer-safety
    extensions and is not used in Stage-1.
    """

    def __init__(
        self,
        model: nn.Module,
        stable_score_dict: Mapping[str, torch.Tensor],
        input_shape: tuple[int, ...] | list[int],
        flops_reduction: float = 0.5,
        min_keep_ratio: float = 0.2,
        safety_beta: float = 0.0,
        eta_g: float = 0.05,
        delta: float = 1e-12,
    ) -> None:
        super().__init__()
        self.model = model
        self.stable_score_dict = {
            name: score.detach().clone()
            for name, score in stable_score_dict.items()
        }
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.flops_reduction = float(flops_reduction)
        self.min_keep_ratio = float(min_keep_ratio)
        self.safety_beta = float(safety_beta)
        self.eta_g = float(eta_g)
        self.delta = float(delta)

        if not 0.0 <= self.flops_reduction <= 1.0:
            raise ValueError("flops_reduction must be in [0.0, 1.0].")
        if not 0.0 <= self.min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be in [0.0, 1.0].")
        if not 0.0 < self.eta_g <= 1.0:
            raise ValueError("eta_g must be in (0.0, 1.0].")

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _validate_finite(self, value: float, name: str) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
        return value

    def _conv_output_shapes(self) -> dict[str, tuple[int, int]]:
        output_shapes: dict[str, tuple[int, int]] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        was_training = self.model.training

        def make_hook(layer_name: str):
            def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                if output.ndim < 4:
                    raise ValueError(f"Conv2d layer {layer_name} produced non-4D output.")
                output_shapes[layer_name] = (int(output.shape[-2]), int(output.shape[-1]))

            return hook

        try:
            for layer_name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(make_hook(layer_name)))

            self.model.eval()
            device = self._model_device()
            dummy = torch.ones(self.input_shape, device=device)
            with torch.no_grad():
                self.model(dummy)
        finally:
            for handle in hooks:
                handle.remove()
            self.model.train(was_training)

        return output_shapes

    def build_structure_groups(self) -> list[StructureGroup]:
        """Build Conv2d output-channel groups with params and FLOPs estimates."""
        output_shapes = self._conv_output_shapes()
        groups: list[StructureGroup] = []

        for layer_name, module in self.model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if layer_name not in output_shapes:
                raise ValueError(f"Missing output shape for Conv2d layer {layer_name}.")

            h_out, w_out = output_shapes[layer_name]
            param_name = f"{layer_name}.weight" if layer_name else "weight"
            per_channel_weight_params = int(module.weight[0].numel())
            per_channel_bias_params = 1 if module.bias is not None else 0
            c_in = int(module.in_channels)
            k_h, k_w = module.kernel_size
            group_flops = float(2 * h_out * w_out * c_in * int(k_h) * int(k_w))
            self._validate_finite(group_flops, f"FLOPs for {layer_name}")

            for out_channel in range(int(module.out_channels)):
                group_id = f"{layer_name}:{out_channel}"
                groups.append(
                    StructureGroup(
                        group_id=group_id,
                        layer_name=layer_name,
                        param_name=param_name,
                        out_channel=out_channel,
                        channel_index=out_channel,
                        num_params=per_channel_weight_params + per_channel_bias_params,
                        flops=group_flops,
                    )
                )

        return groups

    def build_group_scores(self, groups: list[StructureGroup]) -> dict[str, float]:
        """Aggregate stable weight scores into Conv2d output-channel scores."""
        group_scores: dict[str, float] = {}
        for group in groups:
            if group.param_name not in self.stable_score_dict:
                raise ValueError(f"Missing stable score for Conv2d parameter {group.param_name}.")
            score = self.stable_score_dict[group.param_name]
            if group.out_channel >= score.shape[0]:
                raise ValueError(f"Group {group.group_id} is outside score tensor shape {tuple(score.shape)}.")
            value = float(score[group.out_channel].detach().abs().mean().cpu().item())
            group_scores[group.group_id] = self._validate_finite(value, f"group score {group.group_id}")
        return group_scores

    def _min_keep_by_layer(self, groups: list[StructureGroup]) -> dict[str, int]:
        layer_counts: dict[str, int] = defaultdict(int)
        for group in groups:
            layer_counts[group.layer_name] += 1
        return {
            layer_name: max(1, min(count, math.ceil(self.min_keep_ratio * count)))
            for layer_name, count in layer_counts.items()
        }

    def build_group_mask(
        self,
        groups: list[StructureGroup],
        group_scores: Mapping[str, float],
    ) -> dict[str, int]:
        """Build a greedy group mask under FLOPs budget and layer safety lower bound."""
        group_mask = {group.group_id: 1 for group in groups}
        kept_by_layer: dict[str, int] = defaultdict(int)
        for group in groups:
            kept_by_layer[group.layer_name] += 1

        min_keep_by_layer = self._min_keep_by_layer(groups)
        total_flops = sum(group.flops for group in groups)
        target_drop_flops = self.flops_reduction * total_flops
        dropped_flops = 0.0

        sorted_groups = sorted(
            groups,
            key=lambda group: (group_scores[group.group_id], group.layer_name, group.out_channel),
        )
        for group in sorted_groups:
            if dropped_flops >= target_drop_flops:
                break
            if kept_by_layer[group.layer_name] <= min_keep_by_layer[group.layer_name]:
                continue
            group_mask[group.group_id] = 0
            kept_by_layer[group.layer_name] -= 1
            dropped_flops += group.flops

        return group_mask

    def build_group_omega(
        self,
        groups: list[StructureGroup],
        group_scores: Mapping[str, float],
    ) -> dict[str, float]:
        """Build finite positive group omega values from per-layer score ranks."""
        groups_by_layer: dict[str, list[StructureGroup]] = defaultdict(list)
        for group in groups:
            groups_by_layer[group.layer_name].append(group)

        omega: dict[str, float] = {}
        for layer_groups in groups_by_layer.values():
            scores = [group_scores[group.group_id] for group in layer_groups]
            score_min = min(scores)
            score_max = max(scores)
            for group in layer_groups:
                if math.isclose(score_max, score_min, rel_tol=0.0, abs_tol=self.delta):
                    normalized = 0.0
                else:
                    normalized = (group_scores[group.group_id] - score_min) / (score_max - score_min + self.delta)
                floor = self.eta_g + (1.0 - self.eta_g) * normalized
                value = min(1.0 / floor, 1e6)
                omega[group.group_id] = self._validate_finite(value, f"group omega {group.group_id}")
                if omega[group.group_id] <= 0.0:
                    raise ValueError(f"group omega {group.group_id} must be positive.")
        return omega

    def _build_metrics(self, groups: list[StructureGroup], group_mask: Mapping[str, int]) -> dict[str, float]:
        total_groups = len(groups)
        kept_groups = sum(int(group_mask[group.group_id]) for group in groups)
        dropped_groups = total_groups - kept_groups
        total_params = sum(group.num_params for group in groups)
        kept_params = sum(group.num_params for group in groups if group_mask[group.group_id])
        total_flops = sum(group.flops for group in groups)
        kept_flops = sum(group.flops for group in groups if group_mask[group.group_id])

        param_reduction = 0.0 if total_params == 0 else 1.0 - (kept_params / total_params)
        actual_flops_reduction = 0.0 if total_flops == 0.0 else 1.0 - (kept_flops / total_flops)

        metrics = {
            "total_groups": float(total_groups),
            "kept_groups": float(kept_groups),
            "dropped_groups": float(dropped_groups),
            "total_conv_params": float(total_params),
            "kept_conv_params": float(kept_params),
            "param_reduction": param_reduction,
            "total_conv_flops": total_flops,
            "kept_conv_flops": kept_flops,
            "flops_reduction": actual_flops_reduction,
            "target_flops_reduction": self.flops_reduction,
        }
        for name, value in metrics.items():
            self._validate_finite(float(value), f"metric {name}")
        return metrics

    def run(self) -> EGROOutput:
        """Run the EGRO Stage-1 structural group selection pipeline."""
        groups = self.build_structure_groups()
        group_scores = self.build_group_scores(groups)
        group_mask = self.build_group_mask(groups, group_scores)
        group_omega = self.build_group_omega(groups, group_scores)
        metrics = self._build_metrics(groups, group_mask)
        return EGROOutput(
            group_scores=group_scores,
            group_mask=group_mask,
            group_omega=group_omega,
            metrics=metrics,
            groups=groups,
        )
