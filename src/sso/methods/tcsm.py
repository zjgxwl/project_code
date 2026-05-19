"""Chapter 4 TCSM minimal engineering loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from sso.pruning import compute_sparsity, global_topk_mask, iter_prunable_named_parameters
from .base import BaseSparseMethod


@dataclass
class TCSMOutput:
    """Outputs produced by the minimal TCSM pipeline."""

    base_score_dict: dict[str, torch.Tensor]
    calibration_score_dict: dict[str, torch.Tensor]
    sign_consistency_dict: dict[str, torch.Tensor]
    stable_score_dict: dict[str, torch.Tensor]
    stable_mask_dict: dict[str, torch.Tensor]
    stable_omega_dict: dict[str, torch.Tensor]
    metrics: dict[str, float]


class TCSMMethod(BaseSparseMethod):
    """Topology-consistency sparse mask modeling minimal implementation."""

    def __init__(
        self,
        model: nn.Module,
        base_score_dict: Mapping[str, torch.Tensor],
        dataloader: Iterable | None,
        criterion: nn.Module | None,
        device: torch.device | str | None,
        sparsity: float,
        alpha: float = 0.5,
        beta: float = 0.5,
        eta: float = 0.05,
        delta: float = 1e-12,
        calibration_batches: int = 1,
        sign_batches: int = 1,
    ) -> None:
        super().__init__()
        if dataloader is None:
            raise ValueError("TCSMMethod requires a dataloader.")
        if criterion is None:
            raise ValueError("TCSMMethod requires a criterion.")
        if device is None:
            raise ValueError("TCSMMethod requires a device.")

        self.model = model
        self.base_score_dict = {
            name: score.detach().clone()
            for name, score in base_score_dict.items()
        }
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = torch.device(device)
        self.sparsity = float(sparsity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = float(eta)
        self.delta = float(delta)
        self.calibration_batches = int(calibration_batches)
        self.sign_batches = int(sign_batches)

    def _state_clone(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _restore_state(self, state: Mapping[str, torch.Tensor], was_training: bool) -> None:
        self.model.load_state_dict(state, strict=True)
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)

    def _prunable_items(self) -> list[tuple[str, nn.Parameter]]:
        return list(iter_prunable_named_parameters(self.model))

    def _bounded_batches(self, max_batches: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            if batch_idx >= max_batches:
                break
            yield inputs.to(self.device), targets.to(self.device)

    def _average_loss(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        total_loss: torch.Tensor | None = None
        for inputs, targets in batches:
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            raise ValueError("TCSMMethod received no batches from dataloader.")
        return total_loss / len(batches)

    def _grad_dict_for_batches(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor | None]:
        prunable_items = self._prunable_items()
        loss = self._average_loss(batches)
        grads = torch.autograd.grad(
            loss,
            [parameter for _name, parameter in prunable_items],
            allow_unused=True,
        )
        return {
            name: grad
            for (name, _parameter), grad in zip(prunable_items, grads, strict=True)
        }

    def compute_calibration_score(self) -> dict[str, torch.Tensor]:
        """Compute SNIP-style calibration scores over D_cal."""
        was_training = self.model.training
        saved_state = self._state_clone()
        self.model.zero_grad(set_to_none=True)
        self.model.train()

        try:
            batches = list(self._bounded_batches(self.calibration_batches))
            grad_dict = self._grad_dict_for_batches(batches)
            score_dict: dict[str, torch.Tensor] = {}
            for name, parameter in self._prunable_items():
                grad = grad_dict.get(name)
                if grad is None:
                    score_dict[name] = torch.zeros_like(parameter)
                else:
                    score_dict[name] = (parameter * grad).detach().abs().clone()
            return score_dict
        finally:
            self._restore_state(saved_state, was_training)

    def _sign_batches_pair(self) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        needed = max(1, self.sign_batches)
        batches = list(self._bounded_batches(needed * 2))
        if len(batches) >= needed * 2:
            return batches[:needed], batches[needed:needed * 2]
        first = batches[:needed]
        second = list(self._bounded_batches(needed))
        return first, second

    def compute_sign_consistency(self) -> dict[str, torch.Tensor]:
        """Compute per-parameter gradient sign consistency across two views."""
        was_training = self.model.training
        saved_state = self._state_clone()
        self.model.zero_grad(set_to_none=True)
        self.model.train()

        try:
            batches_a, batches_b = self._sign_batches_pair()
            grad_a = self._grad_dict_for_batches(batches_a)
            self.model.zero_grad(set_to_none=True)
            grad_b = self._grad_dict_for_batches(batches_b)

            consistency: dict[str, torch.Tensor] = {}
            for name, parameter in self._prunable_items():
                first = grad_a.get(name)
                second = grad_b.get(name)
                if first is None or second is None:
                    consistency[name] = torch.zeros_like(parameter)
                else:
                    consistency[name] = (torch.sign(first) == torch.sign(second)).to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
            return consistency
        finally:
            self._restore_state(saved_state, was_training)

    def _global_normalize(self, score_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        total = sum(score.detach().abs().sum() for score in score_dict.values())
        return {
            name: score.detach().abs().clone() / (total.to(score.device) + self.delta)
            for name, score in score_dict.items()
        }

    def build_stable_score(
        self,
        calibration_score_dict: Mapping[str, torch.Tensor],
        sign_consistency_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse base, calibration, and sign consistency into stable scores."""
        normalized_base = self._global_normalize(self.base_score_dict)
        normalized_calibration = self._global_normalize(calibration_score_dict)

        stable_score: dict[str, torch.Tensor] = {}
        for name, base_score in normalized_base.items():
            calibration = normalized_calibration[name].to(
                device=base_score.device,
                dtype=base_score.dtype,
            )
            sign_consistency = sign_consistency_dict[name].to(
                device=base_score.device,
                dtype=base_score.dtype,
            )
            score = (1.0 - self.alpha) * base_score + self.alpha * calibration * (
                1.0 + self.beta * sign_consistency
            )
            stable_score[name] = torch.clamp(score.detach().clone(), min=0.0)
        return stable_score

    def build_stable_mask(
        self,
        stable_score_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build the stable global Top-K mask."""
        return global_topk_mask(stable_score_dict, self.sparsity)

    def build_stable_omega(
        self,
        stable_score_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build finite positive stable omega tensors from stable scores."""
        omega_dict: dict[str, torch.Tensor] = {}
        for name, score in stable_score_dict.items():
            score_abs = score.detach().abs().clone()
            score_min = score_abs.min()
            score_max = score_abs.max()
            if torch.isclose(score_max, score_min):
                normalized = torch.zeros_like(score_abs)
            else:
                normalized = (score_abs - score_min) / (score_max - score_min + self.delta)
            floor = self.eta + (1.0 - self.eta) * normalized
            omega_dict[name] = torch.clamp(1.0 / floor, max=1e6)
        return omega_dict

    def _base_stable_jaccard(
        self,
        stable_mask_dict: Mapping[str, torch.Tensor],
    ) -> float:
        base_mask = global_topk_mask(self.base_score_dict, self.sparsity)
        intersections = 0
        unions = 0
        for name, base in base_mask.items():
            stable = stable_mask_dict[name]
            base_bool = base.detach().to(dtype=torch.bool).flatten().cpu()
            stable_bool = stable.detach().to(dtype=torch.bool).flatten().cpu()
            intersections += int(torch.logical_and(base_bool, stable_bool).sum().item())
            unions += int(torch.logical_or(base_bool, stable_bool).sum().item())
        if unions == 0:
            return 1.0
        return intersections / unions

    def _mean_value(self, tensor_dict: Mapping[str, torch.Tensor]) -> float:
        total_sum = sum(float(tensor.detach().sum().cpu().item()) for tensor in tensor_dict.values())
        total_count = sum(tensor.numel() for tensor in tensor_dict.values())
        if total_count == 0:
            return 0.0
        return total_sum / total_count

    def run(self) -> TCSMOutput:
        """Run the minimal TCSM data flow."""
        calibration_score = self.compute_calibration_score()
        sign_consistency = self.compute_sign_consistency()
        stable_score = self.build_stable_score(calibration_score, sign_consistency)
        stable_mask = self.build_stable_mask(stable_score)
        stable_omega = self.build_stable_omega(stable_score)
        metrics = {
            "sparsity": compute_sparsity(stable_mask),
            "mean_stable_score": self._mean_value(stable_score),
            "mean_stable_omega": self._mean_value(stable_omega),
            "base_stable_jaccard": self._base_stable_jaccard(stable_mask),
        }
        return TCSMOutput(
            base_score_dict={
                name: score.detach().clone()
                for name, score in self.base_score_dict.items()
            },
            calibration_score_dict=calibration_score,
            sign_consistency_dict=sign_consistency,
            stable_score_dict=stable_score,
            stable_mask_dict=stable_mask,
            stable_omega_dict=stable_omega,
            metrics=metrics,
        )
