"""Chapter 4 TCSM minimal engineering loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from sso.metrics import mask_jaccard, score_rank_correlation
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
    metrics: dict[str, object]


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
        calibration_mode: str = "tcsm",
        stability_repeats: int = 1,
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
        self.calibration_mode = calibration_mode.lower().replace("-", "_")
        self.stability_repeats = int(stability_repeats)

        supported_modes = {"tcsm", "base_only", "random_subset", "full_data"}
        if self.calibration_mode not in supported_modes:
            raise ValueError(f"Unsupported TCSM calibration_mode: {calibration_mode}")
        if self.calibration_batches < 0:
            raise ValueError("calibration_batches must be non-negative.")
        if self.sign_batches < 0:
            raise ValueError("sign_batches must be non-negative.")
        if self.stability_repeats < 1:
            raise ValueError("stability_repeats must be positive.")

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

    def _bounded_batches(self, max_batches: int | None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            yield inputs.to(self.device), targets.to(self.device)

    def _clone_like_scores(self, value: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            name: torch.full_like(score, fill_value=value)
            for name, score in self.base_score_dict.items()
        }

    def _clone_base_scores(self) -> dict[str, torch.Tensor]:
        return {
            name: score.detach().clone()
            for name, score in self.base_score_dict.items()
        }

    def _calibration_batch_limit(self) -> int | None:
        if self.calibration_mode == "full_data":
            return None
        return max(1, self.calibration_batches)

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

    def _calibration_score_from_batches(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        grad_dict = self._grad_dict_for_batches(batches)
        score_dict: dict[str, torch.Tensor] = {}
        for name, parameter in self._prunable_items():
            grad = grad_dict.get(name)
            if grad is None:
                score_dict[name] = torch.zeros_like(parameter)
            else:
                score_dict[name] = (parameter * grad).detach().abs().clone()
        return score_dict

    def compute_calibration_score(self) -> dict[str, torch.Tensor]:
        """Compute calibration scores over D_cal according to the selected mode."""
        if self.calibration_mode == "base_only":
            return self._clone_base_scores()

        was_training = self.model.training
        saved_state = self._state_clone()
        self.model.zero_grad(set_to_none=True)
        self.model.train()

        try:
            batches = list(self._bounded_batches(self._calibration_batch_limit()))
            return self._calibration_score_from_batches(batches)
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
        if self.calibration_mode == "base_only" or self.sign_batches == 0:
            return self._clone_like_scores(0.0)

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
        if self.calibration_mode == "base_only":
            return {
                name: score.detach().clone()
                for name, score in normalized_base.items()
            }
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
        return mask_jaccard(base_mask, stable_mask_dict)

    def _mean_value(self, tensor_dict: Mapping[str, torch.Tensor]) -> float:
        total_sum = sum(float(tensor.detach().sum().cpu().item()) for tensor in tensor_dict.values())
        total_count = sum(tensor.numel() for tensor in tensor_dict.values())
        if total_count == 0:
            return 0.0
        return total_sum / total_count

    def stability_metrics(self) -> dict[str, float]:
        """Estimate score/mask stability across lightweight calibration windows."""
        if self.stability_repeats <= 1 or self.calibration_mode == "base_only":
            return {
                "stability_repeats": float(self.stability_repeats),
                "score_rank_correlation": 1.0,
                "mask_jaccard": 1.0,
            }

        batches_per_repeat = max(1, self.calibration_batches)
        all_batches = list(self._bounded_batches(batches_per_repeat * self.stability_repeats))
        if len(all_batches) < batches_per_repeat:
            return {
                "stability_repeats": 1.0,
                "score_rank_correlation": 1.0,
                "mask_jaccard": 1.0,
            }

        was_training = self.model.training
        saved_state = self._state_clone()
        self.model.zero_grad(set_to_none=True)
        self.model.train()
        try:
            scores: list[dict[str, torch.Tensor]] = []
            masks: list[dict[str, torch.Tensor]] = []
            for repeat_idx in range(self.stability_repeats):
                start = repeat_idx * batches_per_repeat
                stop = start + batches_per_repeat
                window = all_batches[start:stop]
                if not window:
                    continue
                score = self._calibration_score_from_batches(window)
                scores.append(score)
                masks.append(global_topk_mask(score, self.sparsity))
                self.model.zero_grad(set_to_none=True)
        finally:
            self._restore_state(saved_state, was_training)

        if len(scores) < 2:
            return {
                "stability_repeats": float(len(scores)),
                "score_rank_correlation": 1.0,
                "mask_jaccard": 1.0,
            }

        rank_values = [
            score_rank_correlation(scores[0], score)
            for score in scores[1:]
        ]
        jaccard_values = [
            mask_jaccard(masks[0], mask)
            for mask in masks[1:]
        ]
        return {
            "stability_repeats": float(len(scores)),
            "score_rank_correlation": sum(rank_values) / len(rank_values),
            "mask_jaccard": sum(jaccard_values) / len(jaccard_values),
        }

    def run(self) -> TCSMOutput:
        """Run the minimal TCSM data flow."""
        calibration_score = self.compute_calibration_score()
        sign_consistency = self.compute_sign_consistency()
        stable_score = self.build_stable_score(calibration_score, sign_consistency)
        stable_mask = self.build_stable_mask(stable_score)
        stable_omega = self.build_stable_omega(stable_score)
        metrics = {
            "calibration_mode": self.calibration_mode,
            "sparsity": compute_sparsity(stable_mask),
            "mean_stable_score": self._mean_value(stable_score),
            "mean_stable_omega": self._mean_value(stable_omega),
            "base_stable_jaccard": self._base_stable_jaccard(stable_mask),
            **self.stability_metrics(),
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
