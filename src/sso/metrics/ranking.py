"""Ranking metrics for pruning scores."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        for original_index, _value in indexed[index:next_index]:
            ranks[original_index] = average_rank
        index = next_index
    return ranks


def _pearson(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Inputs must have the same length.")
    if len(first) < 2:
        return 1.0
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    numerator = sum(a * b for a, b in zip(centered_first, centered_second, strict=True))
    denom_first = math.sqrt(sum(value * value for value in centered_first))
    denom_second = math.sqrt(sum(value * value for value in centered_second))
    denominator = denom_first * denom_second
    if denominator == 0.0:
        return 1.0 if first == second else 0.0
    return numerator / denominator


def score_rank_correlation(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> float:
    """Return Spearman rank correlation between two score dictionaries."""
    if set(first) != set(second):
        raise ValueError("Score dictionaries must have identical keys.")

    first_values: list[float] = []
    second_values: list[float] = []
    for name in sorted(first):
        first_score = first[name].detach().flatten().cpu()
        second_score = second[name].detach().flatten().cpu()
        if tuple(first_score.shape) != tuple(second_score.shape):
            raise ValueError(f"Score shape mismatch for {name}.")
        first_values.extend(float(value) for value in first_score.tolist())
        second_values.extend(float(value) for value in second_score.tolist())

    if not first_values:
        return 1.0
    return _pearson(_average_ranks(first_values), _average_ranks(second_values))
