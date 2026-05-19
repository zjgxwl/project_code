import pytest
import torch
from torch import nn

from sso.metrics import (
    accuracy_from_logits,
    confidence_interval_95,
    count_parameters,
    estimate_conv2d_flops,
    loss_barrier,
    mask_dict_sparsity,
    mask_jaccard,
    mean,
    sample_std,
    score_rank_correlation,
    tensor_sparsity,
    welch_t_statistic,
)


def test_classification_and_sparsity_metrics() -> None:
    logits = torch.tensor([[0.1, 0.9], [2.0, 1.0]])
    targets = torch.tensor([1, 1])

    accuracies = accuracy_from_logits(logits, targets, topk=(1, 2))

    assert accuracies["top1"] == pytest.approx(0.5)
    assert accuracies["top2"] == pytest.approx(1.0)
    assert tensor_sparsity(torch.tensor([0.0, 1.0, 0.0, 2.0])) == pytest.approx(0.5)

    first = {"w": torch.tensor([1, 1, 0, 0])}
    second = {"w": torch.tensor([1, 0, 1, 0])}
    assert mask_dict_sparsity(first) == pytest.approx(0.5)
    assert mask_jaccard(first, second) == pytest.approx(1.0 / 3.0)


def test_ranking_and_statistical_metrics() -> None:
    first = {"w": torch.tensor([1.0, 2.0, 3.0])}
    same = {"w": torch.tensor([10.0, 20.0, 30.0])}
    reversed_scores = {"w": torch.tensor([30.0, 20.0, 10.0])}

    assert score_rank_correlation(first, same) == pytest.approx(1.0)
    assert score_rank_correlation(first, reversed_scores) == pytest.approx(-1.0)
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert sample_std([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    avg, half_width, std = confidence_interval_95([1.0, 2.0, 3.0])
    assert avg == pytest.approx(2.0)
    assert half_width > 0.0
    assert std == pytest.approx(1.0)

    t_stat, degrees = welch_t_statistic([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert t_stat < 0.0
    assert degrees > 0.0


def test_complexity_and_loss_helpers() -> None:
    model = nn.Conv2d(3, 4, kernel_size=3, bias=False)

    assert count_parameters(model) == 3 * 4 * 3 * 3
    assert estimate_conv2d_flops(model, input_shape=(1, 3, 8, 8), device="cpu") == pytest.approx(7776.0)
    assert loss_barrier([1.0, 2.0, 1.0]) == pytest.approx(1.0)
