"""Small statistical helpers for experiment summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean."""
    if not values:
        raise ValueError("values must not be empty.")
    return sum(float(value) for value in values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    """Return sample standard deviation with Bessel correction."""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((float(value) - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def confidence_interval_95(values: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(mean, half_width, std)`` for a two-sided 95% interval."""
    avg = mean(values)
    std = sample_std(values)
    if len(values) < 2:
        return avg, 0.0, std
    degrees = len(values) - 1
    critical = _T_CRITICAL_975.get(degrees, 1.96)
    half_width = critical * std / math.sqrt(len(values))
    return avg, half_width, std


def welch_t_statistic(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    """Return Welch's t statistic and approximate degrees of freedom."""
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Welch statistic requires at least two samples per group.")
    mean_first = mean(first)
    mean_second = mean(second)
    var_first = sample_std(first) ** 2
    var_second = sample_std(second) ** 2
    scaled_first = var_first / len(first)
    scaled_second = var_second / len(second)
    denominator = math.sqrt(scaled_first + scaled_second)
    if denominator == 0.0:
        return 0.0, float("inf")
    t_stat = (mean_first - mean_second) / denominator
    numerator = (scaled_first + scaled_second) ** 2
    df_first = 0.0 if scaled_first == 0.0 else (scaled_first**2) / (len(first) - 1)
    df_second = 0.0 if scaled_second == 0.0 else (scaled_second**2) / (len(second) - 1)
    degrees = numerator / (df_first + df_second) if (df_first + df_second) > 0.0 else float("inf")
    return t_stat, degrees
