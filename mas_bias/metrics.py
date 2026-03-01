from __future__ import annotations

import math

from .constants import CHOICE_KEYS


def _probability_values(choice_probabilities: dict[str, float]) -> list[float]:
    return [float(choice_probabilities.get(choice_key, 0.0)) for choice_key in CHOICE_KEYS]


def calculate_variance(choice_probabilities: dict[str, float]) -> float:
    probability_values = _probability_values(choice_probabilities)
    mean_value = sum(probability_values) / len(probability_values)
    return sum((value - mean_value) ** 2 for value in probability_values) / len(probability_values)


def calculate_entropy(choice_probabilities: dict[str, float]) -> float:
    probability_values = _probability_values(choice_probabilities)
    entropy_value = 0.0
    for probability_value in probability_values:
        if probability_value > 0:
            entropy_value -= probability_value * math.log2(probability_value)
    return entropy_value


def calculate_gini(choice_probabilities: dict[str, float]) -> float:
    probability_values = sorted(_probability_values(choice_probabilities))
    if not probability_values:
        return 0.0

    denominator = len(probability_values) - 1
    if denominator <= 0:
        return 0.0

    numerator = 0.0
    number_of_choices = len(probability_values)
    for index, value in enumerate(probability_values, start=1):
        numerator += (2 * index - number_of_choices - 1) * value
    return numerator / denominator


def calculate_uniform_kl_divergence(choice_probabilities: dict[str, float]) -> float:
    probability_values = _probability_values(choice_probabilities)
    epsilon = 1e-9
    clipped_values = [max(value, epsilon) for value in probability_values]
    normalizer = sum(clipped_values)
    normalized_values = [value / normalizer for value in clipped_values]

    uniform_probability = 1.0 / len(normalized_values)
    kl_divergence = 0.0
    for probability_value in normalized_values:
        kl_divergence += probability_value * math.log2(probability_value / uniform_probability)
    return kl_divergence


def metric_bundle(choice_probabilities: dict[str, float]) -> dict[str, float]:
    return {
        "variance": calculate_variance(choice_probabilities),
        "entropy": calculate_entropy(choice_probabilities),
        "gini": calculate_gini(choice_probabilities),
        "uniform_kl": calculate_uniform_kl_divergence(choice_probabilities),
    }

