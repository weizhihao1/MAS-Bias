from __future__ import annotations

import json
import math
import re

from .constants import CHOICE_KEYS


def _extract_json_candidates(response_text: str) -> list[str]:
    candidates: list[str] = []

    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", response_text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced_matches)

    brace_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    stripped_text = response_text.strip()
    if stripped_text:
        candidates.append(stripped_text)

    return candidates


def _safe_json_loads(json_text: str) -> dict | None:
    cleaned_text = re.sub(r"[\x00-\x09\x0B\x0C\x0E-\x1F]", " ", json_text).strip()
    if not cleaned_text:
        return None

    try:
        parsed_value = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed_value, dict):
        return parsed_value
    return None


def _quantize_distribution(normalized_values: list[float]) -> list[float]:
    scaled_values = [value * 10.0 for value in normalized_values]
    floor_values = [math.floor(value) for value in scaled_values]
    total_tenths = sum(floor_values)
    remaining_tenths = 10 - total_tenths

    fractional_parts = [(scaled_values[index] - floor_values[index], index) for index in range(len(scaled_values))]
    fractional_parts.sort(reverse=True)

    quantized_values = floor_values[:]
    for _, index in fractional_parts[:remaining_tenths]:
        quantized_values[index] += 1

    return [value / 10.0 for value in quantized_values]


def normalize_probabilities(raw_probabilities: dict[str, float] | None) -> dict[str, float]:
    if not raw_probabilities:
        return {"A": 0.3, "B": 0.3, "C": 0.4}

    unnormalized_values: list[float] = []
    for choice_key in CHOICE_KEYS:
        raw_value = raw_probabilities.get(choice_key, 0.0)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        unnormalized_values.append(max(numeric_value, 0.0))

    value_sum = sum(unnormalized_values)
    if value_sum <= 0:
        return {"A": 0.3, "B": 0.3, "C": 0.4}

    normalized_values = [value / value_sum for value in unnormalized_values]
    quantized_values = _quantize_distribution(normalized_values)
    return {choice_key: quantized_values[index] for index, choice_key in enumerate(CHOICE_KEYS)}


def extract_choice_and_reasoning(response_text: str) -> dict[str, object]:
    for candidate_text in _extract_json_candidates(response_text):
        parsed_json = _safe_json_loads(candidate_text)
        if not parsed_json:
            continue

        probability_payload = parsed_json.get("ChoiceProbabilities")
        if probability_payload is None:
            probability_payload = parsed_json.get("choice_probabilities")
        if probability_payload is None:
            probability_payload = {choice_key: parsed_json.get(choice_key, 0.0) for choice_key in CHOICE_KEYS}

        if isinstance(probability_payload, dict):
            normalized_probabilities = normalize_probabilities(probability_payload)
            reasoning_text = parsed_json.get("Reason")
            if reasoning_text is None:
                reasoning_text = parsed_json.get("reason")
            if reasoning_text is None:
                reasoning_text = parsed_json.get("rationale")

            return {
                "choice_probabilities": normalized_probabilities,
                "reasoning": str(reasoning_text or "No reasoning provided."),
            }

    return {
        "choice_probabilities": {"A": 0.3, "B": 0.3, "C": 0.4},
        "reasoning": "Failed to parse model output into expected JSON format.",
    }

