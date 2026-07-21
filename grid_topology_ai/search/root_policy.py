"""Utilities for normalized root policies and action sampling."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real

import numpy as np

_TEMPERATURE_EPS = 1e-8


def normalize_policy(
    policy: Mapping[int, Real],
    *,
    context: str = "policy",
) -> dict[int, float]:
    """Validate and normalize positive policy mass."""
    if not policy:
        raise ValueError(f"{context} must not be empty.")

    positive_mass: dict[int, float] = {}
    total = 0.0

    for raw_action_id, raw_probability in policy.items():
        if (
            isinstance(raw_action_id, bool)
            or not isinstance(raw_action_id, Integral)
            or int(raw_action_id) < 0
        ):
            raise ValueError(
                f"{context} contains invalid action ID {raw_action_id!r}."
            )
        if isinstance(raw_probability, bool) or not isinstance(
            raw_probability,
            Real,
        ):
            raise TypeError(
                f"{context} probability for action {raw_action_id!r} "
                "must be numeric."
            )

        probability = float(raw_probability)
        if not math.isfinite(probability):
            raise ValueError(
                f"{context} probability for action {raw_action_id!r} "
                "must be finite."
            )
        if probability < 0.0:
            raise ValueError(
                f"{context} probability for action {raw_action_id!r} "
                "must be non-negative."
            )
        if probability == 0.0:
            continue

        action_id = int(raw_action_id)
        positive_mass[action_id] = probability
        total += probability

    if total <= 0.0:
        raise ValueError(f"{context} must contain positive probability mass.")

    return {
        action_id: probability / total
        for action_id, probability in positive_mass.items()
    }


def constrain_policy(
    policy: Mapping[int, Real],
    allowed_action_ids: Iterable[int],
    *,
    context: str = "policy",
) -> dict[int, float]:
    """Restrict a policy to allowed actions and renormalize remaining mass."""
    normalized = normalize_policy(policy, context=context)
    allowed = _validated_action_ids(allowed_action_ids, context=context)
    constrained = {
        action_id: probability
        for action_id, probability in normalized.items()
        if action_id in allowed
    }
    if not constrained:
        return {}
    return normalize_policy(constrained, context=f"constrained {context}")


def select_action_from_policy(
    policy: Mapping[int, Real],
    temperature: float,
    rng: np.random.Generator,
    *,
    context: str = "policy",
) -> int:
    """Select exactly one action from the normalized policy support."""
    normalized = normalize_policy(policy, context=context)
    temperature = _validated_temperature(temperature)

    if temperature <= _TEMPERATURE_EPS:
        return max(normalized, key=normalized.__getitem__)

    action_ids = np.fromiter(normalized, dtype=np.int64)
    probabilities = np.fromiter(normalized.values(), dtype=np.float64)
    log_weights = np.log(probabilities) / temperature
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    return int(rng.choice(action_ids, p=weights))


def require_action_in_policy_support(
    action_id: int,
    policy: Mapping[int, Real],
    *,
    context: str = "policy",
) -> None:
    """Reject execution of an action outside positive policy support."""
    normalized = normalize_policy(policy, context=context)
    if (
        isinstance(action_id, bool)
        or not isinstance(action_id, Integral)
        or int(action_id) not in normalized
    ):
        raise ValueError(
            f"Action {action_id!r} is outside the support of {context}."
        )


def _validated_action_ids(
    action_ids: Iterable[int],
    *,
    context: str,
) -> set[int]:
    validated: set[int] = set()
    for raw_action_id in action_ids:
        if (
            isinstance(raw_action_id, bool)
            or not isinstance(raw_action_id, Integral)
            or int(raw_action_id) < 0
        ):
            raise ValueError(
                f"Allowed actions for {context} contain invalid action ID "
                f"{raw_action_id!r}."
            )
        validated.add(int(raw_action_id))
    return validated


def _validated_temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("temperature must be numeric.")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative.")
    return temperature
