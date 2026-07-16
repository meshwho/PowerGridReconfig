from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


PHYSICAL_OBJECTIVE_SCHEMA_VERSION = 1
OVERLOAD_LIMIT_PERCENT = 100.0
HARD_OVERLOAD_LIMIT_PERCENT = 120.0
VOLTAGE_VIOLATION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class PhysicalStateAssessment:
    max_loading_percent: float
    num_overloaded_branches: int
    num_hard_overloaded_branches: int
    total_voltage_violation: float
    thermal_solved: bool
    hard_overload_free: bool
    voltage_feasible: bool
    physically_secure: bool


@dataclass(frozen=True, slots=True)
class StopOutcome:
    solved: bool
    termination_reason: str


STOP_POLICIES: frozenset[str] = frozenset(
    {"never", "solved_only", "no_hard_overloads", "always"}
)


def _require_key(metrics: Mapping[str, object], key: str) -> object:
    if key not in metrics:
        raise KeyError(key)
    return metrics[key]


def _validate_finite_nonnegative_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a numeric value.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite.")
    if numeric < 0.0:
        raise ValueError(f"{key} must be non-negative.")
    return numeric


def _validate_nonnegative_integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer value.")
    if value < 0:
        raise ValueError(f"{key} must be non-negative.")
    return int(value)


def assess_physical_state(metrics: Mapping[str, object]) -> PhysicalStateAssessment:
    max_loading_percent = _validate_finite_nonnegative_number(
        _require_key(metrics, "max_loading_percent"), "max_loading_percent"
    )
    num_overloaded_branches = _validate_nonnegative_integer(
        _require_key(metrics, "num_overloaded_branches"),
        "num_overloaded_branches",
    )
    num_hard_overloaded_branches = _validate_nonnegative_integer(
        _require_key(metrics, "num_hard_overloaded_branches"),
        "num_hard_overloaded_branches",
    )
    if num_hard_overloaded_branches > num_overloaded_branches:
        raise ValueError(
            "num_hard_overloaded_branches cannot exceed "
            "num_overloaded_branches."
        )
    total_voltage_violation = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_voltage_violation"),
        "total_voltage_violation",
    )

    thermal_solved = num_overloaded_branches == 0
    hard_overload_free = num_hard_overloaded_branches == 0
    voltage_feasible = total_voltage_violation <= VOLTAGE_VIOLATION_TOLERANCE
    physically_secure = thermal_solved and voltage_feasible

    return PhysicalStateAssessment(
        max_loading_percent=max_loading_percent,
        num_overloaded_branches=num_overloaded_branches,
        num_hard_overloaded_branches=num_hard_overloaded_branches,
        total_voltage_violation=total_voltage_violation,
        thermal_solved=thermal_solved,
        hard_overload_free=hard_overload_free,
        voltage_feasible=voltage_feasible,
        physically_secure=physically_secure,
    )


def stop_allowed_for_policy(
    assessment: PhysicalStateAssessment,
    *,
    stop_policy: str,
    include_stop_action: bool = True,
) -> bool:
    if stop_policy not in STOP_POLICIES:
        raise ValueError(f"Unknown stop_policy: {stop_policy}")
    if not include_stop_action:
        return False
    if stop_policy == "never":
        return False
    if stop_policy == "always":
        return True
    if stop_policy == "solved_only":
        return assessment.thermal_solved
    if stop_policy == "no_hard_overloads":
        return assessment.hard_overload_free
    raise ValueError(f"Unknown stop_policy: {stop_policy}")


def classify_stop_outcome(
    assessment: PhysicalStateAssessment,
    *,
    allow_handoff_with_hard_overloads: bool,
) -> StopOutcome:
    if assessment.thermal_solved:
        return StopOutcome(solved=True, termination_reason="solved")
    if assessment.hard_overload_free:
        return StopOutcome(solved=False, termination_reason="handoff_to_redispatch")
    if allow_handoff_with_hard_overloads:
        return StopOutcome(
            solved=False,
            termination_reason="handoff_to_redispatch_with_hard_overload",
        )
    return StopOutcome(
        solved=False,
        termination_reason="unsafe_stop_with_hard_overload",
    )


def physical_objective_contract() -> dict[str, object]:
    return {
        "schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "overload_limit_percent": OVERLOAD_LIMIT_PERCENT,
        "hard_overload_limit_percent": HARD_OVERLOAD_LIMIT_PERCENT,
        "voltage_violation_tolerance": VOLTAGE_VIOLATION_TOLERANCE,
        "solved_definition": (
            "No active branch has loading above overload_limit_percent."
        ),
        "hard_overload_free_definition": (
            "No active branch has loading above hard_overload_limit_percent."
        ),
        "voltage_feasible_definition": (
            "total_voltage_violation is at or below voltage_violation_tolerance."
        ),
        "physically_secure_definition": "thermal_solved and voltage_feasible.",
        "existing_solved_flag": "thermal_solved",
        "safe_handoff_definition": "Not thermal_solved and hard_overload_free.",
    }
