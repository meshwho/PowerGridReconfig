from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

from grid_topology_ai.outcome import TerminationReason


PHYSICAL_OBJECTIVE_SCHEMA_VERSION = 2
OVERLOAD_LIMIT_PERCENT = 100.0
HARD_OVERLOAD_LIMIT_PERCENT = 120.0
VOLTAGE_VIOLATION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class PhysicalStateAssessment:
    max_loading_percent: float
    num_overloaded_branches: int
    num_hard_overloaded_branches: int
    total_voltage_violation: float
    power_flow_converged: bool
    state_finite: bool
    topology_connected: bool
    thermal_solved: bool
    hard_overload_free: bool
    voltage_feasible: bool
    generator_p_feasible: bool
    generator_q_feasible: bool
    angle_difference_feasible: bool
    physically_secure: bool
    num_voltage_violations: int
    num_generator_p_violations: int
    num_generator_q_violations: int
    num_angle_difference_violations: int

    @property
    def objective_solved(self) -> bool:
        return self.physically_secure


@dataclass(frozen=True, slots=True)
class StopOutcome:
    solved: bool
    termination_reason: TerminationReason


STOP_POLICIES: frozenset[str] = frozenset(
    {"never", "solved_only", "no_hard_overloads", "always"}
)


def _require_key(metrics: Mapping[str, object], key: str) -> object:
    if key not in metrics:
        raise KeyError(key)
    return metrics[key]


def _validate_finite_nonnegative_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{key} must be a numeric value.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite.")
    if numeric < 0.0:
        raise ValueError(f"{key} must be non-negative.")
    return numeric


def _validate_nonnegative_integer(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer-valued number.")

    try:
        integer = operator.index(value)
    except TypeError:
        if not isinstance(value, Real):
            raise TypeError(f"{key} must be an integer-valued number.")

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(f"{key} must be finite.")

        if not numeric.is_integer():
            raise ValueError(f"{key} must be integer-valued.")

        integer = int(numeric)

    if integer < 0:
        raise ValueError(f"{key} must be non-negative.")

    return int(integer)


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
    power_flow_converged = bool(metrics.get("power_flow_converged", True))
    state_finite = bool(metrics.get("state_finite", True))
    topology_connected = bool(metrics.get("topology_connected", True))
    generator_p_feasible = bool(metrics.get("generator_p_feasible", True))
    generator_q_feasible = bool(metrics.get("generator_q_feasible", True))
    angle_difference_feasible = bool(metrics.get("angle_difference_feasible", True))
    num_voltage_violations = _validate_nonnegative_integer(metrics.get("num_voltage_violations", metrics.get("num_low_voltage_buses", 0) + metrics.get("num_high_voltage_buses", 0)), "num_voltage_violations")
    num_generator_p_violations = _validate_nonnegative_integer(metrics.get("num_generator_p_violations", 0), "num_generator_p_violations")
    num_generator_q_violations = _validate_nonnegative_integer(metrics.get("num_generator_q_violations", 0), "num_generator_q_violations")
    num_angle_difference_violations = _validate_nonnegative_integer(metrics.get("num_angle_difference_violations", 0), "num_angle_difference_violations")
    physically_secure = (
        power_flow_converged
        and state_finite
        and topology_connected
        and thermal_solved
        and voltage_feasible
        and generator_p_feasible
        and generator_q_feasible
        and angle_difference_feasible
    )

    return PhysicalStateAssessment(
        max_loading_percent=max_loading_percent,
        num_overloaded_branches=num_overloaded_branches,
        num_hard_overloaded_branches=num_hard_overloaded_branches,
        total_voltage_violation=total_voltage_violation,
        power_flow_converged=power_flow_converged,
        state_finite=state_finite,
        topology_connected=topology_connected,
        thermal_solved=thermal_solved,
        hard_overload_free=hard_overload_free,
        voltage_feasible=voltage_feasible,
        generator_p_feasible=generator_p_feasible,
        generator_q_feasible=generator_q_feasible,
        angle_difference_feasible=angle_difference_feasible,
        physically_secure=physically_secure,
        num_voltage_violations=num_voltage_violations,
        num_generator_p_violations=num_generator_p_violations,
        num_generator_q_violations=num_generator_q_violations,
        num_angle_difference_violations=num_angle_difference_violations,
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
        return assessment.physically_secure
    if stop_policy == "no_hard_overloads":
        return assessment.hard_overload_free
    raise ValueError(f"Unknown stop_policy: {stop_policy}")


def classify_stop_outcome(
    assessment: PhysicalStateAssessment,
    *,
    allow_handoff_with_hard_overloads: bool,
) -> StopOutcome:
    if assessment.physically_secure:
        return StopOutcome(solved=True, termination_reason=TerminationReason.SOLVED)
    if assessment.hard_overload_free:
        return StopOutcome(solved=False, termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH)
    if allow_handoff_with_hard_overloads:
        return StopOutcome(
            solved=False,
            termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        )
    return StopOutcome(
        solved=False,
        termination_reason=TerminationReason.UNSAFE_STOP,
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
        "physically_secure_definition": "power_flow_converged and state_finite and topology_connected and thermal_solved and voltage_feasible and generator_p_feasible and generator_q_feasible and angle_difference_feasible.",
        "existing_solved_flag": "physically_secure",
        "safe_handoff_definition": "Not thermal_solved and hard_overload_free.",
    }
