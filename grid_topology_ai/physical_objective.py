from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Mapping

from grid_topology_ai.termination import TerminationReason


PHYSICAL_OBJECTIVE_SCHEMA_VERSION = 3
OVERLOAD_LIMIT_PERCENT = 100.0
HARD_OVERLOAD_LIMIT_PERCENT = 120.0
THERMAL_LIMIT_TOLERANCE_PERCENT = 1e-6
VOLTAGE_LIMIT_TOLERANCE_PU = 1e-6
GENERATOR_LIMIT_TOLERANCE_MW = 1e-6
GENERATOR_LIMIT_TOLERANCE_MVAR = 1e-6
ANGLE_LIMIT_TOLERANCE_DEGREES = 1e-6


@dataclass(frozen=True, slots=True)
class PhysicalStateAssessment:
    power_flow_converged: bool
    all_values_finite: bool
    topology_connected: bool
    max_loading_percent: float
    num_overloaded_branches: int
    num_hard_overloaded_branches: int
    total_thermal_overload_mva: float
    thermal_solved: bool
    thermal_feasible: bool
    hard_overload_free: bool
    num_low_voltage_buses: int
    num_high_voltage_buses: int
    total_voltage_violation: float
    voltage_feasible: bool
    num_generator_p_violations: int
    total_generator_p_violation_mw: float
    generator_p_feasible: bool
    num_generator_q_violations: int
    total_generator_q_violation_mvar: float
    generator_q_feasible: bool
    num_angle_difference_violations: int
    total_angle_difference_violation_degrees: float
    angle_difference_feasible: bool
    physically_secure: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _validate_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool.")
    return value


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
    power_flow_converged = _validate_bool(
        _require_key(metrics, "power_flow_converged"), "power_flow_converged"
    )
    all_values_finite = _validate_bool(
        _require_key(metrics, "all_values_finite"), "all_values_finite"
    )
    topology_connected = _validate_bool(
        _require_key(metrics, "topology_connected"), "topology_connected"
    )
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
            "num_hard_overloaded_branches cannot exceed num_overloaded_branches."
        )
    total_thermal_overload_mva = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_thermal_overload_mva"),
        "total_thermal_overload_mva",
    )
    num_low_voltage_buses = _validate_nonnegative_integer(
        _require_key(metrics, "num_low_voltage_buses"), "num_low_voltage_buses"
    )
    num_high_voltage_buses = _validate_nonnegative_integer(
        _require_key(metrics, "num_high_voltage_buses"),
        "num_high_voltage_buses",
    )
    total_voltage_violation = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_voltage_violation"),
        "total_voltage_violation",
    )
    num_generator_p_violations = _validate_nonnegative_integer(
        _require_key(metrics, "num_generator_p_violations"),
        "num_generator_p_violations",
    )
    total_generator_p_violation_mw = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_generator_p_violation_mw"),
        "total_generator_p_violation_mw",
    )
    num_generator_q_violations = _validate_nonnegative_integer(
        _require_key(metrics, "num_generator_q_violations"),
        "num_generator_q_violations",
    )
    total_generator_q_violation_mvar = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_generator_q_violation_mvar"),
        "total_generator_q_violation_mvar",
    )
    num_angle_difference_violations = _validate_nonnegative_integer(
        _require_key(metrics, "num_angle_difference_violations"),
        "num_angle_difference_violations",
    )
    total_angle_difference_violation_degrees = _validate_finite_nonnegative_number(
        _require_key(metrics, "total_angle_difference_violation_degrees"),
        "total_angle_difference_violation_degrees",
    )

    thermal_feasible = num_overloaded_branches == 0
    hard_overload_free = num_hard_overloaded_branches == 0
    voltage_feasible = (
        num_low_voltage_buses == 0 and num_high_voltage_buses == 0
    )
    generator_p_feasible = num_generator_p_violations == 0
    generator_q_feasible = num_generator_q_violations == 0
    angle_difference_feasible = num_angle_difference_violations == 0
    physically_secure = all(
        (
            power_flow_converged,
            all_values_finite,
            topology_connected,
            thermal_feasible,
            voltage_feasible,
            generator_p_feasible,
            generator_q_feasible,
            angle_difference_feasible,
        )
    )

    return PhysicalStateAssessment(
        power_flow_converged=power_flow_converged,
        all_values_finite=all_values_finite,
        topology_connected=topology_connected,
        max_loading_percent=max_loading_percent,
        num_overloaded_branches=num_overloaded_branches,
        num_hard_overloaded_branches=num_hard_overloaded_branches,
        total_thermal_overload_mva=total_thermal_overload_mva,
        thermal_solved=thermal_feasible,
        thermal_feasible=thermal_feasible,
        hard_overload_free=hard_overload_free,
        num_low_voltage_buses=num_low_voltage_buses,
        num_high_voltage_buses=num_high_voltage_buses,
        total_voltage_violation=total_voltage_violation,
        voltage_feasible=voltage_feasible,
        num_generator_p_violations=num_generator_p_violations,
        total_generator_p_violation_mw=total_generator_p_violation_mw,
        generator_p_feasible=generator_p_feasible,
        num_generator_q_violations=num_generator_q_violations,
        total_generator_q_violation_mvar=total_generator_q_violation_mvar,
        generator_q_feasible=generator_q_feasible,
        num_angle_difference_violations=num_angle_difference_violations,
        total_angle_difference_violation_degrees=(
            total_angle_difference_violation_degrees
        ),
        angle_difference_feasible=angle_difference_feasible,
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
    if not include_stop_action or stop_policy == "never":
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
        return StopOutcome(True, TerminationReason.SOLVED)
    if assessment.hard_overload_free:
        return StopOutcome(False, TerminationReason.HANDOFF_TO_REDISPATCH)
    if allow_handoff_with_hard_overloads:
        return StopOutcome(
            False,
            TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        )
    return StopOutcome(False, TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD)


def physical_objective_contract() -> dict[str, object]:
    return {
        "schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "overload_limit_percent": OVERLOAD_LIMIT_PERCENT,
        "hard_overload_limit_percent": HARD_OVERLOAD_LIMIT_PERCENT,
        "thermal_limit_tolerance_percent": THERMAL_LIMIT_TOLERANCE_PERCENT,
        "voltage_limit_tolerance_pu": VOLTAGE_LIMIT_TOLERANCE_PU,
        "generator_limit_tolerance_mw": GENERATOR_LIMIT_TOLERANCE_MW,
        "generator_limit_tolerance_mvar": GENERATOR_LIMIT_TOLERANCE_MVAR,
        "angle_limit_tolerance_degrees": ANGLE_LIMIT_TOLERANCE_DEGREES,
        "solved_definition": "assessment.physically_secure",
        "thermal_solved_definition": (
            "Diagnostic only: no active rated branch exceeds RATE_A."
        ),
        "physically_secure_definition": (
            "power_flow_converged and all_values_finite and "
            "topology_connected and thermal_feasible and voltage_feasible and "
            "generator_p_feasible and generator_q_feasible and "
            "angle_difference_feasible"
        ),
        "safe_handoff_definition": (
            "Not physically_secure and hard_overload_free."
        ),
    }
