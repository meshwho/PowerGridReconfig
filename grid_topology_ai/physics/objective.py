from __future__ import annotations

import json
import math
import operator
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from numbers import Integral, Real
from typing import Mapping

import numpy as np

from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)

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


def physical_objective_contract(physics_config=None) -> dict[str, object]:
    """Describe the actual physics contract used to produce an artifact."""
    from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    return {
        "physics_config": config.to_dict(),
        "overload_limit_percent": config.overload_limit_percent,
        "hard_overload_limit_percent": config.hard_overload_limit_percent,
        "thermal_limit_tolerance_percent": config.thermal_tolerance_percent,
        "voltage_limit_tolerance_pu": config.voltage_tolerance_pu,
        "generator_limit_tolerance_mw": config.generator_p_tolerance_mw,
        "generator_limit_tolerance_mvar": config.generator_q_tolerance_mvar,
        "angle_limit_tolerance_degrees": config.angle_tolerance_degrees,
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


# Kept temporarily for teacher checkpoint/state metadata written by the existing
# teacher runner. TerminalOutcomeEvidence itself is intentionally unversioned.
TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION = 3


class RedispatchStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    VALIDATED = "validated"


_REQUESTED_REDISPATCH_REASONS = frozenset(
    {
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
    }
)
_SAFE_HANDOFF_REASONS = frozenset(
    {
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        TerminationReason.REDISPATCH_VALIDATED,
    }
)
_HARD_OVERLOAD_REASONS = frozenset(
    {
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD,
    }
)

_BOOLEAN_ASSESSMENT_FIELDS = frozenset(
    {
        "power_flow_converged",
        "all_values_finite",
        "topology_connected",
        "thermal_solved",
        "thermal_feasible",
        "hard_overload_free",
        "voltage_feasible",
        "generator_p_feasible",
        "generator_q_feasible",
        "angle_difference_feasible",
        "physically_secure",
    }
)
_INTEGER_ASSESSMENT_FIELDS = frozenset(
    {
        "num_overloaded_branches",
        "num_hard_overloaded_branches",
        "num_low_voltage_buses",
        "num_high_voltage_buses",
        "num_generator_p_violations",
        "num_generator_q_violations",
        "num_angle_difference_violations",
    }
)


def redispatch_status_for_reason(
    termination_reason: TerminationReason | str,
) -> RedispatchStatus:
    reason = parse_termination_reason(
        termination_reason,
        allow_none=False,
    )
    assert reason is not None

    if reason is TerminationReason.REDISPATCH_VALIDATED:
        return RedispatchStatus.VALIDATED
    if reason in _REQUESTED_REDISPATCH_REASONS:
        return RedispatchStatus.REQUESTED
    return RedispatchStatus.NOT_REQUESTED


def _parse_redispatch_status(value: object) -> RedispatchStatus:
    if isinstance(value, RedispatchStatus):
        return value
    if not isinstance(value, str):
        raise TypeError(
            "redispatch_status must be a string or RedispatchStatus."
        )
    try:
        return RedispatchStatus(value.strip())
    except ValueError:
        allowed = ", ".join(status.value for status in RedispatchStatus)
        raise ValueError(
            f"Unknown redispatch_status {value!r}. Expected one of: {allowed}."
        ) from None


def _validate_assessment_values(
    assessment: PhysicalStateAssessment,
    *,
    field_name: str,
) -> None:
    for field in fields(PhysicalStateAssessment):
        value = getattr(assessment, field.name)
        qualified_name = f"{field_name}.{field.name}"

        if field.name in _BOOLEAN_ASSESSMENT_FIELDS:
            if not isinstance(value, bool):
                raise TypeError(f"{qualified_name} must be a bool.")
            continue

        if field.name in _INTEGER_ASSESSMENT_FIELDS:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                raise ValueError(
                    f"{qualified_name} must be a non-negative integer."
                )
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"{qualified_name} must be a finite non-negative number."
            )

    if (
        assessment.num_hard_overloaded_branches
        > assessment.num_overloaded_branches
    ):
        raise ValueError(
            f"{field_name}.num_hard_overloaded_branches cannot exceed "
            f"{field_name}.num_overloaded_branches."
        )

    expected_thermal_feasible = assessment.num_overloaded_branches == 0
    expected_hard_overload_free = (
        assessment.num_hard_overloaded_branches == 0
    )
    expected_voltage_feasible = (
        assessment.num_low_voltage_buses == 0
        and assessment.num_high_voltage_buses == 0
    )
    expected_generator_p_feasible = (
        assessment.num_generator_p_violations == 0
    )
    expected_generator_q_feasible = (
        assessment.num_generator_q_violations == 0
    )
    expected_angle_feasible = (
        assessment.num_angle_difference_violations == 0
    )

    expected_flags = {
        "thermal_solved": expected_thermal_feasible,
        "thermal_feasible": expected_thermal_feasible,
        "hard_overload_free": expected_hard_overload_free,
        "voltage_feasible": expected_voltage_feasible,
        "generator_p_feasible": expected_generator_p_feasible,
        "generator_q_feasible": expected_generator_q_feasible,
        "angle_difference_feasible": expected_angle_feasible,
    }
    for name, expected in expected_flags.items():
        if getattr(assessment, name) != expected:
            raise ValueError(
                f"{field_name}.{name} contradicts the physical counts."
            )

    expected_secure = all(
        (
            assessment.power_flow_converged,
            assessment.all_values_finite,
            assessment.topology_connected,
            expected_thermal_feasible,
            expected_voltage_feasible,
            expected_generator_p_feasible,
            expected_generator_q_feasible,
            expected_angle_feasible,
        )
    )
    if assessment.physically_secure != expected_secure:
        raise ValueError(
            f"{field_name}.physically_secure contradicts the physical flags."
        )


def _assessment_from_mapping(
    value: object,
    *,
    field_name: str,
) -> PhysicalStateAssessment | None:
    if value is None:
        return None
    if isinstance(value, PhysicalStateAssessment):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping, "
            "PhysicalStateAssessment, or None."
        )

    expected_fields = {
        field.name
        for field in fields(PhysicalStateAssessment)
    }
    observed_fields = set(value)
    if observed_fields != expected_fields:
        missing = sorted(expected_fields - observed_fields)
        unexpected = sorted(observed_fields - expected_fields)
        raise ValueError(
            f"{field_name} fields do not match the current state: "
            f"missing={missing}, unexpected={unexpected}."
        )

    assessment = PhysicalStateAssessment(
        **{name: value[name] for name in expected_fields}
    )
    _validate_assessment_values(
        assessment,
        field_name=field_name,
    )
    return assessment


def _parse_topology_utility(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("topology_utility must be a finite number in [-1, 1].")
    utility = float(value)
    if not math.isfinite(utility) or not -1.0 <= utility <= 1.0:
        raise ValueError("topology_utility must be a finite number in [-1, 1].")
    return utility


def _reject_json_constant(value: str) -> None:
    raise ValueError(
        f"Non-finite JSON constant {value!r} is not allowed."
    )


@dataclass(frozen=True, slots=True)
class TerminalOutcomeEvidence:
    """Physical evidence for one classified terminal episode outcome."""

    solved: bool
    termination_reason: TerminationReason
    assessment: PhysicalStateAssessment | None
    redispatch_status: RedispatchStatus
    topology_utility: float
    redispatch_assessment: PhysicalStateAssessment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solved, bool):
            raise TypeError("solved must be a bool.")

        reason = parse_termination_reason(
            self.termination_reason,
            allow_none=False,
        )
        assert reason is not None
        status = _parse_redispatch_status(self.redispatch_status)
        topology_utility = _parse_topology_utility(self.topology_utility)
        assessment = _assessment_from_mapping(
            self.assessment,
            field_name="assessment",
        )
        redispatch_assessment = _assessment_from_mapping(
            self.redispatch_assessment,
            field_name="redispatch_assessment",
        )

        object.__setattr__(self, "termination_reason", reason)
        object.__setattr__(self, "redispatch_status", status)
        object.__setattr__(self, "topology_utility", topology_utility)
        object.__setattr__(self, "assessment", assessment)
        object.__setattr__(
            self,
            "redispatch_assessment",
            redispatch_assessment,
        )

        validate_outcome_invariants(
            solved=self.solved,
            termination_reason=reason,
        )

        expected_status = redispatch_status_for_reason(reason)
        if status is not expected_status:
            raise ValueError(
                f"{reason.value} requires "
                f"redispatch_status={expected_status.value!r}."
            )

        if assessment is None:
            if reason is not TerminationReason.POWER_FLOW_FAILED:
                raise ValueError(
                    f"{reason.value} requires a terminal physical assessment."
                )
            if topology_utility != -1.0:
                raise ValueError(
                    "A terminal outcome without a physical assessment requires "
                    "topology_utility=-1.0."
                )
        else:
            _validate_assessment_values(
                assessment,
                field_name="assessment",
            )
            validate_outcome_invariants(
                solved=self.solved,
                termination_reason=reason,
                physically_secure=assessment.physically_secure,
            )

            if assessment.physically_secure and topology_utility != 1.0:
                raise ValueError(
                    "A physically secure topology requires topology_utility=1.0."
                )
            if not assessment.physically_secure and topology_utility >= 1.0:
                raise ValueError(
                    "An insecure topology must have topology_utility < 1.0."
                )
            if (
                reason is TerminationReason.POWER_FLOW_FAILED
                and assessment.power_flow_converged
            ):
                raise ValueError(
                    "power_flow_failed cannot carry a converged assessment."
                )
            if (
                reason in _SAFE_HANDOFF_REASONS
                and not assessment.hard_overload_free
            ):
                raise ValueError(
                    f"{reason.value} requires a hard-overload-free assessment."
                )
            if (
                reason in _HARD_OVERLOAD_REASONS
                and assessment.hard_overload_free
            ):
                raise ValueError(
                    f"{reason.value} requires a hard-overloaded assessment."
                )

        if status is RedispatchStatus.VALIDATED:
            if redispatch_assessment is None:
                raise ValueError(
                    "Validated redispatch requires redispatch_assessment."
                )
            _validate_assessment_values(
                redispatch_assessment,
                field_name="redispatch_assessment",
            )
            if not redispatch_assessment.physically_secure:
                raise ValueError(
                    "Validated redispatch requires a physically secure "
                    "redispatch_assessment."
                )
        elif redispatch_assessment is not None:
            raise ValueError(
                "redispatch_assessment is allowed only for validated redispatch."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "solved": self.solved,
            "termination_reason": self.termination_reason.value,
            "redispatch_status": self.redispatch_status.value,
            "topology_utility": self.topology_utility,
            "assessment": (
                None if self.assessment is None else asdict(self.assessment)
            ),
            "redispatch_assessment": (
                None
                if self.redispatch_assessment is None
                else asdict(self.redispatch_assessment)
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: object) -> "TerminalOutcomeEvidence":
        if not isinstance(value, str) or not value.strip():
            raise TypeError(
                "terminal outcome evidence JSON must be a non-empty string."
            )
        try:
            payload = json.loads(
                value,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Invalid terminal outcome evidence JSON.") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Terminal outcome evidence JSON must contain an object."
            )
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "TerminalOutcomeEvidence":
        if not isinstance(value, Mapping):
            raise TypeError("terminal outcome evidence must be a mapping.")

        expected_fields = {
            "solved",
            "termination_reason",
            "redispatch_status",
            "topology_utility",
            "assessment",
            "redispatch_assessment",
        }
        observed_fields = set(value)
        if observed_fields != expected_fields:
            missing = sorted(expected_fields - observed_fields)
            unexpected = sorted(observed_fields - expected_fields)
            raise ValueError(
                "terminal outcome evidence fields do not match the current "
                f"format: missing={missing}, unexpected={unexpected}."
            )

        return cls(
            solved=value["solved"],
            termination_reason=value["termination_reason"],
            assessment=_assessment_from_mapping(
                value["assessment"],
                field_name="assessment",
            ),
            redispatch_status=value["redispatch_status"],
            topology_utility=value["topology_utility"],
            redispatch_assessment=_assessment_from_mapping(
                value["redispatch_assessment"],
                field_name="redispatch_assessment",
            ),
        )


def parse_terminal_outcome_fields(
    *,
    solved: object,
    termination_reason: object,
) -> tuple[bool, TerminationReason]:
    """Parse the two scalar fields that identify a terminal outcome."""

    if not isinstance(solved, (bool, np.bool_)):
        raise ValueError(
            f"solved must be a boolean, got {solved!r}"
        )

    parsed_solved = bool(solved)
    reason = parse_termination_reason(
        termination_reason,
        allow_none=False,
    )
    assert reason is not None
    validate_outcome_invariants(
        solved=parsed_solved,
        termination_reason=reason,
    )
    return parsed_solved, reason


def terminal_evidence_from_metadata(
    metadata: Mapping[str, object],
    *,
    source: str,
    solved: object,
    termination_reason: object,
) -> TerminalOutcomeEvidence:
    """Read terminal evidence from current artifact metadata."""

    parsed_solved, reason = parse_terminal_outcome_fields(
        solved=solved,
        termination_reason=termination_reason,
    )

    raw_evidence = metadata.get("terminal_outcome_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError(
            f"{source} is missing terminal_outcome_evidence metadata."
        )

    try:
        evidence = TerminalOutcomeEvidence.from_mapping(raw_evidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} contains invalid terminal_outcome_evidence: {exc}"
        ) from exc

    if (
        evidence.solved is not parsed_solved
        or evidence.termination_reason is not reason
    ):
        raise ValueError(
            f"{source} terminal_outcome_evidence contradicts "
            "episode_solved or episode_termination_reason."
        )

    return evidence
