"""Strict evidence attached to one terminal episode outcome."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from numbers import Integral, Real
from typing import Mapping

from grid_topology_ai.physics.objective import PhysicalStateAssessment
from grid_topology_ai.termination import (
    TerminationReason,
    parse_termination_reason,
    validate_outcome_invariants,
)


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
            f"{field_name} fields do not match the current contract: "
            f"missing={missing}, unexpected={unexpected}."
        )

    assessment = PhysicalStateAssessment(
        **{
            name: value[name]
            for name in expected_fields
        }
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
        status = _parse_redispatch_status(
            self.redispatch_status
        )
        topology_utility = _parse_topology_utility(self.topology_utility)
        assessment = _assessment_from_mapping(
            self.assessment,
            field_name="assessment",
        )
        redispatch_assessment = _assessment_from_mapping(
            self.redispatch_assessment,
            field_name="redispatch_assessment",
        )

        object.__setattr__(
            self,
            "termination_reason",
            reason,
        )
        object.__setattr__(
            self,
            "redispatch_status",
            status,
        )
        object.__setattr__(
            self,
            "topology_utility",
            topology_utility,
        )
        object.__setattr__(
            self,
            "assessment",
            assessment,
        )
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
                    f"{reason.value} requires a "
                    "hard-overload-free assessment."
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
                "redispatch_assessment is allowed only for validated "
                "redispatch."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION,
            "solved": self.solved,
            "termination_reason": self.termination_reason.value,
            "redispatch_status": self.redispatch_status.value,
            "topology_utility": self.topology_utility,
            "assessment": (
                None
                if self.assessment is None
                else asdict(self.assessment)
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
    def from_json(
        cls,
        value: object,
    ) -> "TerminalOutcomeEvidence":
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
            raise ValueError(
                "Invalid terminal outcome evidence JSON."
            ) from exc
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
            "schema_version",
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
                "terminal outcome evidence fields do not match the "
                f"current contract: missing={missing}, "
                f"unexpected={unexpected}."
            )

        version = value["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, Integral)
            or int(version)
            != TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported terminal outcome evidence schema version: "
                f"{version!r}."
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
