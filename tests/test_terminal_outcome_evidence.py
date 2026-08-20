from __future__ import annotations

from dataclasses import replace

import pytest

from grid_topology_ai.outcome_record import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.termination import TerminationReason


def _metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 90.0,
        "num_overloaded_branches": 0,
        "num_hard_overloaded_branches": 0,
        "total_thermal_overload_mva": 0.0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }
    metrics.update(overrides)
    return metrics


def _assessment(**overrides: object):
    return assess_physical_state(_metrics(**overrides))


def test_solved_evidence_round_trips_without_schema_metadata() -> None:
    evidence = TerminalOutcomeEvidence(
        solved=True,
        termination_reason=TerminationReason.SOLVED,
        assessment=_assessment(),
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
        topology_utility=1.0,
    )
    payload = evidence.to_dict()
    assert "schema_version" not in payload
    assert TerminalOutcomeEvidence.from_mapping(payload) == evidence


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
    ],
)
def test_safe_handoff_requires_requested_redispatch(reason) -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=reason,
        assessment=_assessment(
            max_loading_percent=110.0,
            num_overloaded_branches=1,
            total_thermal_overload_mva=4.0,
        ),
        redispatch_status=RedispatchStatus.REQUESTED,
        topology_utility=-1.0,
    )
    assert evidence.assessment is not None
    assert evidence.assessment.hard_overload_free is True


@pytest.mark.parametrize(
    "reason, redispatch_status",
    [
        (
            TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
            RedispatchStatus.REQUESTED,
        ),
        (
            TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD,
            RedispatchStatus.NOT_REQUESTED,
        ),
    ],
)
def test_hard_overload_outcomes_require_hard_overload_evidence(
    reason,
    redispatch_status,
) -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=reason,
        assessment=_assessment(
            max_loading_percent=130.0,
            num_overloaded_branches=1,
            num_hard_overloaded_branches=1,
            total_thermal_overload_mva=12.0,
        ),
        redispatch_status=redispatch_status,
        topology_utility=-1.0,
    )
    assert evidence.assessment is not None
    assert evidence.assessment.hard_overload_free is False


def test_power_flow_failure_allows_missing_assessment() -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.POWER_FLOW_FAILED,
        assessment=None,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
        topology_utility=-1.0,
    )
    assert TerminalOutcomeEvidence.from_mapping(evidence.to_dict()) == evidence


def test_unsolved_reason_rejects_secure_assessment() -> None:
    with pytest.raises(ValueError, match="physically_secure"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.MAX_STEPS_REACHED,
            assessment=_assessment(),
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=1.0,
        )


def test_safe_handoff_rejects_hard_overload() -> None:
    with pytest.raises(ValueError, match="hard-overload-free"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH,
            assessment=_assessment(
                max_loading_percent=130.0,
                num_overloaded_branches=1,
                num_hard_overloaded_branches=1,
                total_thermal_overload_mva=12.0,
            ),
            redispatch_status=RedispatchStatus.REQUESTED,
            topology_utility=-1.0,
        )


def test_hard_overload_reason_rejects_safe_assessment() -> None:
    with pytest.raises(ValueError, match="hard-overloaded"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD,
            assessment=_assessment(
                max_loading_percent=110.0,
                num_overloaded_branches=1,
                total_thermal_overload_mva=4.0,
            ),
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=-1.0,
        )


@pytest.mark.parametrize(
    "reason, status",
    [
        (
            TerminationReason.HANDOFF_TO_REDISPATCH,
            RedispatchStatus.NOT_REQUESTED,
        ),
        (
            TerminationReason.MAX_STEPS_REACHED,
            RedispatchStatus.REQUESTED,
        ),
    ],
)
def test_redispatch_status_must_match_reason(reason, status) -> None:
    with pytest.raises(ValueError, match="redispatch_status"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=reason,
            assessment=_assessment(
                max_loading_percent=110.0,
                num_overloaded_branches=1,
                total_thermal_overload_mva=4.0,
            ),
            redispatch_status=status,
            topology_utility=-1.0,
        )


def test_non_failure_reason_requires_assessment() -> None:
    with pytest.raises(ValueError, match="physical assessment"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.MAX_STEPS_REACHED,
            assessment=None,
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=-1.0,
        )


def test_converged_assessment_rejected_for_power_flow_failure() -> None:
    with pytest.raises(ValueError, match="cannot carry a converged"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.POWER_FLOW_FAILED,
            assessment=_assessment(
                max_loading_percent=110.0,
                num_overloaded_branches=1,
                total_thermal_overload_mva=4.0,
            ),
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=-1.0,
        )


def test_mapping_rejects_unknown_fields() -> None:
    payload = TerminalOutcomeEvidence(
        solved=True,
        termination_reason=TerminationReason.SOLVED,
        assessment=_assessment(),
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
        topology_utility=1.0,
    ).to_dict()
    payload["debug"] = True
    with pytest.raises(ValueError, match="unexpected"):
        TerminalOutcomeEvidence.from_mapping(payload)


def test_mapping_rejects_inconsistent_derived_flags() -> None:
    assessment = _assessment(
        max_loading_percent=110.0,
        num_overloaded_branches=1,
        total_thermal_overload_mva=4.0,
    )
    payload = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
        assessment=assessment,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
        topology_utility=-1.0,
    ).to_dict()
    assessment_payload = dict(payload["assessment"])
    assessment_payload["thermal_feasible"] = True
    payload["assessment"] = assessment_payload
    with pytest.raises(ValueError, match="thermal_feasible"):
        TerminalOutcomeEvidence.from_mapping(payload)


def test_direct_assessment_rejects_inconsistent_flags() -> None:
    assessment = _assessment(
        max_loading_percent=110.0,
        num_overloaded_branches=1,
        total_thermal_overload_mva=4.0,
    )
    inconsistent = replace(assessment, hard_overload_free=False)
    with pytest.raises(ValueError, match="hard_overload_free"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.MAX_STEPS_REACHED,
            assessment=inconsistent,
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
            topology_utility=-1.0,
        )
