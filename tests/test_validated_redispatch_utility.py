from __future__ import annotations

import pytest

from grid_topology_ai.physics.objective import RedispatchStatus, TerminalOutcomeEvidence
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.termination import TeacherOutcome, TerminationReason
from grid_topology_ai.value_targets import (
    terminal_evidence_from_row,
    topology_utility_from_evidence,
)


def _metrics(
    *,
    overloaded: int,
    hard_overloaded: int = 0,
) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": (
            90.0 if overloaded == 0 else 130.0 if hard_overloaded else 110.0
        ),
        "num_overloaded_branches": overloaded,
        "num_hard_overloaded_branches": hard_overloaded,
        "total_thermal_overload_mva": (
            0.0 if overloaded == 0 else 12.0 if hard_overloaded else 4.0
        ),
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


def _assessment(*, overloaded: int, hard_overloaded: int = 0):
    return assess_physical_state(
        _metrics(overloaded=overloaded, hard_overloaded=hard_overloaded)
    )


def _validated_evidence(
    *,
    topology_utility: float = 0.35,
    hard_overloaded: int = 0,
) -> TerminalOutcomeEvidence:
    return TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.REDISPATCH_VALIDATED,
        assessment=_assessment(overloaded=1, hard_overloaded=hard_overloaded),
        redispatch_status=RedispatchStatus.VALIDATED,
        topology_utility=topology_utility,
        redispatch_assessment=_assessment(overloaded=0),
    )


def test_validated_redispatch_uses_pre_redispatch_topology_utility() -> None:
    evidence = _validated_evidence(topology_utility=0.35)

    assert topology_utility_from_evidence(evidence) == pytest.approx(0.35)


def test_redispatch_status_does_not_change_primary_topology_utility() -> None:
    assessment = _assessment(overloaded=1)
    requested = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
        assessment=assessment,
        redispatch_status=RedispatchStatus.REQUESTED,
        topology_utility=0.35,
    )
    validated = _validated_evidence(topology_utility=0.35)

    assert topology_utility_from_evidence(requested) == pytest.approx(0.35)
    assert topology_utility_from_evidence(validated) == pytest.approx(0.35)


def test_validated_redispatch_rejects_unsafe_result() -> None:
    with pytest.raises(ValueError, match="physically secure"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.REDISPATCH_VALIDATED,
            assessment=_assessment(overloaded=1),
            redispatch_status=RedispatchStatus.VALIDATED,
            topology_utility=0.35,
            redispatch_assessment=_assessment(overloaded=1),
        )


def test_validated_redispatch_accepts_hard_overloaded_pre_redispatch_state() -> None:
    evidence = _validated_evidence(
        topology_utility=-0.5,
        hard_overloaded=1,
    )

    assert evidence.assessment is not None
    assert evidence.assessment.hard_overload_free is False
    assert evidence.redispatch_status is RedispatchStatus.VALIDATED
    assert evidence.redispatch_assessment is not None
    assert evidence.redispatch_assessment.physically_secure is True
    assert topology_utility_from_evidence(evidence) == pytest.approx(-0.5)


def test_validated_evidence_maps_to_redispatch_teacher_outcome() -> None:
    evidence = _validated_evidence(topology_utility=0.2, hard_overloaded=1)
    row = {
        "solved": False,
        "teacher_outcome": TeacherOutcome.REDISPATCH.value,
        "terminal_outcome_evidence_json": evidence.to_json(),
    }

    parsed = terminal_evidence_from_row(row, context="validated row")

    assert parsed == evidence
    assert parsed.redispatch_status is RedispatchStatus.VALIDATED


def test_validated_evidence_rejects_wrong_semantic_outcome() -> None:
    evidence = _validated_evidence(topology_utility=0.2)
    row = {
        "solved": False,
        "teacher_outcome": TeacherOutcome.MAX_STEPS_REACHED.value,
        "terminal_outcome_evidence_json": evidence.to_json(),
    }

    with pytest.raises(ValueError, match="contradicts terminal evidence"):
        terminal_evidence_from_row(row, context="validated row")
