from __future__ import annotations

import pytest

from grid_topology_ai.outcome_contract import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.physical_objective import (
    assess_physical_state,
    classify_stop_outcome,
)
from grid_topology_ai.termination import TerminationReason


def _metrics(
    *,
    overloaded: int,
    hard_overloaded: int,
) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": (
            90.0
            if overloaded == 0
            else 130.0
            if hard_overloaded
            else 110.0
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


@pytest.mark.parametrize(
    (
        "overloaded",
        "hard_overloaded",
        "allow_hard_handoff",
        "expected_reason",
        "expected_status",
    ),
    [
        (
            0,
            0,
            False,
            TerminationReason.SOLVED,
            RedispatchStatus.NOT_REQUESTED,
        ),
        (
            1,
            0,
            False,
            TerminationReason.HANDOFF_TO_REDISPATCH,
            RedispatchStatus.REQUESTED,
        ),
        (
            1,
            1,
            True,
            TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
            RedispatchStatus.REQUESTED,
        ),
        (
            1,
            1,
            False,
            TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD,
            RedispatchStatus.NOT_REQUESTED,
        ),
    ],
)
def test_stop_classification_builds_consistent_terminal_evidence(
    overloaded: int,
    hard_overloaded: int,
    allow_hard_handoff: bool,
    expected_reason: TerminationReason,
    expected_status: RedispatchStatus,
) -> None:
    assessment = assess_physical_state(
        _metrics(
            overloaded=overloaded,
            hard_overloaded=hard_overloaded,
        )
    )
    outcome = classify_stop_outcome(
        assessment,
        allow_handoff_with_hard_overloads=allow_hard_handoff,
    )

    evidence = TerminalOutcomeEvidence(
        solved=outcome.solved,
        termination_reason=outcome.termination_reason,
        assessment=assessment,
        redispatch_status=expected_status,
    )

    assert evidence.termination_reason is expected_reason
    assert evidence.solved is assessment.physically_secure


def test_max_steps_accepts_an_unsecure_converged_assessment() -> None:
    assessment = assess_physical_state(
        _metrics(
            overloaded=1,
            hard_overloaded=0,
        )
    )

    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
        assessment=assessment,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )

    assert evidence.assessment is assessment
    assert evidence.termination_reason is TerminationReason.MAX_STEPS_REACHED


def test_power_flow_failure_payload_is_explicitly_assessment_free() -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.POWER_FLOW_FAILED,
        assessment=None,
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )

    payload = evidence.to_dict()

    assert payload["assessment"] is None
    assert payload["termination_reason"] == "power_flow_failed"
    assert payload["redispatch_status"] == "not_requested"
