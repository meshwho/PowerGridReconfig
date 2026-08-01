from __future__ import annotations

import pytest

from grid_topology_ai.outcome_contract import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.return_contract import terminal_utility_from_outcome
from grid_topology_ai.termination import TerminationReason


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


def _assessment(*, overloaded: int, hard_overloaded: int = 0):
    return assess_physical_state(
        _metrics(
            overloaded=overloaded,
            hard_overloaded=hard_overloaded,
        )
    )


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
    ],
)
def test_unvalidated_handoff_is_negative(reason: TerminationReason) -> None:
    assert terminal_utility_from_outcome(False, reason) == (
        -1.0,
        reason.value,
    )


def test_validated_redispatch_requires_evidence() -> None:
    with pytest.raises(ValueError, match="requires terminal outcome evidence"):
        terminal_utility_from_outcome(
            False,
            TerminationReason.REDISPATCH_VALIDATED,
        )


def test_validated_redispatch_is_the_only_neutral_outcome() -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.REDISPATCH_VALIDATED,
        assessment=_assessment(overloaded=1),
        redispatch_status=RedispatchStatus.VALIDATED,
        redispatch_assessment=_assessment(overloaded=0),
    )

    assert terminal_utility_from_outcome(
        False,
        TerminationReason.REDISPATCH_VALIDATED,
        evidence=evidence,
    ) == (0.0, "redispatch_validated")


def test_validated_redispatch_rejects_unsafe_result() -> None:
    with pytest.raises(ValueError, match="physically secure"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.REDISPATCH_VALIDATED,
            assessment=_assessment(overloaded=1),
            redispatch_status=RedispatchStatus.VALIDATED,
            redispatch_assessment=_assessment(overloaded=1),
        )


def test_validated_redispatch_rejects_hard_overload_handoff() -> None:
    with pytest.raises(ValueError, match="hard-overload-free"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.REDISPATCH_VALIDATED,
            assessment=_assessment(
                overloaded=1,
                hard_overloaded=1,
            ),
            redispatch_status=RedispatchStatus.VALIDATED,
            redispatch_assessment=_assessment(overloaded=0),
        )


def test_utility_rejects_mismatched_evidence() -> None:
    evidence = TerminalOutcomeEvidence(
        solved=False,
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
        assessment=_assessment(overloaded=1),
        redispatch_status=RedispatchStatus.NOT_REQUESTED,
    )

    with pytest.raises(ValueError, match="termination_reason"):
        terminal_utility_from_outcome(
            False,
            TerminationReason.POWER_FLOW_FAILED,
            evidence=evidence,
        )
