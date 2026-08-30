from __future__ import annotations

from grid_topology_ai.physics.objective import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
    assess_physical_state,
)
from grid_topology_ai.termination import (
    TerminationReason,
    classify_teacher_outcome,
    parse_termination_reason,
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


def _default_redispatch_status(reason: TerminationReason) -> RedispatchStatus:
    if reason is TerminationReason.REDISPATCH_VALIDATED:
        return RedispatchStatus.VALIDATED
    if reason in {
        TerminationReason.HANDOFF_TO_REDISPATCH,
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        TerminationReason.MAX_STEPS_REACHED,
        TerminationReason.TEACHER_DEPTH_LIMIT,
    }:
        return RedispatchStatus.REQUESTED
    return RedispatchStatus.NOT_REQUESTED


def terminal_evidence(
    termination_reason: TerminationReason | str,
    *,
    topology_utility: float | None = None,
    redispatch_status: RedispatchStatus | None = None,
) -> TerminalOutcomeEvidence:
    reason = parse_termination_reason(termination_reason, allow_none=False)
    assert reason is not None

    if reason is TerminationReason.POWER_FLOW_FAILED:
        assessment = None
    elif reason is TerminationReason.SOLVED:
        assessment = assess_physical_state(_metrics(overloaded=0))
    elif reason in {
        TerminationReason.HANDOFF_TO_REDISPATCH_WITH_HARD_OVERLOAD,
        TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD,
    }:
        assessment = assess_physical_state(
            _metrics(overloaded=1, hard_overloaded=1)
        )
    else:
        assessment = assess_physical_state(_metrics(overloaded=1))

    status = redispatch_status or _default_redispatch_status(reason)
    redispatch_assessment = (
        assess_physical_state(_metrics(overloaded=0))
        if status is RedispatchStatus.VALIDATED
        else None
    )

    if topology_utility is None:
        topology_utility = 1.0 if reason is TerminationReason.SOLVED else -1.0

    return TerminalOutcomeEvidence(
        solved=reason is TerminationReason.SOLVED,
        termination_reason=reason,
        assessment=assessment,
        redispatch_status=status,
        topology_utility=float(topology_utility),
        redispatch_assessment=redispatch_assessment,
    )


def terminal_evidence_fields(
    termination_reason: object,
    *,
    topology_utility: float | None = None,
    redispatch_status: RedispatchStatus | None = None,
) -> dict[str, object]:
    try:
        evidence = terminal_evidence(  # type: ignore[arg-type]
            termination_reason,
            topology_utility=topology_utility,
            redispatch_status=redispatch_status,
        )
    except (TypeError, ValueError):
        return {
            "teacher_outcome": "",
            "terminal_outcome_evidence_json": "{}",
        }

    outcome = classify_teacher_outcome(
        topology_solved=evidence.solved,
        redispatch_validated=(
            evidence.redispatch_status is RedispatchStatus.VALIDATED
        ),
    )
    return {
        "teacher_outcome": outcome.value,
        "terminal_outcome_evidence_json": evidence.to_json(),
    }


def terminal_evidence_metadata(
    termination_reason: TerminationReason | str,
    *,
    topology_utility: float | None = None,
    redispatch_status: RedispatchStatus | None = None,
) -> dict[str, object]:
    evidence = terminal_evidence(
        termination_reason,
        topology_utility=topology_utility,
        redispatch_status=redispatch_status,
    )
    outcome = classify_teacher_outcome(
        topology_solved=evidence.solved,
        redispatch_validated=(
            evidence.redispatch_status is RedispatchStatus.VALIDATED
        ),
    )
    return {
        "episode_teacher_outcome": outcome.value,
        "terminal_outcome_evidence": evidence.to_dict(),
    }
