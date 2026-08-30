from __future__ import annotations

import copy

import numpy as np
import pytest

from grid_topology_ai.physics.objective import RedispatchStatus
from grid_topology_ai.termination import TeacherOutcome, TerminationReason
from grid_topology_ai.value_targets import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
    add_outcome_value_targets_to_rows,
    terminal_evidence_from_row,
    topology_utility_from_evidence,
)
from tests.outcome_evidence_helpers import terminal_evidence_fields


def valid_row(
    *,
    scenario_id: object = 1,
    step: object = 0,
    done: object = True,
    reason: object = TerminationReason.SOLVED,
    topology_utility: float | None = None,
    redispatch_status: RedispatchStatus | None = None,
) -> dict[str, object]:
    evidence_fields = terminal_evidence_fields(
        reason,
        topology_utility=topology_utility,
        redispatch_status=redispatch_status,
    )
    solved = reason is TerminationReason.SOLVED or reason == TerminationReason.SOLVED.value
    return {
        "run_id": "test-run",
        "iteration": 1,
        "episode_id": f"test-episode-{scenario_id}",
        "scenario_id": scenario_id,
        "step": step,
        "solved": bool(solved),
        "done": done,
        **evidence_fields,
    }


def assert_rejected_without_target_mutation(
    rows: list[dict[str, object]],
    *,
    gamma: object = TERMINAL_UTILITY_GAMMA,
    group_keys: object = ("scenario_id",),
    match: str | None = None,
) -> None:
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match=match):
        add_outcome_value_targets_to_rows(
            rows,
            gamma=gamma,  # type: ignore[arg-type]
            group_keys=group_keys,  # type: ignore[arg-type]
        )
    assert rows == before
    assert all("outcome_value_target" not in row for row in rows)


@pytest.mark.parametrize(
    "gamma",
    [float("nan"), float("inf"), True, "1.0", None, -0.01, 1.01],
)
def test_invalid_gamma_is_atomic(gamma: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row()],
        gamma=gamma,
        match="gamma",
    )


@pytest.mark.parametrize("gamma", [0.0, 0.95, 1.0, np.float32(0.95)])
def test_caller_gamma_is_normalized_to_terminal_contract(gamma: object) -> None:
    rows = [valid_row()]
    add_outcome_value_targets_to_rows(rows, gamma=gamma)  # type: ignore[arg-type]

    row = rows[0]
    assert row["outcome_gamma"] == pytest.approx(1.0)
    assert row["outcome_value_target"] == pytest.approx(1.0)
    assert row["outcome_steps_to_terminal"] == 1
    assert row["outcome_value_target_mode"] == VALUE_TARGET_MODE
    assert "outcome_value_target_contract_version" not in row


@pytest.mark.parametrize(
    ("reason", "status", "utility", "expected_outcome"),
    [
        (
            TerminationReason.SOLVED,
            RedispatchStatus.NOT_REQUESTED,
            1.0,
            TeacherOutcome.SOLVED,
        ),
        (
            TerminationReason.REDISPATCH_VALIDATED,
            RedispatchStatus.VALIDATED,
            0.35,
            TeacherOutcome.REDISPATCH,
        ),
        (
            TerminationReason.MAX_STEPS_REACHED,
            RedispatchStatus.REQUESTED,
            -0.25,
            TeacherOutcome.MAX_STEPS_REACHED,
        ),
        (
            TerminationReason.TEACHER_DEPTH_LIMIT,
            RedispatchStatus.REQUESTED,
            -0.1,
            TeacherOutcome.MAX_STEPS_REACHED,
        ),
    ],
)
def test_value_target_uses_semantic_teacher_outcome_and_topology_utility(
    reason: TerminationReason,
    status: RedispatchStatus,
    utility: float,
    expected_outcome: TeacherOutcome,
) -> None:
    rows = [
        valid_row(
            step=step,
            reason=reason,
            topology_utility=utility,
            redispatch_status=status,
        )
        for step in (0, 1)
    ]

    add_outcome_value_targets_to_rows(rows, gamma=TERMINAL_UTILITY_GAMMA)

    assert [row["outcome_steps_to_terminal"] for row in rows] == [2, 1]
    for row in rows:
        assert row["teacher_outcome"] == expected_outcome.value
        assert row["outcome_class"] == expected_outcome.value
        assert row["outcome_value_target"] == pytest.approx(utility)


def test_diagnostic_reason_does_not_define_public_outcome() -> None:
    row = valid_row(
        reason=TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER,
        topology_utility=0.2,
        redispatch_status=RedispatchStatus.REQUESTED,
    )
    assert row["teacher_outcome"] == TeacherOutcome.MAX_STEPS_REACHED.value

    add_outcome_value_targets_to_rows([row], gamma=1.0)
    assert row["outcome_class"] == TeacherOutcome.MAX_STEPS_REACHED.value
    assert row["outcome_value_target"] == pytest.approx(0.2)


def test_missing_or_unknown_teacher_outcome_is_rejected_atomically() -> None:
    missing = valid_row()
    del missing["teacher_outcome"]
    assert_rejected_without_target_mutation([missing], match="teacher_outcome")

    unknown = valid_row()
    unknown["teacher_outcome"] = "teacher_depth_limit"
    assert_rejected_without_target_mutation([unknown], match="teacher_outcome")


def test_teacher_outcome_must_match_terminal_evidence() -> None:
    row = valid_row(
        reason=TerminationReason.REDISPATCH_VALIDATED,
        redispatch_status=RedispatchStatus.VALIDATED,
        topology_utility=0.3,
    )
    row["teacher_outcome"] = TeacherOutcome.MAX_STEPS_REACHED.value

    assert_rejected_without_target_mutation(
        [row],
        match="contradicts terminal evidence",
    )


def test_invalid_solved_or_done_is_rejected_atomically() -> None:
    bad_solved = valid_row()
    bad_solved["solved"] = "True"
    assert_rejected_without_target_mutation([bad_solved], match="solved")

    bad_done = valid_row()
    bad_done["done"] = False
    assert_rejected_without_target_mutation([bad_done], match="done")


def test_mixed_terminal_evidence_in_one_episode_is_rejected_atomically() -> None:
    rows = [
        valid_row(step=0),
        valid_row(
            step=1,
            reason=TerminationReason.MAX_STEPS_REACHED,
            redispatch_status=RedispatchStatus.REQUESTED,
        ),
    ]
    assert_rejected_without_target_mutation(rows, match="mixed terminal evidence")


@pytest.mark.parametrize("step", [True, "1", 1.0, 1.5, -1])
def test_invalid_step_is_atomic(step: object) -> None:
    assert_rejected_without_target_mutation([valid_row(step=step)], match="step")


def test_missing_duplicate_and_noncontiguous_steps_are_rejected() -> None:
    missing = valid_row()
    del missing["step"]
    assert_rejected_without_target_mutation([missing], match="step")

    assert_rejected_without_target_mutation(
        [valid_row(step=0), valid_row(step=0)],
        match="Duplicate",
    )
    assert_rejected_without_target_mutation(
        [valid_row(step=0), valid_row(step=2)],
        match="contiguous",
    )


def test_unsorted_steps_preserve_rows_and_compute_distance_by_step() -> None:
    later = valid_row(step=1)
    earlier = valid_row(step=0)
    rows = [later, earlier]

    add_outcome_value_targets_to_rows(rows, gamma=1.0)

    assert rows == [later, earlier]
    assert later["outcome_steps_to_terminal"] == 1
    assert earlier["outcome_steps_to_terminal"] == 2


def test_same_scenario_different_episodes_are_isolated() -> None:
    solved = valid_row(step=0)
    solved.update(run_id="run-a", iteration=1, episode_id="episode-a")

    failed = valid_row(
        step=0,
        reason=TerminationReason.MAX_STEPS_REACHED,
        redispatch_status=RedispatchStatus.REQUESTED,
        topology_utility=-0.4,
    )
    failed.update(run_id="run-b", iteration=2, episode_id="episode-b")

    rows = [solved, failed]
    add_outcome_value_targets_to_rows(rows, gamma=1.0)

    assert solved["outcome_class"] == TeacherOutcome.SOLVED.value
    assert solved["outcome_value_target"] == pytest.approx(1.0)
    assert failed["outcome_class"] == TeacherOutcome.MAX_STEPS_REACHED.value
    assert failed["outcome_value_target"] == pytest.approx(-0.4)


def test_topology_utility_public_helper_reads_terminal_evidence() -> None:
    row = valid_row(
        reason=TerminationReason.REDISPATCH_VALIDATED,
        redispatch_status=RedispatchStatus.VALIDATED,
        topology_utility=0.42,
    )
    evidence = terminal_evidence_from_row(row, context="test row")

    assert topology_utility_from_evidence(evidence) == pytest.approx(0.42)
