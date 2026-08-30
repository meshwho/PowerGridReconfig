import inspect
import json

import pytest

from grid_topology_ai.termination import (
    TeacherOutcome,
    TerminationReason,
    classify_teacher_outcome,
    parse_termination_reason,
    termination_reason_value,
    validate_outcome_invariants,
)


def test_termination_reason_json_round_trip_uses_stable_value() -> None:
    payload = json.loads(
        json.dumps(
            {"termination_reason": TerminationReason.SOLVED.value}
        )
    )
    assert parse_termination_reason(payload["termination_reason"]) is (
        TerminationReason.SOLVED
    )


def test_unknown_and_dynamic_reasons_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown termination_reason"):
        parse_termination_reason("reason_42")


@pytest.mark.parametrize(
    ("solved", "reason"),
    [
        (True, TerminationReason.MAX_STEPS_REACHED),
        (False, TerminationReason.SOLVED),
    ],
)
def test_contradictory_solved_reason_pairs_are_rejected(
    solved: bool,
    reason: TerminationReason,
) -> None:
    with pytest.raises(ValueError, match="Contradictory outcome"):
        validate_outcome_invariants(
            solved=solved,
            termination_reason=reason,
        )


def test_action_details_are_separate_from_canonical_reason() -> None:
    metadata = {
        "termination_reason": termination_reason_value(
            TerminationReason.UNSAFE_STOP_WITH_HARD_OVERLOAD
        ),
        "selected_action_id": 42,
    }
    assert metadata["termination_reason"] == "unsafe_stop_with_hard_overload"
    assert metadata["selected_action_id"] == 42


def test_teacher_outcomes_are_exactly_the_three_public_statuses() -> None:
    assert [outcome.value for outcome in TeacherOutcome] == [
        "solved",
        "redispatch",
        "max_steps_reached",
    ]


@pytest.mark.parametrize(
    ("topology_solved", "redispatch_validated", "expected"),
    [
        (True, False, TeacherOutcome.SOLVED),
        (True, True, TeacherOutcome.SOLVED),
        (False, True, TeacherOutcome.REDISPATCH),
        (False, False, TeacherOutcome.MAX_STEPS_REACHED),
    ],
)
def test_teacher_outcome_depends_only_on_terminal_physical_result(
    topology_solved: bool,
    redispatch_validated: bool,
    expected: TeacherOutcome,
) -> None:
    assert classify_teacher_outcome(
        topology_solved=topology_solved,
        redispatch_validated=redispatch_validated,
    ) is expected


def test_diagnostic_termination_reason_is_not_a_teacher_outcome_input() -> None:
    parameters = inspect.signature(classify_teacher_outcome).parameters

    assert set(parameters) == {"topology_solved", "redispatch_validated"}
    assert "termination_reason" not in parameters
