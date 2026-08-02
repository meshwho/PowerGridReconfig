from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)
from grid_topology_ai.return_contract import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
)
from grid_topology_ai.value_targets import (
    add_outcome_value_targets_to_rows,
    terminal_value_from_outcome,
)
from tests.outcome_evidence_helpers import terminal_evidence_fields


def valid_row(
    *,
    scenario_id: object = 1,
    step: object = 0,
    solved: object = True,
    done: object = True,
    termination_reason: object = "solved",
) -> dict[str, object]:
    return {
        "run_id": "test-run",
        "iteration": 1,
        "episode_id": f"test-episode-{scenario_id}",
        "scenario_id": scenario_id,
        "step": step,
        "solved": solved,
        "done": done,
        "termination_reason": termination_reason,
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        **terminal_evidence_fields(termination_reason),
    }


def assert_rejected_without_target_mutation(
    rows: list[dict[str, object]],
    *,
    gamma: object = TERMINAL_UTILITY_GAMMA,
    group_keys: object = ("scenario_id",),
    match: str | None = None,
) -> None:
    before_keys = [set(row) for row in rows]
    before_values = [dict(row) for row in rows]

    with pytest.raises(ValueError, match=match):
        add_outcome_value_targets_to_rows(
            rows,
            gamma=gamma,
            group_keys=group_keys,  # type: ignore[arg-type]
        )

    for row, keys in zip(rows, before_keys):
        assert set(row) == keys
        assert "outcome_value_target" not in row
        assert "outcome_value_target_contract_version" not in row

    for row, before in zip(rows, before_values):
        for key, value in before.items():
            if (
                value is pd.NA
                or value is pd.NaT
                or (
                    isinstance(value, (float, np.floating))
                    and math.isnan(float(value))
                )
                or (
                    isinstance(value, np.datetime64)
                    and np.isnat(value)
                )
            ):
                assert row[key] is value
            else:
                assert row[key] == value


@pytest.mark.parametrize(
    "gamma",
    [
        0.0,
        0.95,
        np.float32(0.95),
        np.float64(0.95),
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        "1.0",
        None,
        -0.01,
        1.01,
    ],
)
def test_non_unit_gamma_is_atomic(gamma: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row()],
        gamma=gamma,
        match="gamma",
    )


@pytest.mark.parametrize(
    "gamma",
    [
        1.0,
        np.float32(1.0),
        np.float64(1.0),
    ],
)
def test_unit_gamma_assigns_complete_contract(
    gamma: object,
) -> None:
    rows = [valid_row()]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=gamma,  # type: ignore[arg-type]
    )

    assert rows[0]["outcome_gamma"] == pytest.approx(1.0)
    assert rows[0]["outcome_value_target"] == pytest.approx(1.0)
    assert rows[0]["outcome_steps_to_terminal"] == 1
    assert rows[0]["outcome_value_target_mode"] == VALUE_TARGET_MODE
    assert VALUE_TARGET_MODE == "alphazero_terminal_utility"
    assert "outcome_value_target_contract_version" in rows[0]


@pytest.mark.parametrize(
    "solved",
    ["False", "True", 0, 1, None, object()],
)
def test_invalid_solved_is_atomic(solved: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row(solved=solved)],
        match="solved",
    )


@pytest.mark.parametrize(
    "done",
    ["False", "True", 0, 1, None, False],
)
def test_invalid_done_is_atomic(done: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row(done=done)],
        match="done",
    )


def test_terminal_value_helper_rejects_string_solved() -> None:
    with pytest.raises(ValueError, match="solved"):
        terminal_value_from_outcome(
            solved="False",  # type: ignore[arg-type]
            termination_reason="solved",
        )


@pytest.mark.parametrize(
    "row",
    [
        valid_row(
            solved=True,
            termination_reason="handoff_to_redispatch",
        ),
        valid_row(
            solved=False,
            termination_reason="solved",
        ),
        valid_row(termination_reason=None),
        valid_row(termination_reason=""),
        valid_row(
            solved=False,
            termination_reason="reason_17",
        ),
        valid_row(done=False),
    ],
)
def test_terminal_invariants_are_atomic(
    row: dict[str, object],
) -> None:
    assert_rejected_without_target_mutation([row])


@pytest.mark.parametrize(
    "rows",
    [
        [
            valid_row(step=0),
            valid_row(
                step=1,
                solved=False,
                termination_reason="max_steps_reached",
            ),
        ],
        [
            valid_row(step=0),
            valid_row(
                step=1,
                termination_reason="max_steps_reached",
            ),
        ],
        [
            valid_row(step=0, done=False),
            valid_row(step=1),
        ],
        [
            valid_row(step=0),
            valid_row(
                step=1,
                solved=False,
                termination_reason="solved",
            ),
        ],
    ],
)
def test_group_outcome_invariants_are_atomic(
    rows: list[dict[str, object]],
) -> None:
    assert_rejected_without_target_mutation(rows)


@pytest.mark.parametrize(
    "reason, solved, expected_class, expected_utility",
    [
        ("solved", True, "solved", 1.0),
        (
            "handoff_to_redispatch",
            False,
            "handoff_to_redispatch",
            -1.0,
        ),
        (
            "max_steps_reached",
            False,
            "max_steps_reached",
            -1.0,
        ),
        (
            "teacher_depth_limit",
            False,
            "teacher_depth_limit",
            -1.0,
        ),
    ],
)
def test_terminal_utility_is_constant_across_episode(
    reason: str,
    solved: bool,
    expected_class: str,
    expected_utility: float,
) -> None:
    rows = [
        valid_row(
            step=0,
            solved=solved,
            termination_reason=reason,
        ),
        valid_row(
            step=1,
            solved=solved,
            termination_reason=reason,
        ),
    ]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )

    for position, row in enumerate(rows):
        assert row["outcome_class"] == expected_class
        assert row["outcome_gamma"] == pytest.approx(1.0)
        assert row["outcome_value_target_mode"] == VALUE_TARGET_MODE
        assert row["outcome_steps_to_terminal"] == 2 - position
        assert row["outcome_value_target_contract_version"]
        assert row["outcome_value_target"] == pytest.approx(
            expected_utility
        )


@pytest.mark.parametrize(
    "step",
    [True, np.bool_(True), "1", 1.0, 1.5, -1],
)
def test_invalid_step_is_atomic(step: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row(step=step)],
        match="step",
    )


def test_missing_and_duplicate_step_are_atomic() -> None:
    missing = valid_row()
    del missing["step"]
    assert_rejected_without_target_mutation(
        [missing],
        match="step",
    )
    assert_rejected_without_target_mutation(
        [
            valid_row(step=0),
            valid_row(step=0),
        ],
        match="Duplicate",
    )


@pytest.mark.parametrize(
    "step",
    [0, np.int32(0), np.int64(0)],
)
def test_valid_integer_steps(step: object) -> None:
    rows = [valid_row(step=step)]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )
    assert rows[0]["outcome_steps_to_terminal"] == 1


def test_unsorted_steps_preserve_input_objects_and_compute_by_step() -> None:
    later = valid_row(step=1)
    earlier = valid_row(step=0)
    rows = [later, earlier]

    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )

    assert rows == [later, earlier]
    assert later["outcome_value_target"] == pytest.approx(1.0)
    assert earlier["outcome_value_target"] == pytest.approx(1.0)
    assert later["outcome_steps_to_terminal"] == 1
    assert earlier["outcome_steps_to_terminal"] == 2


@pytest.mark.parametrize(
    "keys",
    [
        (),
        [],
        ("",),
        (" ",),
        ("scenario_id", "scenario_id"),
        (1,),
    ],
)
def test_invalid_group_keys_are_atomic(keys: object) -> None:
    assert_rejected_without_target_mutation(
        [valid_row()],
        group_keys=keys,
        match="group_keys",
    )


@pytest.mark.parametrize(
    "scenario_id",
    [
        None,
        "",
        " ",
        True,
        np.bool_(True),
        -1,
        1.5,
        float("nan"),
        np.float32("nan"),
        float("inf"),
        float("-inf"),
        pd.NA,
        pd.NaT,
        np.datetime64("NaT"),
        [],
        {},
        set(),
        np.array([1]),
    ],
)
def test_invalid_scenario_id_is_value_error_and_atomic(
    scenario_id: object,
) -> None:
    assert_rejected_without_target_mutation(
        [valid_row(scenario_id=scenario_id)],
        match="scenario_id",
    )


@pytest.mark.parametrize(
    "scenario_id",
    [0, 1, np.int32(1), np.int64(1)],
)
def test_valid_scenario_id(scenario_id: object) -> None:
    rows = [valid_row(scenario_id=scenario_id)]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )
    assert rows[0]["outcome_value_target"] == 1.0


def test_custom_group_key_is_accepted() -> None:
    row = valid_row()
    row["episode_id"] = "episode-1"
    add_outcome_value_targets_to_rows(
        [row],
        gamma=TERMINAL_UTILITY_GAMMA,
        group_keys=("episode_id",),
    )
    assert row["outcome_value_target"] == 1.0


def _assert_second_group_atomic(
    bad_rows: list[dict[str, object]],
) -> None:
    rows = [
        valid_row(scenario_id=1, step=0),
        *bad_rows,
    ]
    assert_rejected_without_target_mutation(rows)


def test_atomic_when_second_group_is_unfinished() -> None:
    _assert_second_group_atomic(
        [
            valid_row(
                scenario_id=2,
                step=0,
                done=False,
            )
        ]
    )


def test_atomic_when_second_group_has_duplicate_step() -> None:
    _assert_second_group_atomic(
        [
            valid_row(scenario_id=2, step=0),
            valid_row(scenario_id=2, step=0),
        ]
    )


def test_atomic_when_second_group_has_mixed_reason() -> None:
    _assert_second_group_atomic(
        [
            valid_row(scenario_id=2, step=0),
            valid_row(
                scenario_id=2,
                step=1,
                termination_reason="max_steps_reached",
            ),
        ]
    )


def test_atomic_when_second_group_has_invalid_key() -> None:
    _assert_second_group_atomic(
        [
            valid_row(scenario_id=2, step=0),
            valid_row(scenario_id=None, step=1),
        ]
    )


def test_same_scenario_episodes_are_isolated() -> None:
    solved = valid_row(step=0)
    solved.update(
        run_id="run-a",
        iteration=1,
        episode_id="episode-a",
    )
    failed = valid_row(
        step=0,
        solved=False,
        termination_reason="max_steps_reached",
    )
    failed.update(
        run_id="run-b",
        iteration=2,
        episode_id="episode-b",
    )

    rows = [solved, failed]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )

    assert solved["outcome_value_target"] == 1.0
    assert failed["outcome_value_target"] == -1.0


def test_episode_rejects_mixed_identity() -> None:
    first = valid_row(step=0)
    second = valid_row(step=1)
    first.update(
        run_id="run-a",
        iteration=1,
        episode_id="episode-a",
    )
    second.update(
        run_id="run-b",
        iteration=1,
        episode_id="episode-a",
    )

    assert_rejected_without_target_mutation(
        [first, second],
        group_keys=("episode_id",),
        match="Mixed run_id",
    )


def test_terminal_value_from_outcome_public_helper() -> None:
    assert terminal_value_from_outcome(
        True,
        "solved",
    ) == (1.0, "solved")
    assert terminal_value_from_outcome(
        False,
        "handoff_to_redispatch",
    ) == (
        -1.0,
        "handoff_to_redispatch",
    )
