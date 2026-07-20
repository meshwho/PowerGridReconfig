from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.self_play.example_validation import (
    REQUIRED_OUTCOME_COLUMNS,
    load_and_validate_examples_csv,
    validate_example_outcome_contracts,
)
from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def write_state(path: Path, **overrides: object) -> Path:
    arrays = {
        "bus_features": np.zeros((2, 3), dtype=np.float32),
        "branch_features": np.zeros((1, 4), dtype=np.float32),
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "action_mask": np.array([True, True], dtype=bool),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


def valid_row(state_path: Path) -> dict[str, object]:
    return {
        "state_path": str(state_path),
        "mcts_policy_json": '{"0": 0.25, "1": 0.75}',
        "scenario_id": 1,
        "step": 0,
        "state_id": "state-1",
        "outcome_value_target": 1.0,
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        "solved": True,
        "done": True,
        "termination_reason": "solved",
        "outcome_class": "solved",
        "outcome_steps_to_terminal": 1,
        "outcome_value_target_mode": "alphazero_discounted",
        "outcome_gamma": 1.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def assert_rejected(path: Path, match: str = "") -> None:
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_and_validate_examples_csv(path)


def test_valid_examples_csv_is_accepted(tmp_path: Path) -> None:
    csv = write_csv(tmp_path / "examples.csv", [valid_row(write_state(tmp_path / "s.npz"))])
    df = load_and_validate_examples_csv(csv)
    assert len(df) == 1


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    csv = tmp_path / "examples.csv"
    csv.write_text("", encoding="utf-8")
    assert_rejected(csv, "no readable columns")


def test_header_only_csv_is_rejected(tmp_path: Path) -> None:
    csv = tmp_path / "examples.csv"
    csv.write_text(
        "state_path,mcts_policy_json,scenario_id,step,state_id,"
        "outcome_value_target,physical_objective_schema_version,"
        "outcome_value_target_contract_version,solved,done,"
        "termination_reason,outcome_class,outcome_steps_to_terminal,"
        "outcome_value_target_mode,outcome_gamma\n",
        encoding="utf-8",
    )
    assert_rejected(csv, "empty")


def test_positive_target_for_unsolved_episode_is_rejected(
    tmp_path: Path,
) -> None:
    row = valid_row(write_state(tmp_path / "s.npz"))
    row.update(
        {
            "solved": False,
            "done": True,
            "termination_reason": "max_steps_reached",
            "outcome_class": "max_steps_reached",
            "outcome_value_target": 1.0,
            "outcome_gamma": 1.0,
            "outcome_steps_to_terminal": 1,
        }
    )

    assert_rejected(
        write_csv(tmp_path / "examples.csv", [row]),
        "contradicts the terminal outcome",
    )

def test_zero_target_for_handoff_is_accepted(
    tmp_path: Path,
) -> None:
    row = valid_row(write_state(tmp_path / "s.npz"))
    row.update(
        {
            "solved": False,
            "done": True,
            "termination_reason": "handoff_to_redispatch",
            "outcome_class": "handoff_to_redispatch",
            "outcome_value_target": 0.0,
            "outcome_gamma": 0.95,
            "outcome_steps_to_terminal": 3,
        }
    )

    csv = write_csv(tmp_path / "examples.csv", [row])

    assert len(load_and_validate_examples_csv(csv)) == 1

def test_missing_required_columns_are_rejected(tmp_path: Path) -> None:
    csv = write_csv(tmp_path / "examples.csv", [{"state_path": "x"}])
    assert_rejected(csv, "missing required columns")


def test_legacy_contract_version_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz"))
    row["physical_objective_schema_version"] = 1
    assert_rejected(
        write_csv(tmp_path / "examples.csv", [row]),
        "legacy artifacts cannot be upgraded safely",
    )


def test_unknown_termination_reason_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz"))
    row["solved"] = False
    row["termination_reason"] = "reason_17"
    assert_rejected(
        write_csv(tmp_path / "examples.csv", [row]),
        "Unknown termination_reason",
    )


def test_null_required_value_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["state_id"] = ""
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "Missing required value")


def test_duplicate_state_id_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz")
    r1 = valid_row(s); r2 = valid_row(s)
    assert_rejected(write_csv(tmp_path / "examples.csv", [r1, r2]), "Duplicate state_id")


def test_fractional_scenario_id_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["scenario_id"] = 1.5
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "scenario_id")


def test_negative_step_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["step"] = -1
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "step")


def test_fractional_step_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["step"] = 1.5
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "step")


def test_non_finite_outcome_value_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["outcome_value_target"] = float("inf")
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "outcome_value_target")


def test_outcome_value_outside_range_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["outcome_value_target"] = 1.1
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "outside")


def test_invalid_policy_json_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = "{"
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "Invalid mcts_policy_json")


def test_policy_json_must_be_object(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = "[1]"
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "object")


def test_empty_policy_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = "{}"
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "empty")


def test_negative_policy_probability_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = '{"0": -0.1}'
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), ">= 0")


def test_non_finite_policy_probability_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = '{"0": NaN}'
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "finite")


def test_zero_policy_mass_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = '{"0": 0.0}'
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "mass")


def test_out_of_range_policy_action_is_rejected(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["mcts_policy_json"] = '{"2": 1.0}'
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "out of range")


def test_masked_policy_action_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", action_mask=np.array([True, False]))
    row = valid_row(s); row["mcts_policy_json"] = '{"1": 1.0}'
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "masked")


def test_missing_state_file_is_rejected(tmp_path: Path) -> None:
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(tmp_path / "missing.npz")]), "State file")


def test_corrupt_npz_is_rejected(tmp_path: Path) -> None:
    s = tmp_path / "s.npz"; s.write_bytes(b"bad")
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "Could not read")


def test_npz_missing_required_array_is_rejected(tmp_path: Path) -> None:
    s = tmp_path / "s.npz"; np.savez(s, bus_features=np.zeros((2, 3)))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "missing required arrays")


def test_invalid_bus_feature_shape_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", bus_features=np.zeros((2,)))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "bus_features")


def test_invalid_branch_feature_shape_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", branch_features=np.zeros((1,)))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "branch_features")


def test_invalid_edge_index_shape_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", edge_index=np.array([[0, 1]]))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "edge_index")


def test_invalid_action_mask_shape_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", action_mask=np.array([[True, True]]))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "action_mask")


def test_action_mask_requires_valid_action(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", action_mask=np.array([False, False]))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "valid action")


def test_non_finite_graph_features_are_rejected(tmp_path: Path) -> None:
    bus = np.zeros((2, 3), dtype=np.float32); bus[0, 0] = np.nan
    s = write_state(tmp_path / "s.npz", bus_features=bus)
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "finite")


def test_edge_index_out_of_bounds_is_rejected(tmp_path: Path) -> None:
    s = write_state(tmp_path / "s.npz", edge_index=np.array([[0], [2]], dtype=np.int64))
    assert_rejected(write_csv(tmp_path / "examples.csv", [valid_row(s)]), "out of bounds")


def test_inconsistent_graph_dimensions_are_rejected(tmp_path: Path) -> None:
    s1 = write_state(tmp_path / "s1.npz")
    s2 = write_state(tmp_path / "s2.npz", bus_features=np.zeros((3, 3)))
    r1 = valid_row(s1); r2 = valid_row(s2); r2["state_id"] = "state-2"
    assert_rejected(write_csv(tmp_path / "examples.csv", [r1, r2]), "dimensions mismatch")


def test_optional_selected_action_must_be_valid(tmp_path: Path) -> None:
    row = valid_row(write_state(tmp_path / "s.npz")); row["selected_action_id"] = 2
    assert_rejected(write_csv(tmp_path / "examples.csv", [row]), "selected_action_id")


def test_selected_action_may_be_absent_from_mcts_policy_support(tmp_path: Path) -> None:
    state = write_state(
        tmp_path / "s.npz",
        branch_features=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        action_mask=np.array([True, True, True], dtype=bool),
    )
    row = valid_row(state)
    row["selected_action_id"] = 0
    row["mcts_policy_json"] = '{"1": 0.7, "2": 0.3}'

    df = load_and_validate_examples_csv(write_csv(tmp_path / "examples.csv", [row]))

    assert len(df) == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("outcome_value_target", True),
        ("outcome_gamma", np.bool_(True)),
        ("outcome_steps_to_terminal", True),
    ],
)
def test_public_outcome_validator_rejects_boolean_numbers(
    tmp_path: Path, column: str, value: object
) -> None:
    row = valid_row(write_state(tmp_path / "s.npz"))
    row[column] = value
    with pytest.raises(ValueError, match=column):
        validate_example_outcome_contracts(pd.DataFrame([row]), source_path="unit")


@pytest.mark.parametrize("column", REQUIRED_OUTCOME_COLUMNS)
def test_public_outcome_validator_requires_every_outcome_column(column: str) -> None:
    row = valid_row(Path("unused.npz"))
    row.pop(column)
    with pytest.raises(ValueError, match=column):
        validate_example_outcome_contracts(pd.DataFrame([row]), source_path="unit")


def test_public_outcome_validator_accepts_independent_scenario_iterations() -> None:
    solved = valid_row(Path("unused-a.npz"))
    solved["replay_iteration"] = 1
    handoff = valid_row(Path("unused-b.npz"))
    handoff.update({
        "state_id": "state-2", "replay_iteration": 2, "solved": False,
        "termination_reason": "handoff_to_redispatch", "outcome_class": "handoff_to_redispatch",
        "outcome_value_target": 0.0, "outcome_gamma": 0.95,
    })
    validate_example_outcome_contracts(pd.DataFrame([solved, handoff]), source_path="unit")
