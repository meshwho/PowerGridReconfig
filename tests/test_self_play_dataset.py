from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.models.self_play_dataset import SelfPlayDataset
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def state(path: Path) -> Path:
    np.savez(
        path,
        bus_features=np.zeros((2, 3), np.float32),
        branch_features=np.zeros((1, 4), np.float32),
        edge_index=np.array([[0], [1]]),
        action_mask=np.array([True, True]),
    )
    return path


def row(path: Path) -> dict[str, object]:
    return {
        "state_path": str(path),
        "mcts_policy_json": '{"0": 1.0}',
        "scenario_id": 1,
        "step": 0,
        "state_id": "row-1",
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        "solved": False,
        "done": True,
        "termination_reason": "handoff_to_redispatch",
        "outcome_class": "handoff_to_redispatch",
        "outcome_steps_to_terminal": 1,
        "outcome_value_target_mode": "alphazero_discounted",
        "outcome_gamma": 0.95,
        "outcome_value_target": 0.0,
    }


def write(path: Path, value: dict[str, object]) -> Path:
    pd.DataFrame([value]).to_csv(path, index=False)
    return path


def test_flat_dataset_accepts_valid_strict_row(tmp_path: Path) -> None:
    dataset = SelfPlayDataset(
        write(tmp_path / "examples.csv", row(state(tmp_path / "state.npz")))
    )
    assert len(dataset) == 1
    assert dataset[0]["target_value"].item() == pytest.approx(0.0)


def test_flat_dataset_rejects_positive_handoff_target(tmp_path: Path) -> None:
    value = row(tmp_path / "missing.npz")
    value["outcome_value_target"] = 0.95
    with pytest.raises(ValueError, match="outcome_value_target"):
        SelfPlayDataset(write(tmp_path / "examples.csv", value))


def test_flat_dataset_rejects_legacy_outcome_version(tmp_path: Path) -> None:
    value = row(tmp_path / "missing.npz")
    value["outcome_value_target_contract_version"] = 1
    with pytest.raises(ValueError, match="outcome/value-target contract"):
        SelfPlayDataset(write(tmp_path / "examples.csv", value))


def test_flat_dataset_rejects_missing_outcome_column(tmp_path: Path) -> None:
    value = row(tmp_path / "missing.npz")
    del value["outcome_gamma"]
    with pytest.raises(ValueError, match="missing required columns"):
        SelfPlayDataset(write(tmp_path / "examples.csv", value))


def test_flat_dataset_validates_outcome_before_state_loading(tmp_path: Path) -> None:
    value = row(tmp_path / "does-not-exist.npz")
    value["outcome_value_target"] = 0.95
    with pytest.raises(ValueError, match="outcome_value_target"):
        SelfPlayDataset(write(tmp_path / "examples.csv", value))
