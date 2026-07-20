import json

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import TerminationReason


def _write_fake_state(path):
    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        action_mask=np.array([True, True], dtype=bool),
    )


def test_dataset_rejects_legacy_csv_without_outcome_value_target(tmp_path):
    state_path = tmp_path / "state_0.npz"
    _write_fake_state(state_path)

    examples_csv = tmp_path / "examples.csv"

    df = pd.DataFrame(
        [
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps({"0": 1.0}),
                "discounted_return_from_step": 500.0,
                "scenario_id": 1,
                "step": 0,
                "state_id": "state_0",
                "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
                "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
                "solved": False,
                "termination_reason": "max_steps_reached",
            }
        ]
    )

    df.to_csv(examples_csv, index=False)

    with pytest.raises(ValueError, match="outcome_value_target"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


def test_dataset_reads_strict_outcome_value_target(tmp_path):
    state_path = tmp_path / "state_0.npz"
    _write_fake_state(state_path)

    examples_csv = tmp_path / "examples.csv"

    df = pd.DataFrame(
        [
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps({"0": 1.0}),
                "outcome_value_target": 0.95,
                "scenario_id": 1,
                "step": 0,
                "state_id": "state_0",
                "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
                "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
                "solved": True,
                "done": True,
                "termination_reason": TerminationReason.SOLVED.value,
                "outcome_class": TerminationReason.SOLVED.value,
                "outcome_steps_to_terminal": 1,
                "outcome_value_target_mode": "alphazero_discounted",
                "outcome_gamma": 0.95,
            }
        ]
    )

    df.to_csv(examples_csv, index=False)

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    assert float(sample["target_value"].item()) == pytest.approx(0.95)
