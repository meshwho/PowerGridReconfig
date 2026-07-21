import json

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.training.graph_policy_value import validate_no_scenario_overlap


def _csv_provenance() -> dict[str, object]:
    provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    return {
        **provenance,
        "physics_config": json.dumps(
            provenance["physics_config"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _write_fake_state(path):
    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        action_mask=np.array([True, True], dtype=bool),
        metadata_json=np.array(
            json.dumps(physics_provenance(DEFAULT_PHYSICS_CONFIG))
        ),
    )


def _write_examples_csv(path, state_path, scenario_ids):
    rows = []

    for i, scenario_id in enumerate(scenario_ids):
        rows.append(
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps({"0": 1.0}),
                "outcome_value_target": 0.0,
                "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
                "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
                **_csv_provenance(),
                "scenario_id": scenario_id,
                "step": i,
                "state_id": f"state_{i}",
                "solved": False,
                "done": True,
                "termination_reason": "handoff_to_redispatch",
                "outcome_class": "handoff_to_redispatch",
                "outcome_steps_to_terminal": 1,
                "outcome_value_target_mode": "alphazero_discounted",
                "outcome_gamma": 0.95,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_no_scenario_overlap_accepts_disjoint_splits(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path)

    train_csv = tmp_path / "examples_train.csv"
    val_csv = tmp_path / "examples_val.csv"

    _write_examples_csv(
        path=train_csv,
        state_path=state_path,
        scenario_ids=[1, 2, 3],
    )

    _write_examples_csv(
        path=val_csv,
        state_path=state_path,
        scenario_ids=[4, 5],
    )

    train_dataset = GraphSelfPlayDataset(
        examples_csv=train_csv,
        normalize_features=False,
    )

    val_dataset = GraphSelfPlayDataset(
        examples_csv=val_csv,
        normalize_features=False,
    )

    validate_no_scenario_overlap(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def test_validate_no_scenario_overlap_rejects_leakage(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path)

    train_csv = tmp_path / "examples_train.csv"
    val_csv = tmp_path / "examples_val.csv"

    _write_examples_csv(
        path=train_csv,
        state_path=state_path,
        scenario_ids=[1, 2, 3],
    )

    _write_examples_csv(
        path=val_csv,
        state_path=state_path,
        scenario_ids=[3, 4],
    )

    train_dataset = GraphSelfPlayDataset(
        examples_csv=train_csv,
        normalize_features=False,
    )

    val_dataset = GraphSelfPlayDataset(
        examples_csv=val_csv,
        normalize_features=False,
    )

    with pytest.raises(ValueError, match="scenario leakage"):
        validate_no_scenario_overlap(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )


def test_validate_no_scenario_overlap_accepts_missing_validation(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path)

    train_csv = tmp_path / "examples_train.csv"

    _write_examples_csv(
        path=train_csv,
        state_path=state_path,
        scenario_ids=[1, 2, 3],
    )

    train_dataset = GraphSelfPlayDataset(
        examples_csv=train_csv,
        normalize_features=False,
    )

    validate_no_scenario_overlap(
        train_dataset=train_dataset,
        val_dataset=None,
    )
