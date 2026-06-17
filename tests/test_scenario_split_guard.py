import json

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from scripts.self_play.train_graph_baseline import validate_no_scenario_overlap


def _write_fake_state(path):
    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        action_mask=np.array([True, True], dtype=bool),
    )


def _write_examples_csv(path, state_path, scenario_ids):
    rows = []

    for i, scenario_id in enumerate(scenario_ids):
        rows.append(
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps({"0": 1.0}),
                "outcome_value_target": 0.0,
                "scenario_id": scenario_id,
                "step": i,
                "state_id": f"state_{i}",
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