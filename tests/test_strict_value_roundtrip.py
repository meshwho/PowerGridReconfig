import json

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def _current(rows):
    for row in rows:
        row["physical_objective_schema_version"] = (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        )
    return rows

def _write_fake_state(path):
    """
    Minimal valid graph state:

    - 2 buses
    - 1 branch
    - 2 actions:
        action 0 = stop / handoff
        action 1 = switch branch 0
    """

    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        action_mask=np.array([True, True], dtype=bool),
    )


def test_outcome_value_target_roundtrip_from_generator_to_dataset(tmp_path):
    state_0 = tmp_path / "state_0.npz"
    state_1 = tmp_path / "state_1.npz"

    _write_fake_state(state_0)
    _write_fake_state(state_1)

    rows = [
        {
            "state_path": str(state_0),
            "mcts_policy_json": json.dumps({"1": 1.0}),
            "scenario_id": 10,
            "step": 0,
            "state_id": "state_0",
            "solved": False,
            "done": False,
            "termination_reason": "",
        },
        {
            "state_path": str(state_1),
            "mcts_policy_json": json.dumps({"1": 1.0}),
            "scenario_id": 10,
            "step": 1,
            "state_id": "state_1",
            "solved": True,
            "done": True,
            "termination_reason": "solved",
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.9)

    examples_csv = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(examples_csv, index=False)

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample_0 = dataset[0]
    sample_1 = dataset[1]

    assert sample_0["target_value"].item() == pytest.approx(0.9**2)
    assert sample_1["target_value"].item() == pytest.approx(0.9)

    assert sample_0["target_policy"].shape == torch.Size([2])
    assert sample_1["target_policy"].shape == torch.Size([2])

    assert sample_0["target_policy"].sum().item() == pytest.approx(1.0)
    assert sample_1["target_policy"].sum().item() == pytest.approx(1.0)


def test_roundtrip_handoff_target_is_zero(tmp_path):
    state_0 = tmp_path / "state_0.npz"
    _write_fake_state(state_0)

    rows = [
        {
            "state_path": str(state_0),
            "mcts_policy_json": json.dumps({"0": 1.0}),
            "scenario_id": 11,
            "step": 0,
            "state_id": "state_0",
            "solved": False,
            "done": True,
            "termination_reason": "handoff_to_redispatch_teacher",
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.95)

    examples_csv = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(examples_csv, index=False)

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    assert sample["target_value"].item() == pytest.approx(0.0)
    assert sample["target_policy"][0].item() == pytest.approx(1.0)


def test_roundtrip_failed_target_is_negative(tmp_path):
    state_0 = tmp_path / "state_0.npz"
    _write_fake_state(state_0)

    rows = [
        {
            "state_path": str(state_0),
            "mcts_policy_json": json.dumps({"1": 1.0}),
            "scenario_id": 12,
            "step": 0,
            "state_id": "state_0",
            "solved": False,
            "done": True,
            "termination_reason": "max_steps_reached",
        },
    ]

    add_outcome_value_targets_to_rows(_current(rows), gamma=0.95)

    examples_csv = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(examples_csv, index=False)

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    assert sample["target_value"].item() == pytest.approx(-0.95)
