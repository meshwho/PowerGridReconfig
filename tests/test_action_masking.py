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
from grid_topology_ai.return_contract import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
)


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


def _state_metadata() -> np.ndarray:
    return np.array(json.dumps(physics_provenance(DEFAULT_PHYSICS_CONFIG)))


def _write_fake_state(path, action_mask):
    """Minimal graph state for action masking tests."""

    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        action_mask=np.asarray(action_mask, dtype=bool),
        metadata_json=_state_metadata(),
    )


def _write_examples_csv(
    path,
    state_path,
    mcts_policy,
    outcome_value_target=-1.0,
):
    pd.DataFrame(
        [
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps(mcts_policy),
                "outcome_value_target": outcome_value_target,
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                **_csv_provenance(),
                "scenario_id": 1,
                "step": 0,
                "state_id": "state_0",
                "solved": False,
                "done": True,
                "termination_reason": "handoff_to_redispatch",
                "outcome_class": "handoff_to_redispatch",
                "outcome_steps_to_terminal": 1,
                "outcome_value_target_mode": VALUE_TARGET_MODE,
                "outcome_gamma": TERMINAL_UTILITY_GAMMA,
            }
        ]
    ).to_csv(path, index=False)


def test_dataset_rejects_probability_on_masked_action(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, False])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(
        examples_csv,
        state_path,
        {"1": 0.25, "2": 0.75},
    )

    with pytest.raises(ValueError, match="masked"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


def test_dataset_keeps_stop_action_when_it_is_valid(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, True])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(examples_csv, state_path, {"0": 1.0})

    policy = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )[0]["target_policy"].numpy()

    assert policy[0] == pytest.approx(1.0)
    assert policy[1] == pytest.approx(0.0)
    assert policy[2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)


def test_dataset_rejects_policy_with_only_masked_actions(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, False])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(examples_csv, state_path, {"2": 1.0})

    with pytest.raises(ValueError, match="masked"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


def test_dataset_rejects_policy_actions_outside_action_space(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, True])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(
        examples_csv,
        state_path,
        {"1": 0.25, "99": 0.75},
    )

    with pytest.raises(ValueError, match="out of range"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


@pytest.mark.parametrize(
    "policy",
    [
        {"1": 0.8},
        {"1": 1.2},
    ],
)
def test_dataset_rejects_non_unit_policy_mass(tmp_path, policy):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, True])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(examples_csv, state_path, policy)

    with pytest.raises(ValueError, match="sum to 1"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


def test_dataset_preserves_valid_policy_probabilities(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True, True])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(
        examples_csv,
        state_path,
        {"1": 0.25, "2": 0.75},
    )

    policy = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )[0]["target_policy"].numpy()

    np.testing.assert_allclose(
        policy,
        np.array([0.0, 0.25, 0.75], dtype=np.float32),
    )


def test_dataset_rejects_wrong_action_mask_length(tmp_path):
    state_path = tmp_path / "state.npz"
    _write_fake_state(state_path, [True, True])
    examples_csv = tmp_path / "examples.csv"
    _write_examples_csv(examples_csv, state_path, {"0": 1.0})

    with pytest.raises(
        ValueError,
        match=r"action_mask must be 1D with 3 entries",
    ):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )


def test_npz_loader_closes_file_before_returning_arrays(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "state.npz"
    state_path.write_bytes(b"placeholder")

    arrays = {
        "bus_features": np.zeros((2, 3), dtype=np.float32),
        "branch_features": np.zeros((2, 4), dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "branch_status": np.ones(2, dtype=np.float32),
        "action_mask": np.ones(3, dtype=bool),
    }

    class FakeNpz:
        def __init__(self):
            self.files = list(arrays)
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.closed = True

        def __getitem__(self, name):
            return arrays[name]

    fake_npz = FakeNpz()
    monkeypatch.setattr(
        "grid_topology_ai.models.graph_self_play_dataset.np.load",
        lambda *args, **kwargs: fake_npz,
    )

    dataset = object.__new__(GraphSelfPlayDataset)
    dataset.examples = pd.DataFrame(
        [{"state_path": str(state_path)}]
    )

    loaded = dataset._load_npz_by_index(0)

    assert fake_npz.closed is True
    for name, value in arrays.items():
        assert np.array_equal(loaded[name], value)
        assert loaded[name] is not value
