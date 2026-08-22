import json

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.reward import TERMINAL_UTILITY_GAMMA, VALUE_TARGET_MODE
from grid_topology_ai.state.schema import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.topology_actions import (
    ActionSpaceConfig,
    action_layout_fingerprint,
    action_layout_to_list,
    build_branch_action_slots,
)
from tests.outcome_evidence_helpers import terminal_evidence


_ACTION_CONFIG = ActionSpaceConfig()
_BRANCH_IDS = (0, 1)
_ACTION_LAYOUT = build_branch_action_slots(_BRANCH_IDS)
_ACTION_LAYOUT_PAYLOAD = action_layout_to_list(_ACTION_LAYOUT)
_EVIDENCE = terminal_evidence("handoff_to_redispatch")
_RUN_ID = "test-run"
_ITERATION = 1
_EPISODE_ID = "test-episode-1"


def _state_metadata() -> np.ndarray:
    metadata = {
        "run_id": _RUN_ID,
        "iteration": _ITERATION,
        "episode_id": _EPISODE_ID,
        "physics_config": DEFAULT_PHYSICS_CONFIG.to_dict(),
        "topology_action_config": _ACTION_CONFIG.to_contract_dict(),
        "action_layout": _ACTION_LAYOUT_PAYLOAD,
        "terminal_outcome_evidence": _EVIDENCE.to_dict(),
    }
    return np.array(json.dumps(metadata))


def _write_fake_state(path, action_mask):
    """Write a minimal current Light graph state for policy-mask tests."""

    branch_features = np.zeros(
        (len(_BRANCH_IDS), len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[
        :,
        BRANCH_FEATURE_COLUMNS.index("br_status"),
    ] = 1.0

    np.savez(
        path,
        bus_features=np.zeros(
            (2, len(BUS_FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        branch_features=branch_features,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        bus_ids=np.array([0, 1], dtype=np.int64),
        branch_ids=np.array(_BRANCH_IDS, dtype=np.int64),
        branch_status=np.ones(len(_BRANCH_IDS), dtype=np.float32),
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
                "physics_config": json.dumps(
                    DEFAULT_PHYSICS_CONFIG.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "topology_action_config": json.dumps(
                    _ACTION_CONFIG.to_contract_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "action_layout": json.dumps(
                    _ACTION_LAYOUT_PAYLOAD,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "action_layout_fingerprint": action_layout_fingerprint(
                    _ACTION_LAYOUT
                ),
                "run_id": _RUN_ID,
                "iteration": _ITERATION,
                "episode_id": _EPISODE_ID,
                "scenario_id": 1,
                "step": 0,
                "state_id": "state_0",
                "solved": False,
                "done": True,
                "termination_reason": "handoff_to_redispatch",
                "terminal_outcome_evidence_json": _EVIDENCE.to_json(),
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

    with pytest.raises(ValueError, match="must equal 1.0"):
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
