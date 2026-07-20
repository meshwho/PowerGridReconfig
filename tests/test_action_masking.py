import json

import numpy as np
import pandas as pd
import pytest
from grid_topology_ai.contracts import OUTCOME_VALUE_TARGET_CONTRACT_VERSION
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import TerminationReason

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset


def _write_fake_state(
    path,
    action_mask,
):
    """
    Minimal graph state for action masking tests.

    We use:
    - 2 buses
    - 2 branches
    - 3 actions total:
        action 0 = stop / handoff
        action 1 = switch branch 0
        action 2 = switch branch 1
    """

    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int64,
        ),
        action_mask=np.asarray(action_mask, dtype=bool),
    )


def _write_examples_csv(
    path,
    state_path,
    mcts_policy,
    outcome_value_target=0.0,
):
    df = pd.DataFrame(
        [
            {
                "state_path": str(state_path),
                "mcts_policy_json": json.dumps(mcts_policy),
                "outcome_value_target": outcome_value_target,
                "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
                "outcome_value_target_contract_version": OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
                "solved": False,
                "done": True,
                "termination_reason": TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value,
                "outcome_class": TerminationReason.HANDOFF_TO_REDISPATCH.value,
                "outcome_steps_to_terminal": 1,
                "outcome_value_target_mode": "alphazero_discounted",
                "outcome_gamma": 0.95,
                "scenario_id": 1,
                "step": 0,
                "state_id": "state_0",
            }
        ]
    )

    df.to_csv(path, index=False)


def test_dataset_masks_invalid_actions_and_renormalizes_policy(tmp_path):
    state_path = tmp_path / "state.npz"

    # action 2 is invalid
    _write_fake_state(
        path=state_path,
        action_mask=[True, True, False],
    )

    examples_csv = tmp_path / "examples.csv"

    # Teacher policy assigns probability to both valid action 1 and invalid action 2.
    _write_examples_csv(
        path=examples_csv,
        state_path=state_path,
        mcts_policy={
            "1": 0.25,
            "2": 0.75,
        },
    )

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    policy = sample["target_policy"].numpy()

    assert policy[0] == pytest.approx(0.0)
    assert policy[1] == pytest.approx(1.0)
    assert policy[2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)


def test_dataset_keeps_stop_action_when_it_is_valid(tmp_path):
    state_path = tmp_path / "state.npz"

    # all actions are valid, including action 0
    _write_fake_state(
        path=state_path,
        action_mask=[True, True, True],
    )

    examples_csv = tmp_path / "examples.csv"

    _write_examples_csv(
        path=examples_csv,
        state_path=state_path,
        mcts_policy={
            "0": 1.0,
        },
    )

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    policy = sample["target_policy"].numpy()

    assert policy[0] == pytest.approx(1.0)
    assert policy[1] == pytest.approx(0.0)
    assert policy[2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)


def test_dataset_handles_policy_with_only_invalid_masked_actions(tmp_path):
    state_path = tmp_path / "state.npz"

    # action 2 is invalid
    _write_fake_state(
        path=state_path,
        action_mask=[True, True, False],
    )

    examples_csv = tmp_path / "examples.csv"

    # Teacher policy contains only an action that is masked out.
    _write_examples_csv(
        path=examples_csv,
        state_path=state_path,
        mcts_policy={
            "2": 1.0,
        },
    )

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    policy = sample["target_policy"].numpy()

    assert np.isfinite(policy).all()
    assert policy[0] == pytest.approx(0.0)
    assert policy[1] == pytest.approx(0.0)
    assert policy[2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(0.0)


def test_dataset_ignores_policy_actions_outside_action_space(tmp_path):
    state_path = tmp_path / "state.npz"

    _write_fake_state(
        path=state_path,
        action_mask=[True, True, True],
    )

    examples_csv = tmp_path / "examples.csv"

    # action 99 is outside the action space and must be ignored.
    _write_examples_csv(
        path=examples_csv,
        state_path=state_path,
        mcts_policy={
            "1": 0.25,
            "99": 0.75,
        },
    )

    dataset = GraphSelfPlayDataset(
        examples_csv=examples_csv,
        normalize_features=False,
    )

    sample = dataset[0]

    policy = sample["target_policy"].numpy()

    assert policy[0] == pytest.approx(0.0)
    assert policy[1] == pytest.approx(1.0)
    assert policy[2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)


def test_dataset_rejects_wrong_action_mask_length(tmp_path):
    state_path = tmp_path / "state.npz"

    # There are 2 branches, so expected actions = 2 + 1 = 3.
    # Here we intentionally provide only 2 actions.
    _write_fake_state(
        path=state_path,
        action_mask=[True, True],
    )

    examples_csv = tmp_path / "examples.csv"

    _write_examples_csv(
        path=examples_csv,
        state_path=state_path,
        mcts_policy={
            "0": 1.0,
        },
    )

    with pytest.raises(ValueError, match="Expected num_actions = num_branches"):
        GraphSelfPlayDataset(
            examples_csv=examples_csv,
            normalize_features=False,
        )
