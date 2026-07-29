from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.topology_actions import GridFMAction


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


def _state(
    *,
    branch_status: tuple[int, int],
    loadings: tuple[float, float] = (50.0, 0.0),
):
    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, _LOADING_INDEX] = loadings

    return SimpleNamespace(
        scenario_id=1,
        outaged_branch_ids=tuple(
            branch_id
            for branch_id, status in zip((10, 20), branch_status)
            if status == 0
        ),
        branch_ids=np.asarray([10, 20], dtype=np.int64),
        branch_status=np.asarray(branch_status, dtype=np.int64),
        branch_features=branch_features,
        bus_features=np.zeros((2, 1), dtype=np.float32),
        edge_index=np.asarray([[0, 0], [1, 1]], dtype=np.int64),
    )


def test_inactive_tie_line_keeps_its_slot_and_bypasses_opening_threshold():
    state = _state(branch_status=(1, 0))
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        closeable_branch_ids=(20,),
        enable_cache=False,
    )

    actions = action_space.build_all_actions(state)
    mask = action_space.operational_action_mask(state)

    assert [action.action_id for action in actions] == [0, 1, 2]
    assert actions[1].action_type == "switch_off_branch"
    assert actions[1].target_status == 0
    assert actions[2].action_type == "switch_on_branch"
    assert actions[2].target_status == 1

    # Branch 10 is below the opening threshold. Branch 20 is a configured
    # closure and must not be filtered by its zero pre-closure loading.
    assert mask.tolist() == [True, False, True]


def test_branch_slot_identity_does_not_change_with_branch_status():
    open_state = _state(branch_status=(1, 0))
    closed_state = _state(branch_status=(0, 1))
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        closeable_branch_ids=(10, 20),
        enable_cache=False,
    )

    open_actions = action_space.build_all_actions(open_state)
    closed_actions = action_space.build_all_actions(closed_state)

    assert [action.action_id for action in open_actions] == [0, 1, 2]
    assert [action.action_id for action in closed_actions] == [0, 1, 2]
    assert open_actions[1].target_status == 0
    assert closed_actions[1].target_status == 1
    assert open_actions[2].target_status == 1
    assert closed_actions[2].target_status == 0


def test_backend_applies_both_branch_status_directions():
    branch_df = pd.DataFrame(
        {
            "idx": [10, 20],
            "br_status": [1.0, 0.0],
        }
    )

    GridFMPowerFlowBackend._apply_branch_status(
        branch_df,
        branch_id=10,
        target_status=0,
        context="test grid",
    )
    GridFMPowerFlowBackend._apply_branch_status(
        branch_df,
        branch_id=20,
        target_status=1,
        context="test grid",
    )

    assert branch_df["br_status"].tolist() == [0.0, 1.0]

    with pytest.raises(ValueError, match="already has status 1"):
        GridFMPowerFlowBackend._apply_branch_status(
            branch_df,
            branch_id=20,
            target_status=1,
            context="test grid",
        )


def test_backend_cache_key_uses_topology_after_the_action():
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        enable_cache=False,
    )

    close_state = SimpleNamespace(
        scenario_id=7,
        outaged_branch_ids=[20],
    )
    close_action = GridFMAction(
        action_id=2,
        action_type="switch_on_branch",
        branch_id=20,
        branch_pos=1,
    )

    open_state = SimpleNamespace(
        scenario_id=7,
        outaged_branch_ids=[],
    )
    open_action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )

    close_key = backend._make_cache_key_from_state(
        close_state,
        action=close_action,
    )
    open_key = backend._make_cache_key_from_state(
        open_state,
        action=open_action,
    )

    assert close_key[-1] == ()
    assert open_key[-1] == (10,)
    assert close_key != open_key
