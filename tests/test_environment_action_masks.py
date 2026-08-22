from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.actions import GridFMActionSpace
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.environment import TopologySwitchingEnv


def _state(*, loading: float) -> SimpleNamespace:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index(
            "loading_percent"
        ),
    ] = float(loading)

    return SimpleNamespace(
        scenario_id=1,
        outaged_branch_ids=(),
        branch_ids=np.asarray(
            [100],
            dtype=np.int64,
        ),
        branch_status=np.asarray(
            [1],
            dtype=np.int8,
        ),
        branch_features=branch_features,
        edge_index=np.asarray(
            [[0], [1]],
            dtype=np.int64,
        ),
        bus_features=np.zeros(
            (2, 1),
            dtype=np.float32,
        ),
    )


def test_environment_exposes_structural_and_operational_masks() -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=False,
    )
    env = TopologySwitchingEnv(
        adapter=object(),  # type: ignore[arg-type]
        backend=object(),  # type: ignore[arg-type]
        action_space=action_space,
        reward_fn=object(),  # type: ignore[arg-type]
        max_steps=1,
    )
    env.current_state = _state(
        loading=50.0
    )
    env.initial_scenario_id = 1
    env.done = False

    structural = env.structural_action_mask()
    operational = env.operational_action_mask()

    assert structural.tolist() == [
        True,
        True,
    ]
    assert operational.tolist() == [
        True,
        False,
    ]
