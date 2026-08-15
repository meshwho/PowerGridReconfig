from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS


def _state(scenario_id: int) -> SimpleNamespace:
    branch_features = np.zeros(
        (3, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = [
        50.0,
        90.0,
        120.0,
    ]
    return SimpleNamespace(
        scenario_id=scenario_id,
        branch_ids=np.array([10, 11, 12], dtype=np.int64),
        branch_status=np.array([1, 1, 1], dtype=np.int8),
        branch_features=branch_features,
        edge_index=np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int64),
        bus_features=np.zeros((3, 1), dtype=np.float32),
        outaged_branch_ids=(),
    )


def test_scenario_id_does_not_prevent_structural_cache_hit() -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=True,
    )

    action_space.structural_action_mask(_state(1))
    before = action_space.cache_info()
    action_space.structural_action_mask(_state(999))
    after = action_space.cache_info()

    assert int(after["hits"]) == int(before["hits"]) + 1
    assert int(after["size"]) == 1
