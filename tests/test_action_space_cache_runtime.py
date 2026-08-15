from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS


def _state(statuses: tuple[int, ...]) -> SimpleNamespace:
    count = len(statuses)
    branch_features = np.zeros(
        (count, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = 100.0
    return SimpleNamespace(
        scenario_id=1,
        branch_ids=np.arange(100, 100 + count, dtype=np.int64),
        branch_status=np.asarray(statuses, dtype=np.int8),
        branch_features=branch_features,
        edge_index=np.asarray(
            [(index, index + 1) for index in range(count)],
            dtype=np.int64,
        ).T,
        bus_features=np.zeros((count + 1, 1), dtype=np.float32),
        outaged_branch_ids=[],
    )


def test_action_cache_can_be_disabled_without_changing_masks() -> None:
    state = _state((1, 1, 1))
    cached = GridFMActionSpace(
        require_connected_after_switch=False,
        enable_cache=True,
        structural_cache_max_bytes=4096,
    )
    uncached = GridFMActionSpace(
        require_connected_after_switch=False,
        enable_cache=False,
        structural_cache_max_bytes=0,
    )

    np.testing.assert_array_equal(
        cached.valid_action_mask(state),
        uncached.valid_action_mask(state),
    )


def test_action_cache_never_exceeds_configured_bytes() -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        enable_cache=True,
        structural_cache_max_bytes=512,
    )

    for status in (
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 1),
    ):
        action_space.structural_action_mask(_state(status))
        info = action_space.cache_info()
        assert int(info["bytes"]) <= int(info["max_bytes"])
