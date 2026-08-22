from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.actions import GridFMActionSpace
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS


def _state(
    *,
    loadings: list[float],
    statuses: list[int] | None = None,
    edges: list[tuple[int, int]] | None = None,
    scenario_id: int = 1,
) -> SimpleNamespace:
    branch_count = len(loadings)
    if statuses is None:
        statuses = [1] * branch_count
    if len(statuses) != branch_count:
        raise ValueError("statuses and loadings must have the same length")

    if edges is None:
        edges = [
            (branch_pos, branch_pos + 1)
            for branch_pos in range(branch_count)
        ]
    if len(edges) != branch_count:
        raise ValueError("edges and loadings must have the same length")

    branch_ids = np.arange(100, 100 + branch_count, dtype=np.int64)
    branch_status = np.asarray(statuses, dtype=np.int8)
    branch_features = np.zeros(
        (branch_count, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    branch_features[:, loading_idx] = np.asarray(loadings, dtype=np.float32)

    edge_index = np.asarray(edges, dtype=np.int64).T
    num_buses = int(edge_index.max()) + 1 if edge_index.size else 1

    return SimpleNamespace(
        scenario_id=int(scenario_id),
        outaged_branch_ids=tuple(
            int(branch_id)
            for branch_id, status in zip(
                branch_ids,
                branch_status,
                strict=True,
            )
            if int(status) <= 0
        ),
        branch_ids=branch_ids,
        branch_status=branch_status,
        branch_features=branch_features,
        edge_index=edge_index,
        bus_features=np.zeros((num_buses, 1), dtype=np.float32),
    )


def test_loading_filter_applies_only_to_operational_mask() -> None:
    state = _state(loadings=[50.0, 90.0])
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=False,
    )

    structural = action_space.structural_action_mask(state)
    operational = action_space.operational_action_mask(state)

    assert structural.tolist() == [True, True, True]
    assert operational.tolist() == [True, False, True]
    assert [a.action_id for a in action_space.valid_actions(state)] == [0, 2]
    assert [a.action_id for a in action_space.invalid_actions(state)] == [1]


def test_inactive_branch_is_rejected_by_both_mask_layers() -> None:
    state = _state(loadings=[90.0, 90.0], statuses=[1, 0])
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=False,
    )

    assert action_space.structural_action_mask(state).tolist() == [
        True,
        True,
        False,
    ]
    assert action_space.operational_action_mask(state).tolist() == [
        True,
        True,
        False,
    ]


def test_connectivity_bridge_is_rejected_by_both_layers() -> None:
    state = _state(
        loadings=[90.0, 90.0],
        edges=[(0, 1), (1, 2)],
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=0.0,
        enable_cache=False,
    )

    assert action_space.structural_action_mask(state).tolist() == [
        True,
        False,
        False,
    ]
    assert action_space.operational_action_mask(state).tolist() == [
        True,
        False,
        False,
    ]


def test_disabled_loading_filter_makes_masks_equal() -> None:
    state = _state(loadings=[0.0, 150.0])
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=0.0,
        enable_cache=False,
    )

    np.testing.assert_array_equal(
        action_space.operational_action_mask(state),
        action_space.structural_action_mask(state),
    )


def test_dynamic_loading_is_never_cached() -> None:
    low = _state(loadings=[50.0], scenario_id=7)
    high = _state(loadings=[90.0], scenario_id=7)
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )

    np.testing.assert_array_equal(
        action_space.structural_action_mask(low),
        action_space.structural_action_mask(high),
    )
    assert action_space.operational_action_mask(low).tolist() == [True, False]
    assert action_space.operational_action_mask(high).tolist() == [True, True]

    info = action_space.cache_info()
    assert info["size"] == 1
    assert info["hits"] >= 1
    assert info["bytes"] <= info["max_bytes"]


def test_cached_masks_are_returned_as_copies() -> None:
    state = _state(loadings=[90.0])
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )

    structural = action_space.structural_action_mask(state)
    operational = action_space.operational_action_mask(state)
    structural[:] = False
    operational[:] = False

    assert action_space.structural_action_mask(state).tolist() == [True, True]
    assert action_space.operational_action_mask(state).tolist() == [True, True]


def test_clear_cache_resets_single_structural_cache() -> None:
    state = _state(loadings=[90.0])
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )

    action_space.structural_action_mask(state)
    action_space.operational_action_mask(state)
    action_space.valid_actions(state)
    assert action_space.cache_info()["size"] == 1

    action_space.clear_cache()
    cleared = action_space.cache_info()
    assert cleared["size"] == 0
    assert cleared["bytes"] == 0
    assert cleared["hits"] == 0
    assert cleared["misses"] == 0
