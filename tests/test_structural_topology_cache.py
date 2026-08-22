from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.actions import GridFMActionSpace
from grid_topology_ai.actions import (
    StructuralTopologyCache,
    structural_topology_fingerprint,
)
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS


def _state(
    *,
    scenario_id: int,
    loadings: tuple[float, ...] = (90.0, 80.0, 70.0),
    statuses: tuple[int, ...] = (1, 1, 1),
    edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2)),
) -> SimpleNamespace:
    branch_ids = np.arange(10, 10 + len(statuses), dtype=np.int64)
    branch_status = np.asarray(statuses, dtype=np.int8)
    branch_features = np.zeros(
        (len(statuses), len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loadings
    edge_index = np.asarray(edges, dtype=np.int64).T
    return SimpleNamespace(
        scenario_id=scenario_id,
        branch_ids=branch_ids,
        branch_status=branch_status,
        branch_features=branch_features,
        edge_index=edge_index,
        bus_features=np.zeros((3, 1), dtype=np.float32),
        outaged_branch_ids=[
            int(branch_id)
            for branch_id, status in zip(branch_ids, branch_status, strict=True)
            if status <= 0
        ],
    )


def test_structural_key_ignores_scenario_and_operational_loading() -> None:
    first = _state(scenario_id=1, loadings=(20.0, 30.0, 40.0))
    second = _state(scenario_id=999, loadings=(120.0, 130.0, 140.0))

    first_key = structural_topology_fingerprint(
        first,
        require_connected_after_switch=True,
        closeable_branch_ids=(),
    )
    second_key = structural_topology_fingerprint(
        second,
        require_connected_after_switch=True,
        closeable_branch_ids=(),
    )

    assert first_key == second_key


def test_structural_key_changes_with_topology_or_graph_contract() -> None:
    original = _state(scenario_id=1)
    changed_status = _state(scenario_id=1, statuses=(1, 0, 1))
    changed_edges = _state(
        scenario_id=1,
        edges=((0, 1), (1, 2), (1, 2)),
    )

    def key(state, *, connected=True, closeable=()):
        return structural_topology_fingerprint(
            state,
            require_connected_after_switch=connected,
            closeable_branch_ids=closeable,
        )

    baseline = key(original)
    assert key(changed_status) != baseline
    assert key(changed_edges) != baseline
    assert key(original, connected=False) != baseline
    assert key(original, closeable=(12,)) != baseline


def test_structural_cache_is_bounded_and_returns_independent_masks() -> None:
    state = _state(scenario_id=1)
    cache = StructuralTopologyCache(max_bytes=1024)
    key, cached = cache.lookup(
        state,
        require_connected_after_switch=True,
        closeable_branch_ids=(),
    )
    assert cached is None

    mask = np.array([True, True, False, True], dtype=bool)
    assert cache.store(key, mask)

    _key, first = cache.lookup(
        state,
        require_connected_after_switch=True,
        closeable_branch_ids=(),
    )
    assert first is not None
    first[:] = False

    _key, second = cache.lookup(
        state,
        require_connected_after_switch=True,
        closeable_branch_ids=(),
    )
    assert second is not None
    assert second.tolist() == mask.tolist()
    info = cache.info()
    assert info["bytes"] <= info["max_bytes"]


def test_cached_and_uncached_action_spaces_are_identical() -> None:
    state = _state(scenario_id=7, loadings=(50.0, 90.0, 100.0))
    cached = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=80.0,
        enable_cache=True,
    )
    uncached = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=80.0,
        enable_cache=False,
    )

    np.testing.assert_array_equal(
        cached.structural_action_mask(state),
        uncached.structural_action_mask(state),
    )
    np.testing.assert_array_equal(
        cached.operational_action_mask(state),
        uncached.operational_action_mask(state),
    )
    assert [a.action_id for a in cached.valid_actions(state)] == [
        a.action_id for a in uncached.valid_actions(state)
    ]


def test_structural_cache_reuses_same_topology_across_scenarios() -> None:
    first = _state(scenario_id=1, loadings=(20.0, 30.0, 40.0))
    second = _state(scenario_id=2, loadings=(120.0, 130.0, 140.0))
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=100.0,
        enable_cache=True,
    )

    first_structural = action_space.structural_action_mask(first)
    before = action_space.cache_info()
    second_structural = action_space.structural_action_mask(second)
    after = action_space.cache_info()

    np.testing.assert_array_equal(first_structural, second_structural)
    assert int(after["hits"]) == int(before["hits"]) + 1

    first_operational = action_space.operational_action_mask(first)
    second_operational = action_space.operational_action_mask(second)
    assert first_operational.tolist() != second_operational.tolist()
