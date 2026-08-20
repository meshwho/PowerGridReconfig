from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.topology_actions import (
    action_layout_fingerprint,
    build_branch_action_slots,
)


def _state() -> GridFMState:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pd")] = [40.0, 25.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pg")] = [65.0, 0.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Vm")] = [1.01, 0.99]

    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("pf")] = [35.0, 18.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0

    return GridFMState(
        scenario_id=7,
        load_scenario_idx=2.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array(
            [[0, 1], [1, 0]],
            dtype=np.int64,
        ),
        branch_ids=np.array([10, 20], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=np.array([100, 200], dtype=np.int64),
    )


def _evaluator(
    state: GridFMState,
) -> tuple[NeuralPolicyValueEvaluator, list[GridFMState]]:
    evaluator = object.__new__(NeuralPolicyValueEvaluator)
    evaluator.enable_cache = True
    evaluator._cache = {}
    evaluator.cache_hits = 0
    evaluator.cache_misses = 0
    evaluator.model_type = "graph_policy_value_net_v2"
    evaluator.physics_config = PhysicsConfig()
    evaluator.action_layout_fingerprint = action_layout_fingerprint(
        build_branch_action_slots(state.branch_ids)
    )

    calls: list[GridFMState] = []

    def evaluate_graph(
        state: GridFMState,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        calls.append(state)
        return np.array([0.6, 0.3, 0.1], dtype=np.float32), 0.25

    evaluator._evaluate_graph = evaluate_graph
    return evaluator, calls


def _changed_feature(
    state: GridFMState,
    feature_group: str,
    feature_name: str,
) -> GridFMState:
    if feature_group == "bus":
        bus_features = state.bus_features.copy()
        bus_features[0, BUS_FEATURE_COLUMNS.index(feature_name)] += 1.0
        return replace(state, bus_features=bus_features)

    branch_features = state.branch_features.copy()
    branch_features[0, BRANCH_FEATURE_COLUMNS.index(feature_name)] += 1.0
    return replace(state, branch_features=branch_features)


def test_repeated_evaluation_uses_cache():
    state = _state()
    evaluator, calls = _evaluator(state)
    action_mask = np.array([True, True, False])

    first_policy, first_value = evaluator.evaluate(state, action_mask)
    second_policy, second_value = evaluator.evaluate(state, action_mask.copy())

    np.testing.assert_array_equal(second_policy, first_policy)
    assert second_value == first_value
    assert len(calls) == 1
    assert evaluator.cache_info() == {
        "enabled": True,
        "model_type": "graph_policy_value_net_v2",
        "size": 1,
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
    }


@pytest.mark.parametrize(
    ("feature_group", "feature_name"),
    [
        ("bus", "Pd"),
        ("bus", "Pg"),
        ("bus", "Vm"),
        ("branch", "pf"),
    ],
)
def test_physical_feature_changes_miss_cache(
    feature_group: str,
    feature_name: str,
):
    state = _state()
    changed_state = _changed_feature(
        state,
        feature_group,
        feature_name,
    )
    evaluator, calls = _evaluator(state)
    action_mask = np.array([True, True, False])

    evaluator.evaluate(state, action_mask)
    evaluator.evaluate(changed_state, action_mask)

    assert len(calls) == 2
    assert evaluator.cache_hits == 0
    assert evaluator.cache_misses == 2
    assert len(evaluator._cache) == 2


def test_action_mask_change_misses_cache():
    state = _state()
    evaluator, calls = _evaluator(state)

    evaluator.evaluate(
        state,
        np.array([True, True, False]),
    )
    evaluator.evaluate(
        state,
        np.array([True, False, True]),
    )

    assert len(calls) == 2
    assert evaluator.cache_hits == 0
    assert evaluator.cache_misses == 2


def test_physics_config_changes_cache_key():
    state = _state()
    evaluator, _ = _evaluator(state)
    action_mask = np.array([True, True, False])

    original_key = evaluator._make_cache_key(state, action_mask)
    evaluator.physics_config = replace(
        evaluator.physics_config,
        max_iterations=31,
    )

    assert original_key != evaluator._make_cache_key(state, action_mask)


def test_cached_policy_is_returned_as_a_copy():
    state = _state()
    evaluator, calls = _evaluator(state)
    action_mask = np.array([True, True, False])

    first_policy, _ = evaluator.evaluate(state, action_mask)
    expected_policy = first_policy.copy()
    first_policy[:] = 0.0

    second_policy, _ = evaluator.evaluate(state, action_mask)
    np.testing.assert_array_equal(second_policy, expected_policy)

    second_policy[:] = 1.0
    third_policy, _ = evaluator.evaluate(state, action_mask)
    np.testing.assert_array_equal(third_policy, expected_policy)
    assert len(calls) == 1
    assert evaluator.cache_hits == 2
