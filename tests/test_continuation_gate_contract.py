from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.search.continuation_gate import analyze_root_branches


def _state(*, loading: float, num_hard: int) -> GridFMState:
    branch_features = np.zeros((1, len(BRANCH_FEATURE_COLUMNS)), dtype=np.float32)
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loading
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 1), dtype=np.float32),
        branch_features=branch_features,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([1.0], dtype=np.float32),
        metrics={
            "num_overloaded_branches": int(loading > 100.0),
            "num_hard_overloaded_branches": num_hard,
            "max_loading_percent": loading,
            "total_voltage_violation": 0.0,
        },
        outaged_branch_ids=[],
    )


def _node(
    state: GridFMState,
    *,
    depth: int,
    branch_id: int | None,
    visits: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(current_state=state),
        depth=depth,
        branch_id_from_parent=branch_id,
        reward_from_parent=0.0,
        visit_count=visits,
        children={},
    )


def _result(root: SimpleNamespace) -> SimpleNamespace:
    best_action_id = max(root.children, key=lambda action: root.children[action].visit_count)
    return SimpleNamespace(
        root=root,
        best_action_id=best_action_id,
        best_branch_id=root.children[best_action_id].branch_id_from_parent,
    )


def test_gate_does_not_force_a_hard_overload_fallback() -> None:
    root = _node(_state(loading=130.0, num_hard=1), depth=0, branch_id=None, visits=0)
    root.children = {
        1: _node(_state(loading=130.0, num_hard=1), depth=1, branch_id=11, visits=20),
    }

    decision = analyze_root_branches(
        _result(root),
        min_hard_improvement=0.0,
        min_visits=1,
        min_visit_fraction=0.0,
    )

    assert decision.allowed_action_ids == ()
    assert decision.recommended_action_id is None
    assert decision.recommended_branch_id is None
    assert decision.recommendation_reason == "no_allowed_continuation"


def test_gate_reports_allowed_support_without_executing_it() -> None:
    root = _node(_state(loading=130.0, num_hard=1), depth=0, branch_id=None, visits=0)
    root.children = {
        1: _node(_state(loading=99.0, num_hard=0), depth=1, branch_id=11, visits=20),
        2: _node(_state(loading=130.0, num_hard=1), depth=1, branch_id=22, visits=30),
    }

    decision = analyze_root_branches(
        _result(root),
        min_hard_improvement=0.0,
        min_visits=1,
        min_visit_fraction=0.0,
    )

    assert decision.allowed_action_ids == (1,)
    assert decision.recommended_action_id == 1
    assert decision.recommended_branch_id == 11
    assert decision.recommendation_reason == "best_allowed_by_visits"
    assert decision.selected_action_id == 1
