from types import SimpleNamespace

import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner


def test_loading_priority_applies_only_to_branch_opening() -> None:
    planner = MCTSPlanner(MCTSConfig())
    loading_idx = BRANCH_FEATURE_COLUMNS.index(
        "loading_percent"
    )
    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=float,
    )
    branch_features[0, loading_idx] = 127.5
    state = SimpleNamespace(
        branch_features=branch_features
    )

    opening = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )
    closing = GridFMAction(
        action_id=2,
        action_type="switch_on_branch",
        branch_id=11,
        branch_pos=1,
    )

    assert planner._loading_priority(
        state,
        opening,
    ) == 127.5
    assert planner._loading_priority(
        state,
        closing,
    ) is None


def test_progressive_widening_adds_branch_closing_action() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=1,
            widening_coefficient=1.0,
            widening_exponent=1.0,
        )
    )
    opening = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )
    closing = GridFMAction(
        action_id=2,
        action_type="switch_on_branch",
        branch_id=11,
        branch_pos=1,
    )
    node = MCTSNode(
        env=SimpleNamespace(
            done=False,
            solved=False,
        ),
        depth=0,
        visit_count=1,
        is_expanded=True,
        ranked_actions=[opening, closing],
        action_scores={1: 2.0, 2: 1.0},
    )
    planner._set_active_actions(
        node,
        [opening],
    )

    assert planner._widen_node(node) is True
    assert list(node.actions_by_id) == [1, 2]
