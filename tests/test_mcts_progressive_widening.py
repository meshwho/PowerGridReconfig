from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config.evaluation import EvaluationConfig
from grid_topology_ai.config.generation import GenerationConfig
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner


def _switch_actions(count: int) -> list[GridFMAction]:
    return [
        GridFMAction(
            action_id=index + 1,
            action_type="switch_off_branch",
            branch_id=100 + index,
            branch_pos=index,
        )
        for index in range(count)
    ]


def _node(
    planner: MCTSPlanner,
    *,
    switch_count: int,
    active_switch_count: int,
    visit_count: int = 0,
    depth: int = 0,
    include_stop: bool = False,
) -> MCTSNode:
    switches = _switch_actions(switch_count)
    stop_actions = (
        [
            GridFMAction(
                action_id=0,
                action_type="do_nothing",
            )
        ]
        if include_stop
        else []
    )
    ranked_actions = [*stop_actions, *switches]
    action_scores = {
        int(action.action_id): float(len(ranked_actions) - index)
        for index, action in enumerate(ranked_actions)
    }

    node = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=depth,
        visit_count=visit_count,
        is_expanded=True,
        ranked_actions=ranked_actions,
        action_scores=action_scores,
    )
    planner._set_active_actions(
        node,
        [
            *stop_actions,
            *switches[:active_switch_count],
        ],
    )
    return node


def _active_switch_ids(node: MCTSNode) -> list[int]:
    return [
        action_id
        for action_id, action in node.actions_by_id.items()
        if action.action_type == "switch_off_branch"
    ]


def test_zero_visits_keep_the_initial_width() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=3,
            widening_coefficient=2.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=10,
        active_switch_count=3,
    )

    assert planner._target_switch_width(node) == 3
    assert planner._widen_node(node) is False
    assert _active_switch_ids(node) == [1, 2, 3]


def test_width_grows_with_node_visits() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=3,
            widening_coefficient=1.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=10,
        active_switch_count=3,
        visit_count=25,
    )

    assert planner._target_switch_width(node) == 8
    assert planner._widen_node(node) is True
    assert _active_switch_ids(node) == list(range(1, 9))
    assert sum(node.action_priors.values()) == pytest.approx(1.0)


def test_existing_wider_shortlist_is_not_shrunk() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=3,
            widening_coefficient=2.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=10,
        active_switch_count=5,
    )

    assert planner._target_switch_width(node) == 5
    assert planner._widen_node(node) is False
    assert _active_switch_ids(node) == [1, 2, 3, 4, 5]


def test_width_is_capped_by_the_legal_action_count() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=10.0,
            widening_exponent=1.0,
        )
    )
    node = _node(
        planner,
        switch_count=7,
        active_switch_count=2,
        visit_count=100,
    )

    assert planner._target_switch_width(node) == 7
    assert planner._widen_node(node) is True
    assert _active_switch_ids(node) == list(range(1, 8))


def test_zero_coefficient_disables_growth() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=3,
            widening_coefficient=0.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=10,
        active_switch_count=3,
        visit_count=10_000,
    )

    assert planner._target_switch_width(node) == 3
    assert planner._widen_node(node) is False


def test_stop_action_does_not_consume_switch_width() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=1.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=6,
        active_switch_count=2,
        visit_count=4,
        include_stop=True,
    )

    assert planner._target_switch_width(node) == 4
    assert planner._widen_node(node) is True
    assert list(node.actions_by_id) == [0, 1, 2, 3, 4]


def test_widening_preserves_existing_children() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=1.0,
            widening_exponent=0.5,
        )
    )
    node = _node(
        planner,
        switch_count=8,
        active_switch_count=2,
        visit_count=9,
    )
    child = MCTSNode(
        env=SimpleNamespace(done=False, solved=False),  # type: ignore[arg-type]
        depth=1,
        visit_count=3,
        total_value=1.5,
    )
    node.children[1] = child

    assert planner._widen_node(node) is True
    assert node.children[1] is child
    assert child.visit_count == 3
    assert child.total_value == 1.5


def test_non_root_nodes_use_the_same_widening_rule() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=2,
            widening_coefficient=1.0,
            widening_exponent=0.5,
        )
    )
    child = _node(
        planner,
        switch_count=8,
        active_switch_count=2,
        visit_count=16,
        depth=2,
    )

    assert planner._widen_node(child) is True
    assert _active_switch_ids(child) == [1, 2, 3, 4, 5, 6]


def test_unexpanded_node_is_not_widened() -> None:
    planner = MCTSPlanner(MCTSConfig(top_k_actions=2))
    node = _node(
        planner,
        switch_count=6,
        active_switch_count=2,
        visit_count=100,
    )
    node.is_expanded = False

    assert planner._widen_node(node) is False
    assert _active_switch_ids(node) == [1, 2]


def test_non_positive_top_k_exposes_all_ranked_switches() -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            top_k_actions=0,
            widening_coefficient=0.0,
        )
    )
    node = _node(
        planner,
        switch_count=5,
        active_switch_count=1,
    )

    assert planner._target_switch_width(node) == 5
    assert planner._widen_node(node) is True
    assert _active_switch_ids(node) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"widening_coefficient": -0.1},
        {"widening_coefficient": float("nan")},
        {"widening_coefficient": float("inf")},
        {"widening_coefficient": True},
        {"widening_exponent": 0.0},
        {"widening_exponent": 1.1},
        {"widening_exponent": float("nan")},
        {"widening_exponent": float("inf")},
        {"widening_exponent": True},
    ],
)
def test_mcts_config_rejects_invalid_widening_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="widening_"):
        MCTSConfig(**kwargs)


@pytest.mark.parametrize(
    "config_type",
    [GenerationConfig, EvaluationConfig],
)
def test_typed_configs_keep_widening_values(
    config_type: type[GenerationConfig] | type[EvaluationConfig],
) -> None:
    config = config_type(
        widening_coefficient=1.75,
        widening_exponent=0.35,
    )

    assert config.widening_coefficient == 1.75
    assert config.widening_exponent == 0.35


@pytest.mark.parametrize(
    "config_type",
    [GenerationConfig, EvaluationConfig],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"widening_coefficient": -1.0},
        {"widening_coefficient": float("nan")},
        {"widening_coefficient": True},
        {"widening_exponent": 0.0},
        {"widening_exponent": 1.01},
        {"widening_exponent": float("inf")},
        {"widening_exponent": True},
    ],
)
def test_typed_configs_reject_invalid_widening_values(
    config_type: type[GenerationConfig] | type[EvaluationConfig],
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="widening_"):
        config_type(**kwargs)
