from __future__ import annotations

from types import SimpleNamespace

import pytest

from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    REPLAY_BUFFER_SCHEMA_VERSION,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


def _node(
    *,
    done: bool = False,
    solved: bool = False,
    reason: TerminationReason | None = None,
    reward: float = 0.0,
) -> MCTSNode:
    env = SimpleNamespace(
        done=done,
        solved=solved,
        termination_reason=reason,
        current_state=None,
    )
    return MCTSNode(  # type: ignore[arg-type]
        env=env,
        depth=1,
        reward_from_parent=reward,
    )


def test_mcts_backup_ignores_shaped_environment_rewards() -> None:
    planner = MCTSPlanner(MCTSConfig(gamma=0.5))
    first = _node(reward=10_000.0)
    second = _node(reward=-10_000.0)

    planner._backup([first, second], leaf_value=1.0)

    assert second.visit_count == 1
    assert second.total_value == 0.5
    assert first.visit_count == 1
    assert first.total_value == 0.25


def test_mcts_terminal_leaf_uses_same_outcome_utility_as_value_targets() -> None:
    planner = MCTSPlanner(MCTSConfig())

    assert planner._leaf_value(
        _node(done=True, solved=True, reason=TerminationReason.SOLVED)
    ) == 1.0
    assert planner._leaf_value(
        _node(
            done=True,
            solved=False,
            reason=TerminationReason.HANDOFF_TO_REDISPATCH,
        )
    ) == 0.0
    assert planner._leaf_value(
        _node(
            done=True,
            solved=False,
            reason=TerminationReason.POWER_FLOW_FAILED,
        )
    ) == -1.0


def test_value_targets_equal_mcts_discounted_terminal_backup() -> None:
    gamma = 0.5
    rows: list[dict[str, object]] = [
        {
            "scenario_id": 1,
            "step": 0,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
            "physical_objective_schema_version": (
                PHYSICAL_OBJECTIVE_SCHEMA_VERSION
            ),
        },
        {
            "scenario_id": 1,
            "step": 1,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
            "physical_objective_schema_version": (
                PHYSICAL_OBJECTIVE_SCHEMA_VERSION
            ),
        },
    ]
    add_outcome_value_targets_to_rows(rows, gamma=gamma)

    planner = MCTSPlanner(MCTSConfig(gamma=gamma))
    first = _node(reward=1234.0)
    second = _node(reward=-4321.0)
    planner._backup([first, second], leaf_value=1.0)

    assert rows[0]["outcome_value_target"] == first.total_value
    assert rows[1]["outcome_value_target"] == second.total_value
    assert rows[0]["outcome_steps_to_terminal"] == 2
    assert rows[1]["outcome_steps_to_terminal"] == 1


def test_neural_value_outside_terminal_utility_range_is_rejected() -> None:
    class _Evaluator:
        def evaluate(self, *, state, action_mask):
            return [1.0], 50.0

    planner = MCTSPlanner(MCTSConfig(), evaluator=_Evaluator())  # type: ignore[arg-type]
    node = _node()
    node.env.current_state = object()
    node.env.valid_action_mask = lambda: [True]

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        planner._leaf_value(node)


def test_return_contract_invalidates_legacy_search_artifacts() -> None:
    assert OUTCOME_VALUE_TARGET_CONTRACT_VERSION == 4
    assert CHECKPOINT_CONTRACT_VERSION == 5
    assert REPLAY_BUFFER_SCHEMA_VERSION == 4
