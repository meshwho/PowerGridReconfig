from __future__ import annotations

from types import SimpleNamespace

import pytest

from grid_topology_ai.value_targets import TERMINAL_UTILITY_GAMMA
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows
from tests.outcome_evidence_helpers import (
    terminal_evidence,
    terminal_evidence_fields,
)


def _node(
    *,
    done: bool = False,
    solved: bool = False,
    reason: TerminationReason | None = None,
    reward: float = 0.0,
) -> MCTSNode:
    evidence = (
        terminal_evidence(reason)
        if done and reason is not None
        else None
    )
    env = SimpleNamespace(
        done=done,
        solved=solved,
        termination_reason=reason,
        terminal_outcome_evidence=evidence,
        current_state=None,
    )
    return MCTSNode(  # type: ignore[arg-type]
        env=env,
        depth=1,
        reward_from_parent=reward,
    )


def test_mcts_config_requires_undiscounted_terminal_utility() -> None:
    assert MCTSConfig().gamma == TERMINAL_UTILITY_GAMMA
    with pytest.raises(ValueError, match="gamma must be exactly 1.0"):
        MCTSConfig(gamma=0.95)


def test_mcts_backup_ignores_shaped_environment_rewards() -> None:
    planner = MCTSPlanner(MCTSConfig(gamma=TERMINAL_UTILITY_GAMMA))
    first = _node(reward=10_000.0)
    second = _node(reward=-10_000.0)

    planner._backup([first, second], leaf_value=1.0)

    assert second.visit_count == 1
    assert second.total_value == 1.0
    assert first.visit_count == 1
    assert first.total_value == 1.0


def test_mcts_terminal_leaf_uses_same_outcome_utility_as_value_targets() -> None:
    planner = MCTSPlanner(MCTSConfig())

    assert planner._leaf_value(
        _node(
            done=True,
            solved=True,
            reason=TerminationReason.SOLVED,
        )
    ) == 1.0
    assert planner._leaf_value(
        _node(
            done=True,
            solved=False,
            reason=TerminationReason.HANDOFF_TO_REDISPATCH,
        )
    ) == -1.0
    assert planner._leaf_value(
        _node(
            done=True,
            solved=False,
            reason=TerminationReason.POWER_FLOW_FAILED,
        )
    ) == -1.0


def test_mcts_terminal_leaf_requires_terminal_evidence() -> None:
    planner = MCTSPlanner(MCTSConfig())
    node = _node(
        done=True,
        solved=False,
        reason=TerminationReason.MAX_STEPS_REACHED,
    )
    node.env.terminal_outcome_evidence = None

    with pytest.raises(RuntimeError, match="terminal outcome evidence"):
        planner._leaf_value(node)


def test_value_targets_equal_mcts_undiscounted_terminal_backup() -> None:
    evidence_fields = terminal_evidence_fields(TerminationReason.SOLVED)
    identity = {
        "run_id": "run-1",
        "iteration": 1,
        "episode_id": "episode-1",
    }
    rows: list[dict[str, object]] = [
        {
            **identity,
            "scenario_id": 1,
            "step": step,
            "solved": True,
            "done": True,
            "termination_reason": TerminationReason.SOLVED.value,
            **evidence_fields,
        }
        for step in (0, 1)
    ]
    add_outcome_value_targets_to_rows(
        rows,
        gamma=TERMINAL_UTILITY_GAMMA,
    )

    planner = MCTSPlanner(MCTSConfig())
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

    planner = MCTSPlanner(  # type: ignore[arg-type]
        MCTSConfig(),
        evaluator=_Evaluator(),
    )
    node = _node()
    node.env.current_state = object()
    node.env.operational_action_mask = lambda: [True]

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        planner._leaf_value(node)
