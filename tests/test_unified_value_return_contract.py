from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import grid_topology_ai.evaluation as evaluation_checkpoint
from grid_topology_ai.reward import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
    terminal_utility_from_outcome,
)
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from grid_topology_ai.self_play import generation
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows
from tests.outcome_evidence_helpers import terminal_evidence_fields


def _node(*, reward: float = 0.0) -> MCTSNode:
    return MCTSNode(
        env=SimpleNamespace(
            done=False,
            solved=False,
            termination_reason=None,
            current_state=None,
        ),
        depth=1,
        reward_from_parent=reward,
    )


@pytest.mark.parametrize(
    ("solved", "reason", "expected_utility"),
    [
        (True, TerminationReason.SOLVED, 1.0),
        (False, TerminationReason.HANDOFF_TO_REDISPATCH, -1.0),
        (False, TerminationReason.POWER_FLOW_FAILED, -1.0),
        (False, TerminationReason.MAX_STEPS_REACHED, -1.0),
    ],
)
def test_mcts_backup_and_value_targets_share_terminal_utility(
    solved: bool,
    reason: TerminationReason,
    expected_utility: float,
) -> None:
    utility, _ = terminal_utility_from_outcome(solved, reason)
    assert utility == expected_utility

    evidence_fields = terminal_evidence_fields(reason)
    identity = {"run_id": "run-1", "iteration": 1, "episode_id": "episode-1"}
    rows: list[dict[str, object]] = [
        {
            **identity,
            "scenario_id": 1,
            "step": step,
            "solved": solved,
            "done": True,
            "termination_reason": reason.value,
            **evidence_fields,
        }
        for step in (0, 1)
    ]
    add_outcome_value_targets_to_rows(rows, gamma=TERMINAL_UTILITY_GAMMA)

    path = [_node(reward=10_000.0), _node(reward=-10_000.0)]
    MCTSPlanner(MCTSConfig())._backup(path, leaf_value=utility)

    assert rows[0]["outcome_value_target"] == pytest.approx(utility)
    assert rows[1]["outcome_value_target"] == pytest.approx(utility)
    assert path[0].total_value == pytest.approx(utility)
    assert path[1].total_value == pytest.approx(utility)
    assert all("outcome_value_target_contract_version" not in row for row in rows)


def test_mcts_backup_has_no_dense_reward_path() -> None:
    source = textwrap.dedent(inspect.getsource(MCTSPlanner._backup))
    tree = ast.parse(source)
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    step_reward_reads = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "reward"
            and isinstance(node.value, ast.Name)
            and node.value.id == "step_result"
        )
    ]
    total_value_updates = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "total_value"
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id == "node"
        )
    ]

    assert "reward_from_parent" not in attribute_names
    assert "potential_shaping_reward" not in referenced_names
    assert not step_reward_reads
    assert len(total_value_updates) == 1


def test_training_datasets_have_no_shaped_return_fallback() -> None:
    text = Path("grid_topology_ai/models/graph_self_play_dataset.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        'row.get("outcome_value_target", row["discounted_return_from_step"])',
        'row.get("outcome_value_target", row["final_return"])',
        'target_value = row["discounted_return_from_step"]',
        'target_value = row["final_return"]',
    )
    assert "outcome_value_target" in text
    for token in forbidden:
        assert token not in text


def test_generation_diagnostic_returns_use_transition_rewards_only() -> None:
    source = textwrap.dedent(inspect.getsource(generation.generate_self_play_examples))
    tree = ast.parse(source)
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "discounted_returns"
        )
    ]

    assert "terminal_reward" not in referenced_names
    assert "rewards_with_terminal" not in referenced_names
    assert len(calls) == 1
    assert calls[0].args
    assert isinstance(calls[0].args[0], ast.Attribute)
    assert calls[0].args[0].attr == "rewards"
    assert "final_return=returns[0] if returns else 0.0" in source


def test_evaluation_reward_uses_run_gamma_everywhere() -> None:
    worker_source = textwrap.dedent(inspect.getsource(evaluation_checkpoint.init_worker_context))
    task_source = textwrap.dedent(inspect.getsource(evaluation_checkpoint._make_task_config))
    compact_worker_source = "".join(worker_source.split())
    compact_task_source = "".join(task_source.split())

    assert 'discount_factor=float(task_config["gamma"])' in compact_worker_source
    assert "discount_factor=config.gamma" in compact_task_source
    assert '"reward_config":GridFMReward(' in compact_task_source


def test_light_value_semantics_are_undiscounted_and_unversioned() -> None:
    assert TERMINAL_UTILITY_GAMMA == 1.0
    assert VALUE_TARGET_MODE == "final_topology_state_utility"
