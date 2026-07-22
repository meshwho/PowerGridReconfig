from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    REPLAY_BUFFER_SCHEMA_VERSION,
)
from grid_topology_ai.evaluation import checkpoint as evaluation_checkpoint
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.return_contract import (
    VALUE_TARGET_MODE,
    discounted_terminal_utility,
    terminal_utility_from_outcome,
)
from grid_topology_ai.search.mcts import MCTSConfig, MCTSNode, MCTSPlanner
from grid_topology_ai.self_play import generation
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows


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
        (False, TerminationReason.HANDOFF_TO_REDISPATCH, 0.0),
        (False, TerminationReason.POWER_FLOW_FAILED, -1.0),
        (False, TerminationReason.MAX_STEPS_REACHED, -1.0),
    ],
)
def test_mcts_backup_and_value_targets_share_terminal_utility(
    solved: bool,
    reason: TerminationReason,
    expected_utility: float,
) -> None:
    gamma = 0.8
    utility, _ = terminal_utility_from_outcome(solved, reason)
    assert utility == expected_utility

    rows: list[dict[str, object]] = [
        {
            "scenario_id": 1,
            "step": 0,
            "solved": solved,
            "done": True,
            "termination_reason": reason.value,
            "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        },
        {
            "scenario_id": 1,
            "step": 1,
            "solved": solved,
            "done": True,
            "termination_reason": reason.value,
            "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        },
    ]
    add_outcome_value_targets_to_rows(rows, gamma=gamma)

    path = [_node(reward=10_000.0), _node(reward=-10_000.0)]
    MCTSPlanner(MCTSConfig(gamma=gamma))._backup(path, leaf_value=utility)

    assert rows[0]["outcome_value_target"] == pytest.approx(
        discounted_terminal_utility(
            utility,
            steps_to_terminal=2,
            gamma=gamma,
        )
    )
    assert rows[1]["outcome_value_target"] == pytest.approx(
        discounted_terminal_utility(
            utility,
            steps_to_terminal=1,
            gamma=gamma,
        )
    )
    assert path[0].total_value == pytest.approx(
        rows[0]["outcome_value_target"]
    )
    assert path[1].total_value == pytest.approx(
        rows[1]["outcome_value_target"]
    )


def test_mcts_backup_has_no_dense_reward_path() -> None:
    source = textwrap.dedent(inspect.getsource(MCTSPlanner._backup))
    tree = ast.parse(source)

    attribute_names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    referenced_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
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
    root = Path("grid_topology_ai/models")
    forbidden = (
        'row.get("outcome_value_target", row["discounted_return_from_step"])',
        'row.get("outcome_value_target", row["final_return"])',
        'target_value = row["discounted_return_from_step"]',
        'target_value = row["final_return"]',
    )

    for relative_path in (
        "self_play_dataset.py",
        "graph_self_play_dataset.py",
    ):
        text = (root / relative_path).read_text(encoding="utf-8")
        assert "outcome_value_target" in text
        for token in forbidden:
            assert token not in text


def test_generation_diagnostic_returns_use_transition_rewards_only() -> None:
    source = textwrap.dedent(
        inspect.getsource(generation.generate_self_play_examples)
    )
    tree = ast.parse(source)
    referenced_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
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
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "rewards"
    assert "final_return = returns[0] if returns else 0.0" in source


def test_evaluation_reward_uses_run_gamma_everywhere() -> None:
    worker_source = textwrap.dedent(
        inspect.getsource(evaluation_checkpoint.init_worker_context)
    )
    task_source = textwrap.dedent(
        inspect.getsource(evaluation_checkpoint._make_task_config)
    )

    assert 'discount_factor=float(task_config["gamma"])' in worker_source
    assert "discount_factor=config.gamma" in task_source
    assert '"reward_config": GridFMReward(' in task_source


def test_unified_return_contract_versions_are_pinned() -> None:
    assert VALUE_TARGET_MODE == "alphazero_discounted"
    assert OUTCOME_VALUE_TARGET_CONTRACT_VERSION == 4
    assert CHECKPOINT_CONTRACT_VERSION == 5
    assert REPLAY_BUFFER_SCHEMA_VERSION == 4
