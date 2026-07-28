from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play.examples import ExampleWriter
from grid_topology_ai.self_play.generation import _policy_entropy
from grid_topology_ai.self_play.iteration import (
    _self_play_exploration_metrics,
)


def _exploration_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selection_temperature": 1.0,
                "selection_mode": "sample",
                "policy_target_entropy": math.log(2.0),
                "policy_target_normalized_entropy": 1.0,
                "mcts_legal_action_count": 10,
                "mcts_considered_action_count": 5,
                "mcts_visited_action_count": 3,
                "mcts_action_coverage": 0.5,
                "mcts_visited_action_coverage": 0.3,
            },
            {
                "selection_temperature": 0.0,
                "selection_mode": "argmax",
                "policy_target_entropy": 0.0,
                "policy_target_normalized_entropy": 0.0,
                "mcts_legal_action_count": 8,
                "mcts_considered_action_count": 4,
                "mcts_visited_action_count": 2,
                "mcts_action_coverage": 0.5,
                "mcts_visited_action_coverage": 0.25,
            },
        ]
    )


def test_policy_entropy_is_zero_for_one_hot_policy() -> None:
    entropy, normalized = _policy_entropy(
        {1: 1.0}
    )

    assert entropy == pytest.approx(0.0)
    assert normalized == pytest.approx(0.0)


def test_policy_entropy_is_one_when_uniform_and_normalized() -> None:
    entropy, normalized = _policy_entropy(
        {
            1: 0.25,
            2: 0.25,
            3: 0.25,
            4: 0.25,
        }
    )

    assert entropy == pytest.approx(math.log(4.0))
    assert normalized == pytest.approx(1.0)


def test_policy_entropy_uses_positive_probability_support() -> None:
    entropy, normalized = _policy_entropy(
        {
            1: 0.5,
            2: 0.5,
            3: 0.0,
        }
    )

    assert entropy == pytest.approx(math.log(2.0))
    assert normalized == pytest.approx(1.0)


def test_example_writer_saves_exploration_diagnostics(
    tmp_path: Path,
) -> None:
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    writer.state_store = SimpleNamespace(
        save_state=lambda **kwargs: tmp_path / "states" / "state-1.npz"
    )

    writer.add_example(
        state=object(),  # type: ignore[arg-type]
        state_id="state-1",
        action_mask=[True, True],
        scenario_id=1,
        step=0,
        selected_action_id=1,
        selected_branch_id=4,
        step_reward=1.0,
        final_return=1.0,
        discounted_return_from_step=1.0,
        solved=True,
        done=True,
        termination_reason="solved",
        visit_counts={1: 10},
        mcts_policy={1: 1.0},
        selection_temperature=0.75,
        selection_mode="sample",
        policy_target_entropy=0.4,
        policy_target_normalized_entropy=0.6,
        mcts_legal_action_count=10,
        mcts_considered_action_count=5,
        mcts_visited_action_count=3,
        mcts_action_coverage=0.5,
        mcts_visited_action_coverage=0.3,
    )

    row = pd.read_csv(writer.save()).iloc[0]
    assert row["selection_temperature"] == pytest.approx(0.75)
    assert row["selection_mode"] == "sample"
    assert row["policy_target_entropy"] == pytest.approx(0.4)
    assert row["policy_target_normalized_entropy"] == pytest.approx(0.6)
    assert row["mcts_legal_action_count"] == 10
    assert row["mcts_considered_action_count"] == 5
    assert row["mcts_visited_action_count"] == 3
    assert row["mcts_action_coverage"] == pytest.approx(0.5)
    assert row["mcts_visited_action_coverage"] == pytest.approx(0.3)


def test_exploration_metrics_aggregate_generated_steps() -> None:
    metrics = _self_play_exploration_metrics(
        _exploration_rows()
    )

    assert metrics["steps"] == 2
    assert metrics["sampled_steps"] == 1
    assert metrics["sample_fraction"] == pytest.approx(0.5)
    assert metrics["mean_selection_temperature"] == pytest.approx(0.5)
    assert metrics["mean_policy_target_entropy"] == pytest.approx(
        math.log(2.0) / 2.0
    )
    assert metrics[
        "mean_policy_target_normalized_entropy"
    ] == pytest.approx(0.5)
    assert metrics["mean_mcts_legal_action_count"] == pytest.approx(9.0)
    assert metrics[
        "mean_mcts_considered_action_count"
    ] == pytest.approx(4.5)
    assert metrics["mean_mcts_visited_action_count"] == pytest.approx(2.5)
    assert metrics["mean_mcts_action_coverage"] == pytest.approx(0.5)
    assert metrics["min_mcts_action_coverage"] == pytest.approx(0.5)
    assert metrics[
        "mean_mcts_visited_action_coverage"
    ] == pytest.approx(0.275)
    assert metrics[
        "min_mcts_visited_action_coverage"
    ] == pytest.approx(0.25)


def test_exploration_metrics_reject_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing exploration diagnostic columns",
    ):
        _self_play_exploration_metrics(
            pd.DataFrame(
                [{"selection_mode": "sample"}]
            )
        )


def test_exploration_metrics_reject_empty_dataframe() -> None:
    with pytest.raises(
        ValueError,
        match="empty dataframe",
    ):
        _self_play_exploration_metrics(
            _exploration_rows().iloc[0:0]
        )


def test_exploration_metrics_reject_non_finite_values() -> None:
    examples = _exploration_rows()
    examples.loc[0, "mcts_action_coverage"] = np.nan

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        _self_play_exploration_metrics(examples)


def test_exploration_metrics_reject_unknown_selection_mode() -> None:
    examples = _exploration_rows()
    examples.loc[0, "selection_mode"] = "greedy"

    with pytest.raises(
        ValueError,
        match="Unsupported self-play selection modes: greedy",
    ):
        _self_play_exploration_metrics(examples)
