from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.contracts import EVALUATION_METRICS_CONTRACT_VERSION
from grid_topology_ai.evaluation.episode_result import (
    EvaluationEpisodeTrace,
    build_evaluation_episode_row,
)
from grid_topology_ai.evaluation.metrics import build_evaluation_metrics
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


class _SearchEnv:
    def __init__(self) -> None:
        self.initial_scenario_id = 17
        self.current_state = SimpleNamespace(scenario_id=17)

    def clone(self) -> "_SearchEnv":
        return _SearchEnv()


def _switch_action(action_id: int) -> GridFMAction:
    return GridFMAction(
        action_id=action_id,
        action_type="switch_off_branch",
        branch_id=100 + action_id,
        branch_pos=action_id - 1,
    )


def test_search_result_reports_root_action_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = MCTSPlanner(
        MCTSConfig(
            num_simulations=1,
            use_root_dirichlet_noise=False,
        )
    )
    actions = [_switch_action(action_id) for action_id in range(1, 6)]

    def fake_expand(root) -> None:
        root.ranked_actions = actions
        root.actions_by_id = {
            action.action_id: action
            for action in actions[:3]
        }
        root.action_priors = {
            1: 0.5,
            2: 0.3,
            3: 0.2,
        }
        root.children = {
            1: SimpleNamespace(
                visit_count=4,
                branch_id_from_parent=101,
            ),
            2: SimpleNamespace(
                visit_count=1,
                branch_id_from_parent=102,
            ),
            3: SimpleNamespace(
                visit_count=0,
                branch_id_from_parent=103,
            ),
        }
        root.is_expanded = True

    monkeypatch.setattr(planner, "_expand_node", fake_expand)
    monkeypatch.setattr(
        planner,
        "_run_one_simulation",
        lambda root: None,
    )
    monkeypatch.setattr(
        planner,
        "_principal_variation",
        lambda root: ([], [], [], 0.0, {}),
    )

    result = planner.search_from_env(_SearchEnv())  # type: ignore[arg-type]

    assert result.root_legal_action_count == 5
    assert result.root_considered_action_count == 3
    assert result.root_visited_action_count == 2
    assert result.root_action_coverage == pytest.approx(0.6)
    assert result.root_visited_action_coverage == pytest.approx(0.4)
    assert 0.0 <= result.root_visited_action_coverage <= 1.0
    assert (
        result.root_visited_action_coverage
        <= result.root_action_coverage
    )


def test_action_coverage_rate_handles_empty_legal_set() -> None:
    assert MCTSPlanner._action_coverage_rate(0, 0) == 0.0
    assert MCTSPlanner._action_coverage_rate(3, 10) == pytest.approx(0.3)


def _episode_env() -> SimpleNamespace:
    return SimpleNamespace(
        current_state=None,
        done=False,
        solved=False,
        termination_reason=None,
    )


def _trace(
    *,
    legal: list[int],
    considered: list[int],
    visited: list[int],
    coverage: list[float],
    visited_coverage: list[float],
) -> EvaluationEpisodeTrace:
    return EvaluationEpisodeTrace(
        root_legal_action_counts=legal,
        root_considered_action_counts=considered,
        root_visited_action_counts=visited,
        root_action_coverages=coverage,
        root_visited_action_coverages=visited_coverage,
    )


def test_evaluation_row_serializes_per_search_coverage() -> None:
    trace = _trace(
        legal=[10, 8],
        considered=[4, 5],
        visited=[2, 3],
        coverage=[0.4, 0.625],
        visited_coverage=[0.2, 0.375],
    )

    row = build_evaluation_episode_row(
        scenario_id=7,
        policy_mode="ungated",
        env=_episode_env(),
        trace=trace,
        physics_config=None,
    )

    assert row["mcts_searches"] == 2
    assert row["mcts_root_legal_action_counts_json"] == "[10, 8]"
    assert row["mcts_root_considered_action_counts_json"] == "[4, 5]"
    assert row["mcts_root_visited_action_counts_json"] == "[2, 3]"
    assert row["mcts_mean_action_coverage"] == pytest.approx(0.5125)
    assert row["mcts_min_action_coverage"] == pytest.approx(0.4)
    assert row["mcts_mean_visited_action_coverage"] == pytest.approx(0.2875)
    assert row["mcts_min_visited_action_coverage"] == pytest.approx(0.2)


def test_evaluation_row_does_not_invent_missing_search_coverage() -> None:
    row = build_evaluation_episode_row(
        scenario_id=8,
        policy_mode="ungated",
        env=_episode_env(),
        trace=EvaluationEpisodeTrace(),
        physics_config=None,
    )

    assert row["mcts_searches"] == 0
    assert row["mcts_root_legal_action_counts_json"] == "[]"
    assert row["mcts_mean_action_coverage"] is None
    assert row["mcts_min_action_coverage"] is None
    assert row["mcts_mean_visited_action_coverage"] is None
    assert row["mcts_min_visited_action_coverage"] is None


def test_evaluation_metrics_aggregate_action_coverage() -> None:
    first = build_evaluation_episode_row(
        scenario_id=1,
        policy_mode="ungated",
        env=_episode_env(),
        trace=_trace(
            legal=[10, 8],
            considered=[4, 5],
            visited=[2, 3],
            coverage=[0.4, 0.6],
            visited_coverage=[0.2, 0.3],
        ),
        physics_config=None,
    )
    second = build_evaluation_episode_row(
        scenario_id=2,
        policy_mode="ungated",
        env=_episode_env(),
        trace=_trace(
            legal=[5],
            considered=[4],
            visited=[2],
            coverage=[0.8],
            visited_coverage=[0.5],
        ),
        physics_config=None,
    )

    metrics = build_evaluation_metrics(
        df=pd.DataFrame([first, second]),
        failed_results=[],
        requested_scenarios=2,
        task_config={},
    )

    assert metrics["evaluation_metrics_contract_version"] == 6
    assert metrics["avg_mcts_action_coverage"] == pytest.approx(0.65)
    assert metrics["min_mcts_action_coverage"] == pytest.approx(0.4)
    assert metrics["avg_mcts_visited_action_coverage"] == pytest.approx(0.375)
    assert metrics["min_mcts_visited_action_coverage"] == pytest.approx(0.2)


def test_evaluation_metrics_contract_version_is_six() -> None:
    assert EVALUATION_METRICS_CONTRACT_VERSION == 6
