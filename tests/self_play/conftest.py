from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import wraps
from typing import Any

import pytest

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.acceptance import PRIMARY_ACCEPTANCE_METRIC
from grid_topology_ai.evaluation import checkpoint as evaluation
from grid_topology_ai.self_play import iteration as iteration_module
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.acceptance import (
    accept_candidate as strict_accept_candidate,
)


_COMPONENT_FIELDS = (
    "power_flow_converged",
    "all_values_finite",
    "topology_connected",
    "thermal_solved",
    "thermal_feasible",
    "hard_overload_free",
    "voltage_feasible",
    "generator_p_feasible",
    "generator_q_feasible",
    "angle_difference_feasible",
    "physically_secure",
)

_TERMINAL_ONLY_EPISODE_TESTS = {
    "test_run_episode_adds_physical_row_fields_directly",
    "test_run_episode_rejects_solved_contract_mismatch",
}

_MOCKED_GENERATION_STAGE_TESTS = {
    (
        "test_stages.py",
        "test_run_generate_returns_complete_generator_artifact",
    ),
    (
        "test_typed_stage_config.py",
        "test_run_generate_uses_generation_request",
    ),
    (
        "test_typed_stage_config.py",
        "test_stage_output_logs_exception_and_restores_streams",
    ),
}


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _strictify_legacy_metrics(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    result = dict(metrics)

    if "requested_scenarios" in result:
        return result

    solve_rate = float(result["solve_rate"])
    requested_scenarios = 1000
    failed_scenarios = int(result.get("failed_scenarios", 0))
    evaluated_scenarios = requested_scenarios - failed_scenarios
    solve_count = int(round(solve_rate * evaluated_scenarios))

    result.update(
        {
            "requested_scenarios": requested_scenarios,
            "evaluated_scenarios": evaluated_scenarios,
            "failed_scenarios": failed_scenarios,
            "solve_count": solve_count,
            "solve_rate": _rate(
                solve_count,
                evaluated_scenarios,
            ),
            "solve_rate_requested": _rate(
                solve_count,
                requested_scenarios,
            ),
            "evaluation_coverage_rate": _rate(
                evaluated_scenarios,
                requested_scenarios,
            ),
            "failed_scenario_rate_requested": _rate(
                failed_scenarios,
                requested_scenarios,
            ),
            "power_flow_failure_count": 0,
            "power_flow_failure_rate": 0.0,
            "power_flow_failure_rate_requested": 0.0,
        }
    )

    for field in _COMPONENT_FIELDS:
        count = (
            solve_count
            if field == "physically_secure"
            else evaluated_scenarios
        )
        result[f"{field}_count"] = count
        result[f"{field}_rate"] = _rate(
            count,
            evaluated_scenarios,
        )
        result[f"{field}_rate_requested"] = _rate(
            count,
            requested_scenarios,
        )

    return result


@pytest.fixture(autouse=True)
def migrate_pre_v5_self_play_test_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep orchestration tests focused on orchestration while their compact
    pre-v5 fixture dictionaries are migrated to the strict production schema.

    Dedicated acceptance/config tests call the production APIs directly and
    therefore exercise fail-closed behavior without this adapter.
    """

    original_from_mapping = AcceptanceConfig.from_mapping

    def migrated_from_mapping(
        cls: type[AcceptanceConfig],
        data: Mapping[str, Any],
    ) -> AcceptanceConfig:
        migrated = dict(data)

        if migrated.get("metric") == "solve_rate":
            migrated["metric"] = PRIMARY_ACCEPTANCE_METRIC

        migrated.pop(
            "max_simple_solve_rate_drop",
            None,
        )

        return original_from_mapping(migrated)

    monkeypatch.setattr(
        AcceptanceConfig,
        "from_mapping",
        classmethod(migrated_from_mapping),
    )

    def migrated_accept_candidate(
        *,
        new_metrics: Mapping[str, object],
        best_metrics: Mapping[str, object],
        config: AcceptanceConfig,
    ) -> bool:
        candidate = _strictify_legacy_metrics(new_metrics)
        best = _strictify_legacy_metrics(best_metrics)

        if isinstance(new_metrics, dict):
            new_metrics.update(candidate)

        return strict_accept_candidate(
            new_metrics=candidate,
            best_metrics=best,
            config=config,
        )

    monkeypatch.setattr(
        iteration_module,
        "accept_candidate",
        migrated_accept_candidate,
    )


@pytest.fixture(autouse=True)
def supply_terminal_episode_seed(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep terminal-only row tests focused on physical result contracts."""
    if (
        request.node.path.name == "test_evaluation_api.py"
        and request.node.name in _TERMINAL_ONLY_EPISODE_TESTS
    ):
        original = evaluation.run_episode

        @wraps(original)
        def run_episode_with_seed(*args: object, **kwargs: object):
            kwargs.setdefault("random_seed", 42)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            evaluation,
            "run_episode",
            run_episode_with_seed,
        )

    yield


@pytest.fixture(autouse=True)
def adapt_legacy_iteration_exploration_fixtures(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep compact iteration fakes focused on orchestration behavior."""

    if request.node.path.name != "test_iteration.py":
        yield
        return

    original = iteration_module._self_play_exploration_metrics
    required_columns = {
        *iteration_module._SELF_PLAY_EXPLORATION_NUMERIC_COLUMNS,
        "selection_mode",
    }

    def exploration_metrics(examples):
        if required_columns.issubset(examples.columns):
            return original(examples)

        return {
            "steps": int(len(examples)),
            "sampled_steps": 0,
            "sample_fraction": 0.0,
            "mean_selection_temperature": 0.0,
            "mean_policy_target_entropy": 0.0,
            "mean_policy_target_normalized_entropy": 0.0,
            "mean_mcts_legal_action_count": 0.0,
            "mean_mcts_considered_action_count": 0.0,
            "mean_mcts_visited_action_count": 0.0,
            "mean_mcts_action_coverage": 0.0,
            "min_mcts_action_coverage": 0.0,
            "mean_mcts_visited_action_coverage": 0.0,
            "min_mcts_visited_action_coverage": 0.0,
        }

    monkeypatch.setattr(
        iteration_module,
        "_self_play_exploration_metrics",
        exploration_metrics,
    )

    yield


@pytest.fixture(autouse=True)
def isolate_mocked_generation_stage_dependencies(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep mocked generation tests independent from raw lineage artifacts."""

    key = (request.node.path.name, request.node.name)
    if key not in _MOCKED_GENERATION_STAGE_TESTS:
        return

    monkeypatch.setattr(
        stages,
        "annotate_transitions_csv",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        stages,
        "annotate_examples_csv",
        lambda **kwargs: None,
    )
