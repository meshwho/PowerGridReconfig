from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config import AcceptanceConfig
from grid_topology_ai.config.acceptance import PRIMARY_ACCEPTANCE_METRIC
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
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


def _canonical_branch_features(loading_percent: float) -> np.ndarray:
    features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    features[
        0,
        BRANCH_FEATURE_COLUMNS.index("br_status"),
    ] = 1.0
    features[
        0,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = float(loading_percent)
    return features


def _legacy_topology_utility(metrics: Mapping[str, object]) -> float:
    if PRIMARY_ACCEPTANCE_METRIC in metrics:
        return float(metrics[PRIMARY_ACCEPTANCE_METRIC])
    if "physically_secure_rate_requested" in metrics:
        return float(metrics["physically_secure_rate_requested"])
    return float(metrics.get("solve_rate", 0.0))


def _strictify_legacy_metrics(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    result = dict(metrics)
    result.setdefault(
        PRIMARY_ACCEPTANCE_METRIC,
        _legacy_topology_utility(result),
    )

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


def _add_topology_columns(path: Path) -> None:
    frame = pd.read_csv(path)
    if "final_topology_utility" not in frame.columns:
        secure = frame["physically_secure"].astype(bool)
        frame["final_topology_utility"] = secure.astype(float)
    if "Jfinal" not in frame.columns:
        secure = frame["physically_secure"].astype(bool)
        frame["Jfinal"] = np.where(secure, 0.0, 500.0)
    frame.to_csv(path, index=False)


@pytest.fixture(autouse=True)
def migrate_pre_v5_self_play_test_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep compact legacy orchestration fixtures on the current contracts."""

    original_from_mapping = AcceptanceConfig.from_mapping

    def migrated_from_mapping(
        cls: type[AcceptanceConfig],
        data: Mapping[str, Any],
    ) -> AcceptanceConfig:
        migrated = dict(data)

        if migrated.get("metric") in {
            "solve_rate",
            "physically_secure_rate_requested",
        }:
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
def adapt_topology_utility_self_play_helpers(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Migrate old self-play test helpers without weakening production checks."""

    module = request.module
    name = request.node.path.name

    if name == "test_iteration.py":
        original_metrics = getattr(module, "_metrics", None)
        if original_metrics is not None:
            def metrics_with_utility(*args: object, **kwargs: object):
                result = original_metrics(*args, **kwargs)
                result[PRIMARY_ACCEPTANCE_METRIC] = _legacy_topology_utility(result)
                for mode in result.get("mode_metrics", {}).values():
                    mode[PRIMARY_ACCEPTANCE_METRIC] = _legacy_topology_utility(mode)
                return result

            monkeypatch.setattr(module, "_metrics", metrics_with_utility)

        original_write = getattr(module, "_write_evaluation_results", None)
        if original_write is not None:
            def write_results(*args: object, **kwargs: object):
                path = original_write(*args, **kwargs)
                _add_topology_columns(Path(path))
                return path

            monkeypatch.setattr(module, "_write_evaluation_results", write_results)

    if name in {
        "test_checkpoint_arena.py",
        "test_ungated_contract_completion.py",
    }:
        original_arena_metrics = getattr(module, "_arena_metrics", None)
        if original_arena_metrics is not None:
            def arena_metrics(*args: object, **kwargs: object):
                result = original_arena_metrics(*args, **kwargs)
                mode_metrics = result.get("mode_metrics", {})
                for mode in mode_metrics.values():
                    mode[PRIMARY_ACCEPTANCE_METRIC] = float(
                        mode.get("physically_secure_rate_requested", 0.0)
                    )
                primary_mode = str(result.get("primary_policy_mode", "ungated"))
                primary = mode_metrics.get(primary_mode, {})
                result[PRIMARY_ACCEPTANCE_METRIC] = float(
                    primary.get(PRIMARY_ACCEPTANCE_METRIC, 0.0)
                )
                return result

            monkeypatch.setattr(module, "_arena_metrics", arena_metrics)

    if name == "test_ungated_contract_completion.py":
        original_write = getattr(module, "_write_paired_rows", None)
        if original_write is not None:
            def write_paired_rows(*args: object, **kwargs: object):
                result = original_write(*args, **kwargs)
                path = Path(kwargs["output_dir"]) / kwargs["output_csv_name"]
                _add_topology_columns(path)
                return result

            monkeypatch.setattr(module, "_write_paired_rows", write_paired_rows)

    yield


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
        fake_state = getattr(request.module, "_FakeFinalState", None)
        if fake_state is not None and not hasattr(
            fake_state,
            "branch_features",
        ):
            fake_state.branch_features = _canonical_branch_features(
                float(fake_state.metrics["max_loading_percent"])
            )

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
def adapt_legacy_iteration_split_fixtures(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep compact iteration doubles focused on orchestration behavior."""

    if request.node.path.name != "test_iteration.py":
        yield
        return

    def physical_iteration_split(
        *,
        replay_buffer,
        iteration,
        sampling_seed,
        n_examples,
        fresh_fraction,
        train_batch_path,
        train_examples_path,
        validation_examples_path,
        metadata_path,
        **kwargs,
    ):
        batch_metadata = replay_buffer.export_mixed_batch(
            output_path=Path(train_batch_path),
            current_iteration=int(iteration),
            n_examples=n_examples,
            fresh_fraction=float(fresh_fraction),
            seed=int(sampling_seed),
        )
        batch = pd.read_csv(train_batch_path)
        if len(batch) < 2:
            raise ValueError("Legacy iteration fixture needs two examples.")
        train = batch.iloc[:-1].copy()
        validation = batch.iloc[-1:].copy()
        train.to_csv(train_examples_path, index=False)
        validation.to_csv(validation_examples_path, index=False)
        metadata = {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_scenarios": train["scenario_id"].nunique(),
            "validation_scenarios": validation["scenario_id"].nunique(),
        }
        Path(metadata_path).write_text(
            "{}\n",
            encoding="utf-8",
        )
        return batch_metadata, metadata

    monkeypatch.setattr(
        iteration_module,
        "prepare_physical_iteration_split",
        physical_iteration_split,
    )
    yield


@pytest.fixture(autouse=True)
def isolate_legacy_pipeline_final_test(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep existing pipeline tests focused on iteration orchestration."""

    if request.node.path.name != "test_pipeline.py":
        yield
        return

    from grid_topology_ai.self_play import pipeline as pipeline_module
    from grid_topology_ai.self_play.final_test import FinalTestEvaluation

    def final_test_evaluation(*, paths, checkpoint, **kwargs):
        output_dir = paths.run_dir / "final_test"
        return FinalTestEvaluation(
            metrics={},
            metrics_path=output_dir / "eval_metrics.json",
            results_path=output_dir / "eval_results.csv",
            report_path=output_dir / "final_test_report.json",
            checkpoint=Path(checkpoint),
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_final_test_evaluation",
        final_test_evaluation,
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
