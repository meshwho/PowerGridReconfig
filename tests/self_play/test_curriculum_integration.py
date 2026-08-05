from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.config.pool import CurriculumSamplingConfig
from grid_topology_ai.self_play import pipeline as pipeline_module
from grid_topology_ai.self_play.artifacts import load_json, save_json, sha256_file
from grid_topology_ai.self_play.iteration import IterationResult, _self_play_seeds
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.pipeline import PipelineRequest, run_self_play_pipeline
from grid_topology_ai.self_play.pool_sampling import (
    sample_curriculum_from_pool,
    sample_from_pool,
)
from grid_topology_ai.self_play.pool_state import update_pool_metadata


def _scenario(
    *,
    difficulty: str,
    solved: int,
    solve_rate: float,
) -> dict[str, object]:
    return {
        "difficulty_class": difficulty,
        "times_attempted": 4,
        "times_solved": solved,
        "solve_rate": solve_rate,
        "last_attempted_iter": 5,
        "last_solved_iter": 5 if solved else None,
        "avg_steps_when_solved": 4.0 if solved else None,
        "last_iteration_solve_rate": solve_rate,
        "solve_rate_delta": 0.0,
        "learning_progress": 0.0,
        "uncertainty": 0.5,
        "staleness": 0.0,
        "priority": 0.05,
    }


def test_residual_sampling_preserves_feasible_caps() -> None:
    metadata = {
        "schema_version": 3,
        "transitions_csv": "transitions.csv",
        "last_updated_iteration": 5,
        "scenarios": {
            "1": _scenario(
                difficulty="medium",
                solved=4,
                solve_rate=1.0,
            ),
            "2": _scenario(
                difficulty="simple",
                solved=4,
                solve_rate=1.0,
            ),
            "3": _scenario(
                difficulty="medium",
                solved=2,
                solve_rate=0.5,
            ),
            "4": _scenario(
                difficulty="simple",
                solved=2,
                solve_rate=0.5,
            ),
        },
    }
    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.0,
        hard_min_fraction=0.0,
        simple_max_fraction=1.0 / 3.0,
        frontier_max_fraction=1.0 / 3.0,
    )

    sample = sample_curriculum_from_pool(
        metadata,
        n=3,
        seed=4,
        current_iter=6,
        config=config,
    )

    assert set(sample.scenario_ids) == {1, 2, 3}
    assert sample.report["simple"]["limit"] == 1
    assert sample.report["simple"]["selected"] == 1
    assert sample.report["frontier"]["limit"] == 1
    assert sample.report["frontier"]["selected"] == 1
    assert sample.report["cap_relaxations"] == []


def _raw_config() -> dict[str, object]:
    return {
        "run_name": "curriculum_pipeline_test",
        "seed": 7,
        "n_iterations": 1,
        "n_scenarios_per_iteration": 3,
        "epochs_per_iteration": 1,
        "pool": {
            "transitions_csv": "inputs/pool.csv",
            "raw_dir": "inputs/pool_raw",
            "metadata_path": "runs/curriculum_pipeline_test/inputs/pool_metadata.json",
            "curriculum": {
                "never_solved_min_fraction": 0.0,
                "hard_min_fraction": 0.0,
                "simple_max_fraction": 1.0,
                "frontier_max_fraction": 1.0,
                "stale_after_iterations": 7,
            },
        },
        "eval_csv": "inputs/eval.csv",
        "eval_raw_dir": "inputs/eval_raw",
        "final_test_csv": "inputs/final_test.csv",
        "final_test_raw_dir": "inputs/final_test_raw",
        "bootstrap_checkpoint": "bootstrap/bootstrap.pt",
        "bootstrap_eval_metrics": "bootstrap/metrics.json",
        "checkpoint_dir": "runs/curriculum_pipeline_test",
        "best_checkpoint_path": (
            "runs/curriculum_pipeline_test/checkpoints/best.pt"
        ),
        "best_metrics_path": (
            "runs/curriculum_pipeline_test/checkpoints/best_metrics.json"
        ),
        "replay_buffer": {
            "max_size": 20,
            "min_size_to_train": 1,
            "fresh_fraction": 0.5,
            "random_seed": 7,
        },
        "generation": {
            "simulations": 2,
            "depth": 1,
            "max_steps": 4,
            "top_k": 2,
        },
        "training": {
            "examples_per_iteration": 5,
            "batch_size": 2,
            "learning_rate": 0.001,
            "device": "cpu",
        },
        "evaluation": {
            "simulations": 2,
            "depth": 1,
            "max_steps": 2,
            "top_k": 2,
            "device": "cpu",
        },
        "acceptance": {
            "metric": "solve_rate",
            "min_improvement": 0.0,
            "max_simple_solve_rate_drop": 0.05,
        },
    }


@dataclass(frozen=True, slots=True)
class _RunState:
    completed_iterations: tuple[int, ...] = ()
    start_iteration: int = 1


@dataclass(frozen=True, slots=True)
class _BestState:
    checkpoint: Path
    metrics: dict[str, object]


class _ReplayBuffer:
    def __len__(self) -> int:
        return 0


def test_schema_v3_curriculum_runs_through_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_config = _raw_config()
    config = SelfPlayConfig.from_mapping(raw_config)
    paths = SelfPlayPaths.from_config(
        config=config,
        project_root=tmp_path,
    )
    paths.pool_transitions_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.pool_raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"scenario_id": 1, "difficulty_class": "hard"},
            {"scenario_id": 2, "difficulty_class": "simple"},
            {"scenario_id": 3, "difficulty_class": "medium"},
            {"scenario_id": 4, "difficulty_class": "hard"},
        ]
    ).to_csv(paths.pool_transitions_csv, index=False)

    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_state",
        lambda **kwargs: _RunState(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_pf_alg",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_physics_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_final_test_evaluation",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "RollingReplayBuffer",
        lambda **kwargs: _ReplayBuffer(),
    )

    def initialize_best_state(*, paths: SelfPlayPaths) -> _BestState:
        paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        paths.best_checkpoint.write_bytes(b"best")
        save_json({"solve_rate": 0.0}, paths.best_metrics)
        return _BestState(
            checkpoint=paths.best_checkpoint,
            metrics={"solve_rate": 0.0},
        )

    monkeypatch.setattr(
        pipeline_module,
        "initialize_best_state",
        initialize_best_state,
    )

    def run_iteration(request) -> IterationResult:
        seeds = _self_play_seeds(
            base_seed=int(request.config.seed),
            iteration=request.iteration,
        )
        selected = tuple(
            sample_from_pool(
                request.pool_metadata,
                n=request.config.n_scenarios_per_iteration,
                seed=seeds.scenario_sampling,
                current_iter=request.iteration,
                config=request.config.pool.curriculum,
            )
        )
        update_pool_metadata(
            request.pool_metadata,
            [],
            current_iter=request.iteration,
            selected_scenario_ids=selected,
            stale_after_iterations=(
                request.config.pool.curriculum.stale_after_iterations
            ),
        )

        iteration_dir = request.paths.iteration_dir(request.iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = iteration_dir / "metadata.json"
        save_json(
            {
                "iteration": request.iteration,
                "accepted": True,
                "hashes": {},
                "extra": {},
            },
            metadata_path,
        )
        candidate = iteration_dir / "candidate_checkpoint.pt"
        candidate.write_bytes(b"candidate")
        request.paths.best_checkpoint.write_bytes(b"candidate")
        save_json({"solve_rate": 0.25}, request.paths.best_metrics)

        request.paths.replay_dir.mkdir(parents=True, exist_ok=True)
        replay_iteration = request.paths.replay_iteration_file(
            request.iteration
        )
        replay_iteration.write_bytes(b"replay")
        save_json(
            {
                "files": [
                    {
                        "iteration": request.iteration,
                        "path": replay_iteration.name,
                    }
                ]
            },
            request.paths.replay_manifest,
        )

        return IterationResult(
            iteration=request.iteration,
            accepted=True,
            status="ACCEPTED",
            selected_scenario_ids=selected,
            raw_examples_csv=iteration_dir / "raw.csv",
            train_batch_csv=iteration_dir / "train.csv",
            train_examples_csv=iteration_dir / "train_examples.csv",
            validation_examples_csv=(
                iteration_dir / "validation_examples.csv"
            ),
            split_metadata_path=(
                iteration_dir / "train_validation_split.json"
            ),
            candidate_checkpoint=candidate,
            metadata_path=metadata_path,
            parent_metrics={"solve_rate": 0.0},
            candidate_metrics={"solve_rate": 0.25},
            best_checkpoint=request.paths.best_checkpoint,
            best_metrics={"solve_rate": 0.25},
            pool_metadata=request.pool_metadata,
            learning_curve_row={
                "iteration": request.iteration,
                "n_fresh": len(selected),
                "n_old": 0,
            },
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_self_play_iteration",
        run_iteration,
    )

    final_report = paths.final_test_report
    monkeypatch.setattr(
        pipeline_module,
        "_run_reporting_only_final_test",
        lambda **kwargs: SimpleNamespace(
            checkpoint=paths.best_checkpoint,
            metrics={"solve_rate": 0.25},
            report_path=final_report,
        ),
    )

    result = run_self_play_pipeline(
        PipelineRequest(
            config=config,
            raw_config=raw_config,
            paths=paths,
        )
    )

    report_path = paths.iteration_dir(1) / "curriculum_sampling.json"
    metadata_path = paths.iteration_dir(1) / "metadata.json"
    completion_path = paths.iteration_completion_marker(1)

    assert result.executed_iterations == (1,)
    assert report_path.is_file()
    assert completion_path.is_file()
    assert paths.pool_metadata.is_file()
    assert paths.learning_curve.is_file()

    report = load_json(report_path)
    metadata = load_json(metadata_path)
    pool_metadata = load_json(paths.pool_metadata)
    completion = load_json(completion_path)
    curve = pd.read_csv(paths.learning_curve)

    report_sha = sha256_file(report_path)
    assert metadata["hashes"]["curriculum_sampling_sha256"] == report_sha
    assert metadata["extra"]["curriculum_sampling_sha256"] == report_sha
    assert metadata["extra"]["curriculum_sampling"] == report
    assert metadata["extra"]["curriculum_sampling_path"] == str(
        report_path
    )

    assert pool_metadata["schema_version"] == 3
    assert pool_metadata["last_updated_iteration"] == 1
    assert pool_metadata["curriculum_sampling"][
        "stale_after_iterations"
    ] == 7
    assert all(
        "priority_components" in scenario
        for scenario in pool_metadata["scenarios"].values()
    )

    assert curve.loc[0, "curriculum_hard_fraction"] >= 0.0
    assert curve.loc[0, "curriculum_mean_priority"] > 0.0
    assert completion["iteration"] == 1
    assert completion["artifacts"]["metadata_sha256"] == sha256_file(
        metadata_path
    )
