from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.checkpoint_state import (
    initialize_best_state,
    promote_candidate,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths


def _config() -> SelfPlayConfig:
    return SelfPlayConfig.from_mapping(
        {
            "run_name": "test_self_play",
            "seed": 7,
            "n_iterations": 2,
            "n_scenarios_per_iteration": 3,
            "epochs_per_iteration": 1,
            "pool": {
                "transitions_csv": "inputs/pool.csv",
                "raw_dir": "inputs/pool_raw",
                "metadata_path": "runs/test_self_play/inputs/pool_metadata.json",
            },
            "eval_csv": "inputs/eval.csv",
            "eval_raw_dir": "inputs/eval_raw",
            "bootstrap_checkpoint": "bootstrap/bootstrap.pt",
            "bootstrap_eval_metrics": "bootstrap/metrics.json",
            "checkpoint_dir": "runs/test_self_play",
            "best_checkpoint_path": "runs/test_self_play/checkpoints/best.pt",
            "best_metrics_path": "runs/test_self_play/checkpoints/best_metrics.json",
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
    )


def _paths(tmp_path: Path) -> SelfPlayPaths:
    return SelfPlayPaths.from_config(
        config=_config(),
        project_root=tmp_path,
    )


def _write_bootstrap(
    paths: SelfPlayPaths,
    *,
    checkpoint: bytes = b"bootstrap",
    solve_rate: float = 0.75,
) -> None:
    paths.bootstrap_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.bootstrap_checkpoint.write_bytes(checkpoint)
    paths.bootstrap_metrics.write_text(
        json.dumps({"solve_rate": solve_rate}),
        encoding="utf-8",
    )


def test_initialize_best_state_copies_bootstrap_files(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_bootstrap(paths, checkpoint=b"checkpoint", solve_rate=0.75)

    state = initialize_best_state(paths=paths)

    assert paths.best_checkpoint.read_bytes() == b"checkpoint"
    assert paths.best_metrics.is_file()
    assert state.checkpoint == paths.best_checkpoint
    assert state.metrics["solve_rate"] == 0.75


def test_initialize_best_state_does_not_overwrite_existing_best(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_bootstrap(paths, checkpoint=b"bootstrap", solve_rate=0.1)
    paths.best_checkpoint.parent.mkdir(parents=True)
    paths.best_checkpoint.write_bytes(b"existing")
    paths.best_metrics.write_text(
        json.dumps({"solve_rate": 0.9}),
        encoding="utf-8",
    )

    state = initialize_best_state(paths=paths)

    assert paths.best_checkpoint.read_bytes() == b"existing"
    assert json.loads(paths.best_metrics.read_text(encoding="utf-8")) == {
        "solve_rate": 0.9,
    }
    assert state.metrics["solve_rate"] == 0.9


def test_initialize_best_state_rejects_non_mapping_metrics(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_bootstrap(paths)
    paths.best_metrics.parent.mkdir(parents=True)
    paths.best_metrics.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        initialize_best_state(paths=paths)


def test_promote_candidate_replaces_canonical_best(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.best_checkpoint.parent.mkdir(parents=True)
    paths.best_checkpoint.write_bytes(b"old")
    paths.best_metrics.write_text(
        json.dumps({"solve_rate": 0.1}),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")

    state = promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics={"solve_rate": 0.95},
        paths=paths,
    )

    assert paths.best_checkpoint.read_bytes() == b"candidate"
    assert json.loads(paths.best_metrics.read_text(encoding="utf-8")) == {
        "solve_rate": 0.95,
    }
    assert state.checkpoint == paths.best_checkpoint
    assert state.metrics == {"solve_rate": 0.95}


def test_promote_candidate_requires_checkpoint_file(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="Candidate checkpoint"):
        promote_candidate(
            candidate_checkpoint=tmp_path / "missing.pt",
            candidate_metrics={"solve_rate": 0.95},
            paths=paths,
        )


def test_promote_candidate_creates_parent_directories(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")

    promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics={"solve_rate": 0.95},
        paths=paths,
    )

    assert paths.best_checkpoint.is_file()
    assert paths.best_metrics.is_file()


def test_promote_candidate_does_not_mutate_input_metrics(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    metrics = {"solve_rate": 0.95}

    promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics=metrics,
        paths=paths,
    )

    assert metrics == {"solve_rate": 0.95}
