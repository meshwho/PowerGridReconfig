from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    EVALUATION_METRICS_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    physical_objective_contract,
)
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
    checkpoint: str = "bootstrap",
    solve_rate: float = 0.75,
) -> None:
    paths.bootstrap_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(paths.bootstrap_checkpoint, tag=checkpoint)
    paths.bootstrap_metrics.write_text(
        json.dumps(_metrics(solve_rate)),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path, *, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tag": tag,
            "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
            "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            "outcome_value_target_contract_version": (
                OUTCOME_VALUE_TARGET_CONTRACT_VERSION
            ),
            **physics_provenance(DEFAULT_PHYSICS_CONFIG),
        },
        path,
    )


def _metrics(solve_rate: float) -> dict[str, object]:
    return {
        "solve_rate": solve_rate,
        "evaluation_metrics_contract_version": EVALUATION_METRICS_CONTRACT_VERSION,
        **physics_provenance(DEFAULT_PHYSICS_CONFIG),
        "physical_objective_contract": physical_objective_contract(),
    }


def test_initialize_best_state_copies_bootstrap_files(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_bootstrap(paths, checkpoint="checkpoint", solve_rate=0.75)

    state = initialize_best_state(paths=paths)

    assert torch.load(paths.best_checkpoint, weights_only=False)["tag"] == "checkpoint"
    assert paths.best_metrics.is_file()
    assert state.checkpoint == paths.best_checkpoint
    assert state.metrics["solve_rate"] == 0.75


def test_initialize_best_state_does_not_overwrite_existing_best(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_bootstrap(paths, checkpoint="bootstrap", solve_rate=0.1)
    paths.best_checkpoint.parent.mkdir(parents=True)
    _write_checkpoint(paths.best_checkpoint, tag="existing")
    paths.best_metrics.write_text(
        json.dumps(_metrics(0.9)),
        encoding="utf-8",
    )

    state = initialize_best_state(paths=paths)

    assert torch.load(paths.best_checkpoint, weights_only=False)["tag"] == "existing"
    assert json.loads(paths.best_metrics.read_text(encoding="utf-8")) == _metrics(0.9)
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
        json.dumps(_metrics(0.1)),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(candidate, tag="candidate")

    state = promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics=_metrics(0.95),
        paths=paths,
    )

    assert torch.load(paths.best_checkpoint, weights_only=False)["tag"] == "candidate"
    assert json.loads(paths.best_metrics.read_text(encoding="utf-8")) == _metrics(0.95)
    assert state.checkpoint == paths.best_checkpoint
    assert state.metrics == _metrics(0.95)


def test_promote_candidate_requires_checkpoint_file(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="Candidate checkpoint"):
        promote_candidate(
            candidate_checkpoint=tmp_path / "missing.pt",
            candidate_metrics=_metrics(0.95),
            paths=paths,
        )


def test_promote_candidate_creates_parent_directories(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(candidate, tag="candidate")

    promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics=_metrics(0.95),
        paths=paths,
    )

    assert paths.best_checkpoint.is_file()
    assert paths.best_metrics.is_file()


def test_promote_candidate_does_not_mutate_input_metrics(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(candidate, tag="candidate")
    metrics = _metrics(0.95)

    promote_candidate(
        candidate_checkpoint=candidate,
        candidate_metrics=metrics,
        paths=paths,
    )

    assert metrics == _metrics(0.95)
