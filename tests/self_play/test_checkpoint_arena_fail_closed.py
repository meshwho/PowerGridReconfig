from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.paths import (
    validate_iteration_completion,
    write_iteration_completion_marker,
)


def _completion_artifacts(
    tmp_path: Path,
    *,
    selection_enabled: bool | None,
) -> dict[str, Path]:
    iteration = 1
    iter_dir = tmp_path / "iter_001"
    checkpoints = tmp_path / "checkpoints"
    inputs = tmp_path / "inputs"
    replay = tmp_path / "replay_buffer"

    for path in (iter_dir, checkpoints, inputs, replay):
        path.mkdir(parents=True, exist_ok=True)

    metadata_path = iter_dir / "metadata.json"
    metadata: dict[str, object] = {
        "iteration": iteration,
        "accepted": True,
    }
    if selection_enabled is not None:
        metadata["config"] = {
            "checkpoint_selection": {
                "enabled": selection_enabled,
            }
        }
    save_json(metadata, metadata_path)

    candidate_checkpoint = iter_dir / "candidate_checkpoint.pt"
    candidate_checkpoint.write_bytes(b"candidate")
    best_checkpoint = checkpoints / "best.pt"
    best_checkpoint.write_bytes(b"best")
    best_metrics = checkpoints / "best_metrics.json"
    save_json({"score": 1.0}, best_metrics)
    pool_metadata = inputs / "pool_metadata.json"
    save_json({"last_updated_iteration": iteration}, pool_metadata)

    replay_iteration = replay / "buffer_iter_001.jsonl.gz"
    with gzip.open(replay_iteration, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"x": 1}) + "\n")
    replay_manifest = replay / "buffer_manifest.json"
    save_json(
        {
            "files": [
                {
                    "iteration": iteration,
                    "path": replay_iteration.name,
                }
            ]
        },
        replay_manifest,
    )

    learning_curve = tmp_path / "learning_curve.csv"
    learning_curve.write_text(
        "iteration,accepted,status\n1,True,ACCEPTED\n",
        encoding="utf-8",
    )

    return {
        "iter_dir": iter_dir,
        "marker": iter_dir / "iteration_complete.json",
        "metadata": metadata_path,
        "candidate": candidate_checkpoint,
        "best_checkpoint": best_checkpoint,
        "best_metrics": best_metrics,
        "pool_metadata": pool_metadata,
        "replay_iteration": replay_iteration,
        "replay_manifest": replay_manifest,
        "learning_curve": learning_curve,
    }


def _write_completion(paths: dict[str, Path]) -> Path:
    return write_iteration_completion_marker(
        path=paths["marker"],
        iteration=1,
        accepted=True,
        status="ACCEPTED",
        metadata_path=paths["metadata"],
        candidate_checkpoint=paths["candidate"],
        best_checkpoint_after=paths["best_checkpoint"],
        best_metrics_path=paths["best_metrics"],
        pool_metadata_path=paths["pool_metadata"],
        replay_manifest_path=paths["replay_manifest"],
        replay_iteration_path=paths["replay_iteration"],
        learning_curve_path=paths["learning_curve"],
    )


def test_multiple_checkpoint_training_requires_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    train_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    validation_csv.write_text("scenario_id\n2\n", encoding="utf-8")
    training_started = False

    def fake_train(request):
        nonlocal training_started
        training_started = True
        return request.output_path

    monkeypatch.setattr(
        stages,
        "train_graph_policy_value_model",
        fake_train,
    )

    with pytest.raises(
        RuntimeError,
        match="resolved self-play config before training",
    ):
        stages.run_train(
            project_root=tmp_path,
            examples_csv=train_csv,
            validation_examples_csv=validation_csv,
            init_checkpoint=tmp_path / "best.pt",
            output_dir=tmp_path / "iter_001",
            config=TrainingConfig(save_multiple_best=True),
            physics_config=DEFAULT_PHYSICS_CONFIG,
            iteration=1,
            seed=8,
        )

    assert training_started is False


def test_completion_requires_arena_report_when_enabled(
    tmp_path: Path,
) -> None:
    paths = _completion_artifacts(
        tmp_path,
        selection_enabled=True,
    )

    with pytest.raises(
        FileNotFoundError,
        match="checkpoint selection report",
    ):
        _write_completion(paths)

    assert not paths["marker"].exists()


def test_resume_requires_arena_report_when_enabled(
    tmp_path: Path,
) -> None:
    paths = _completion_artifacts(
        tmp_path,
        selection_enabled=None,
    )
    _write_completion(paths)

    metadata = load_json(paths["metadata"])
    metadata["config"] = {
        "checkpoint_selection": {
            "enabled": True,
        }
    }
    save_json(metadata, paths["metadata"])

    marker = load_json(paths["marker"])
    marker["artifacts"]["metadata_sha256"] = sha256_file(
        paths["metadata"]
    )
    save_json(marker, paths["marker"])

    with pytest.raises(
        FileNotFoundError,
        match="checkpoint selection report",
    ):
        validate_iteration_completion(
            iteration_dir=paths["iter_dir"],
            expected_iteration=1,
        )


def test_legacy_completion_allows_missing_arena_report(
    tmp_path: Path,
) -> None:
    paths = _completion_artifacts(
        tmp_path,
        selection_enabled=None,
    )

    marker_path = _write_completion(paths)
    marker = validate_iteration_completion(
        iteration_dir=paths["iter_dir"],
        expected_iteration=1,
    )

    assert marker_path == paths["marker"]
    assert marker["iteration"] == 1
