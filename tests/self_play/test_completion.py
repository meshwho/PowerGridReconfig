from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from grid_topology_ai.self_play.artifacts import load_json, save_json, sha256_file
from grid_topology_ai.self_play.completion import (
    COMPLETION_SCHEMA_VERSION,
    load_iteration_completion_marker,
    validate_iteration_completion,
    write_iteration_completion_marker,
)


@pytest.fixture
def completion_artifacts(tmp_path: Path) -> dict[str, Path]:
    iteration = 1
    iter_dir = tmp_path / "iter_001"
    checkpoints = tmp_path / "checkpoints"
    inputs = tmp_path / "inputs"
    replay = tmp_path / "replay_buffer"

    iter_dir.mkdir()
    checkpoints.mkdir()
    inputs.mkdir()
    replay.mkdir()

    metadata_path = iter_dir / "metadata.json"
    candidate_checkpoint = iter_dir / "candidate_checkpoint.pt"
    best_checkpoint = checkpoints / "best.pt"
    best_metrics = checkpoints / "best_metrics.json"
    pool_metadata = inputs / "pool_metadata.json"
    replay_iteration = replay / "buffer_iter_001.jsonl.gz"
    replay_manifest = replay / "buffer_manifest.json"
    learning_curve = tmp_path / "learning_curve.csv"

    save_json({"iteration": iteration, "accepted": True}, metadata_path)
    candidate_checkpoint.write_bytes(b"candidate")
    best_checkpoint.write_bytes(b"best")
    save_json({"score": 1.0}, best_metrics)
    save_json({"last_updated_iteration": iteration}, pool_metadata)
    with gzip.open(replay_iteration, "wt", encoding="utf-8") as file:
        file.write(json.dumps({"x": 1}) + "\n")
    save_json(
        {"files": [{"iteration": iteration, "path": replay_iteration.name}]},
        replay_manifest,
    )
    learning_curve.write_text(
        "iteration,accepted,status\n1,True,ACCEPTED\n",
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "iter_dir": iter_dir,
        "marker": iter_dir / "iteration_complete.json",
        "metadata_path": metadata_path,
        "candidate_checkpoint": candidate_checkpoint,
        "best_checkpoint": best_checkpoint,
        "best_metrics": best_metrics,
        "pool_metadata": pool_metadata,
        "replay_iteration": replay_iteration,
        "replay_manifest": replay_manifest,
        "learning_curve": learning_curve,
    }


def _write_marker(paths: dict[str, Path]) -> Path:
    return write_iteration_completion_marker(
        path=paths["marker"],
        iteration=1,
        accepted=True,
        status="ACCEPTED",
        metadata_path=paths["metadata_path"],
        candidate_checkpoint=paths["candidate_checkpoint"],
        best_checkpoint_after=paths["best_checkpoint"],
        best_metrics_path=paths["best_metrics"],
        pool_metadata_path=paths["pool_metadata"],
        replay_manifest_path=paths["replay_manifest"],
        replay_iteration_path=paths["replay_iteration"],
        learning_curve_path=paths["learning_curve"],
    )


def test_completion_marker_is_written_atomically(
    completion_artifacts: dict[str, Path],
) -> None:
    marker_path = _write_marker(completion_artifacts)

    marker = load_json(marker_path)

    assert marker_path.is_file()
    assert marker["schema_version"] == COMPLETION_SCHEMA_VERSION
    assert marker["iteration"] == 1
    assert marker["accepted"] is True
    assert marker["status"] == "ACCEPTED"
    assert marker["artifacts"] == {
        "metadata_sha256": sha256_file(completion_artifacts["metadata_path"]),
        "candidate_checkpoint_sha256": sha256_file(
            completion_artifacts["candidate_checkpoint"]
        ),
        "replay_iteration_sha256": sha256_file(
            completion_artifacts["replay_iteration"]
        ),
    }


def test_completion_marker_rejects_existing_marker(
    completion_artifacts: dict[str, Path],
) -> None:
    _write_marker(completion_artifacts)

    with pytest.raises(FileExistsError):
        _write_marker(completion_artifacts)


@pytest.mark.parametrize(
    "artifact_key",
    [
        "metadata_path",
        "candidate_checkpoint",
        "best_checkpoint",
        "best_metrics",
        "pool_metadata",
        "replay_manifest",
        "replay_iteration",
        "learning_curve",
    ],
)
def test_completion_marker_requires_all_artifacts(
    completion_artifacts: dict[str, Path],
    artifact_key: str,
) -> None:
    completion_artifacts[artifact_key].unlink()

    with pytest.raises(FileNotFoundError):
        _write_marker(completion_artifacts)


def test_completion_marker_validates_metadata(
    completion_artifacts: dict[str, Path],
) -> None:
    save_json({"iteration": 2, "accepted": True}, completion_artifacts["metadata_path"])

    with pytest.raises(ValueError, match="metadata.json iteration"):
        _write_marker(completion_artifacts)


def test_load_completion_marker_rejects_corrupt_schema(
    completion_artifacts: dict[str, Path],
) -> None:
    save_json(
        {
            "schema_version": 999,
            "iteration": 1,
            "accepted": True,
            "status": "ACCEPTED",
            "artifacts": {
                "metadata_sha256": "x",
                "candidate_checkpoint_sha256": "y",
                "replay_iteration_sha256": "z",
            },
        },
        completion_artifacts["marker"],
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_iteration_completion_marker(completion_artifacts["marker"])


def test_validate_iteration_completion_detects_hash_mismatch(
    completion_artifacts: dict[str, Path],
) -> None:
    _write_marker(completion_artifacts)
    completion_artifacts["candidate_checkpoint"].write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="candidate_checkpoint_sha256"):
        validate_iteration_completion(
            iteration_dir=completion_artifacts["iter_dir"],
            expected_iteration=1,
        )
