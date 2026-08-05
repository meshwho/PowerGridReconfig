from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.checkpoint_provenance import (
    CHECKPOINT_SELECTION_HASH_KEY,
)
from grid_topology_ai.self_play.completion import (
    validate_iteration_completion,
    write_iteration_completion_marker,
)


def _provenance_artifacts(tmp_path: Path) -> dict[str, Path]:
    iteration = 1
    iter_dir = tmp_path / "iter_001"
    selection_dir = iter_dir / "checkpoint_selection"
    candidates_dir = selection_dir / "candidates"
    checkpoints = tmp_path / "checkpoints"
    inputs = tmp_path / "inputs"
    replay = tmp_path / "replay_buffer"
    for path in (
        iter_dir,
        candidates_dir,
        checkpoints,
        inputs,
        replay,
    ):
        path.mkdir(parents=True, exist_ok=True)

    candidate = iter_dir / "candidate_checkpoint.pt"
    candidate.write_bytes(b"arena-winner")
    archived = candidates_dir / "candidate_01_best_policy.pt"
    archived.write_bytes(b"arena-winner")
    report = selection_dir / "checkpoint_selection.json"
    save_json(
        {
            "schema_version": 2,
            "selection_method": "closed_loop_tuning_arena",
            "metric": "physically_secure_rate_requested",
            "metric_direction": "maximize",
            "selected_source_checkpoint": str(
                iter_dir / "candidate_checkpoint_best_policy_loss.pt"
            ),
            "selected_archived_checkpoint": str(archived),
            "selected_checkpoint": str(candidate),
            "selected_checkpoint_sha256": sha256_file(candidate),
            "selected_metric_value": 0.75,
            "candidates": [
                {
                    "source_checkpoint": str(
                        iter_dir
                        / "candidate_checkpoint_best_policy_loss.pt"
                    ),
                    "archived_checkpoint": str(archived),
                    "checkpoint_sha256": sha256_file(archived),
                }
            ],
        },
        report,
    )

    metadata = iter_dir / "metadata.json"
    save_json(
        {
            "iteration": iteration,
            "accepted": True,
            "candidate_checkpoint": str(candidate),
        },
        metadata,
    )
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
        "iteration,checkpoint_selection_metric\n1,validation_loss\n",
        encoding="utf-8",
    )

    return {
        "iter_dir": iter_dir,
        "marker": iter_dir / "iteration_complete.json",
        "metadata": metadata,
        "candidate": candidate,
        "archived": archived,
        "report": report,
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


def test_completion_seals_checkpoint_selection_provenance(
    tmp_path: Path,
) -> None:
    paths = _provenance_artifacts(tmp_path)

    marker_path = _write_completion(paths)

    metadata = load_json(paths["metadata"])
    marker = load_json(marker_path)
    report_hash = sha256_file(paths["report"])
    assert metadata["hashes"][CHECKPOINT_SELECTION_HASH_KEY] == report_hash
    assert (
        metadata["extra"]["checkpoint_selection"]["selection_method"]
        == "closed_loop_tuning_arena"
    )
    assert marker["artifacts"][CHECKPOINT_SELECTION_HASH_KEY] == report_hash

    with paths["learning_curve"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["checkpoint_selection_metric"] == "closed_loop_arena"
    assert row["checkpoint_arena_metric"] == (
        "physically_secure_rate_requested"
    )
    assert row["checkpoint_arena_metric_value"] == "0.75"
    assert row["checkpoint_arena_candidate_count"] == "1"
    assert row[CHECKPOINT_SELECTION_HASH_KEY] == report_hash

    validate_iteration_completion(
        iteration_dir=paths["iter_dir"],
        expected_iteration=1,
    )


def test_completion_detects_checkpoint_selection_report_tampering(
    tmp_path: Path,
) -> None:
    paths = _provenance_artifacts(tmp_path)
    _write_completion(paths)
    report = load_json(paths["report"])
    report["selected_metric_value"] = 0.1
    save_json(report, paths["report"])

    with pytest.raises(
        ValueError,
        match="checkpoint selection report",
    ):
        validate_iteration_completion(
            iteration_dir=paths["iter_dir"],
            expected_iteration=1,
        )


def test_completion_detects_archived_candidate_tampering(
    tmp_path: Path,
) -> None:
    paths = _provenance_artifacts(tmp_path)
    _write_completion(paths)
    paths["archived"].write_bytes(b"tampered")

    with pytest.raises(
        ValueError,
        match="archived checkpoint candidate",
    ):
        validate_iteration_completion(
            iteration_dir=paths["iter_dir"],
            expected_iteration=1,
        )
