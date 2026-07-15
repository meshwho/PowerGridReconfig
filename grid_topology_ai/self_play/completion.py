from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grid_topology_ai.self_play.artifacts import load_json, save_json, sha256_file
from grid_topology_ai.self_play.paths import ITERATION_COMPLETION_FILENAME

COMPLETION_SCHEMA_VERSION = 1
COMPLETION_MARKER_FILENAME = ITERATION_COMPLETION_FILENAME

_VALID_STATUSES = {"ACCEPTED", "REJECTED"}
_REQUIRED_HASHES = {
    "metadata_sha256",
    "candidate_checkpoint_sha256",
    "replay_iteration_sha256",
}


def _validate_status(*, accepted: bool, status: str) -> None:
    if not isinstance(accepted, bool):
        raise ValueError(
            f"accepted must be a bool, got {type(accepted).__name__}"
        )

    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid iteration completion status: {status}")

    expected_accepted = status == "ACCEPTED"
    if accepted != expected_accepted:
        raise ValueError(
            "accepted must match status "
            f"(accepted={accepted!r}, status={status!r})"
        )


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _validate_metadata(path: Path, *, iteration: int, accepted: bool) -> None:
    metadata = load_json(path)
    if int(metadata.get("iteration", -1)) != int(iteration):
        raise ValueError(f"metadata.json iteration does not match {iteration}: {path}")

    metadata_accepted = metadata.get("accepted")
    if not isinstance(metadata_accepted, bool):
        raise ValueError(f"metadata.json accepted must be a bool: {path}")
    if metadata_accepted != accepted:
        raise ValueError(f"metadata.json accepted does not match {accepted}: {path}")


def _validate_pool_metadata(path: Path, *, iteration: int) -> None:
    pool_metadata = load_json(path)
    if int(pool_metadata.get("last_updated_iteration", -1)) != int(iteration):
        raise ValueError(
            f"pool_metadata.json last_updated_iteration does not match {iteration}: {path}"
        )


def _validate_replay_manifest(
    path: Path,
    *,
    iteration: int,
    replay_iteration_path: Path,
) -> None:
    manifest = load_json(path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Replay manifest files must be a list: {path}")

    for item in files:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("iteration", item.get("iter", -1))) != int(iteration):
            continue
        if str(item.get("path")) == replay_iteration_path.name:
            return

    raise ValueError(
        "Replay manifest does not contain expected iteration file "
        f"for iteration {iteration}: {replay_iteration_path.name}"
    )


def _validate_learning_curve(path: Path, *, iteration: int) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row_iteration = int(row.get("iteration", ""))
            except ValueError:
                continue
            if row_iteration == int(iteration):
                return

    raise ValueError(f"learning_curve.csv is missing iteration {iteration}: {path}")


def write_iteration_completion_marker(
    *,
    path: Path,
    iteration: int,
    accepted: bool,
    status: str,
    metadata_path: Path,
    candidate_checkpoint: Path,
    best_checkpoint_after: Path,
    best_metrics_path: Path,
    pool_metadata_path: Path,
    replay_manifest_path: Path,
    replay_iteration_path: Path,
    learning_curve_path: Path,
) -> Path:
    iteration = int(iteration)
    if iteration <= 0:
        raise ValueError("iteration must be > 0")

    _validate_status(accepted=accepted, status=status)

    if path.exists():
        raise FileExistsError(f"Iteration completion marker already exists: {path}")

    for label, artifact_path in {
        "metadata_path": metadata_path,
        "candidate_checkpoint": candidate_checkpoint,
        "best_checkpoint_after": best_checkpoint_after,
        "best_metrics_path": best_metrics_path,
        "pool_metadata_path": pool_metadata_path,
        "replay_manifest_path": replay_manifest_path,
        "replay_iteration_path": replay_iteration_path,
        "learning_curve_path": learning_curve_path,
    }.items():
        _require_file(Path(artifact_path), label=label)

    _validate_metadata(metadata_path, iteration=iteration, accepted=accepted)
    _validate_pool_metadata(pool_metadata_path, iteration=iteration)
    _validate_replay_manifest(
        replay_manifest_path,
        iteration=iteration,
        replay_iteration_path=replay_iteration_path,
    )
    _validate_learning_curve(learning_curve_path, iteration=iteration)

    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "iteration": iteration,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": bool(accepted),
        "status": status,
        "artifacts": {
            "metadata_sha256": sha256_file(metadata_path),
            "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
            "replay_iteration_sha256": sha256_file(replay_iteration_path),
        },
        "best_checkpoint_after": str(best_checkpoint_after),
        "best_metrics_path": str(best_metrics_path),
        "pool_metadata_path": str(pool_metadata_path),
        "replay_manifest_path": str(replay_manifest_path),
        "learning_curve_path": str(learning_curve_path),
    }

    return save_json(payload, path)


def load_iteration_completion_marker(
    path: Path,
    *,
    expected_iteration: int | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Iteration completion marker not found: {path}")

    payload = load_json(path)

    if payload.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise ValueError(f"Invalid completion marker schema_version: {path}")

    iteration = int(payload.get("iteration", -1))
    if iteration <= 0:
        raise ValueError(f"Invalid completion marker iteration: {path}")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise ValueError(
            f"Completion marker iteration {iteration} does not match "
            f"expected {expected_iteration}: {path}"
        )

    status = payload.get("status")
    accepted = payload.get("accepted")
    if not isinstance(status, str):
        raise ValueError(f"Completion marker status must be a string: {path}")
    if not isinstance(accepted, bool):
        raise ValueError(f"Completion marker accepted must be a bool: {path}")
    _validate_status(accepted=accepted, status=status)

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Completion marker artifacts must be an object: {path}")

    missing = [key for key in sorted(_REQUIRED_HASHES) if not artifacts.get(key)]
    if missing:
        raise ValueError(
            f"Completion marker artifacts are missing required hashes {missing}: {path}"
        )

    return dict(payload)


def validate_iteration_completion(
    *,
    iteration_dir: Path,
    expected_iteration: int,
) -> dict[str, object]:
    marker_path = iteration_dir / COMPLETION_MARKER_FILENAME
    marker = load_iteration_completion_marker(
        marker_path,
        expected_iteration=expected_iteration,
    )
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Completion marker artifacts must be an object: {marker_path}")

    artifact_paths = {
        "metadata_sha256": iteration_dir / "metadata.json",
        "candidate_checkpoint_sha256": iteration_dir / "candidate_checkpoint.pt",
        "replay_iteration_sha256": (
            iteration_dir.parent
            / "replay_buffer"
            / f"buffer_iter_{int(expected_iteration):03d}.jsonl.gz"
        ),
    }

    for hash_key, artifact_path in artifact_paths.items():
        _require_file(artifact_path, label=hash_key)
        actual_hash = sha256_file(artifact_path)
        if actual_hash != artifacts.get(hash_key):
            raise ValueError(
                f"Corrupt completed iteration artifact {hash_key}: {artifact_path}"
            )

    return marker
