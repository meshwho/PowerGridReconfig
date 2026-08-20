from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.artifacts import load_json, save_json, sha256_file
from grid_topology_ai.self_play.checkpoint_provenance import (
    CHECKPOINT_SELECTION_HASH_KEY,
    CHECKPOINT_SELECTION_REPORT,
    attach_checkpoint_selection_provenance,
    validate_checkpoint_selection_provenance,
)


ITERATION_COMPLETION_FILENAME = "iteration_complete.json"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_MARKER_FILENAME = ITERATION_COMPLETION_FILENAME

_VALID_STATUSES = {"ACCEPTED", "REJECTED"}
_REQUIRED_HASHES = {
    "metadata_sha256",
    "candidate_checkpoint_sha256",
    "replay_iteration_sha256",
}
_CURRICULUM_REPORT_FILENAME = "curriculum_sampling.json"
_CURRICULUM_HASH_KEY = "curriculum_sampling_sha256"


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def discover_project_root(start: str | Path | None = None) -> Path:
    """
    Find repository root by walking upward until project markers are found.
    """

    current = Path.cwd() if start is None else Path(start).resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (
            (candidate / "grid_topology_ai").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not discover project root. Run from inside PowerGridReconfig."
    )


@dataclass(frozen=True, slots=True)
class SelfPlayPaths:
    project_root: Path
    run_dir: Path

    pool_transitions_csv: Path
    pool_raw_dir: Path
    pool_metadata: Path

    eval_csv: Path
    eval_raw_dir: Path

    final_test_csv: Path
    final_test_raw_dir: Path

    bootstrap_checkpoint: Path
    bootstrap_metrics: Path

    best_checkpoint: Path
    best_metrics: Path

    tuning_csv: Path | None = None
    tuning_raw_dir: Path | None = None

    @classmethod
    def from_config(
        cls,
        config: SelfPlayConfig,
        project_root: str | Path,
    ) -> "SelfPlayPaths":
        root = Path(project_root).resolve()
        checkpoint_selection = config.checkpoint_selection
        tuning_csv = (
            checkpoint_selection.tuning_csv
            if checkpoint_selection.enabled
            else None
        )
        tuning_raw_dir = (
            checkpoint_selection.tuning_raw_dir
            if checkpoint_selection.enabled
            else None
        )

        return cls(
            project_root=root,
            run_dir=_resolve(root, config.checkpoint_dir),
            pool_transitions_csv=_resolve(
                root,
                config.pool.transitions_csv,
            ),
            pool_raw_dir=_resolve(
                root,
                config.pool.raw_dir,
            ),
            pool_metadata=_resolve(
                root,
                config.pool.metadata_path,
            ),
            eval_csv=_resolve(root, config.eval_csv),
            eval_raw_dir=_resolve(root, config.eval_raw_dir),
            final_test_csv=_resolve(
                root,
                config.final_test_csv,
            ),
            final_test_raw_dir=_resolve(
                root,
                config.final_test_raw_dir,
            ),
            bootstrap_checkpoint=_resolve(
                root,
                config.bootstrap_checkpoint,
            ),
            bootstrap_metrics=_resolve(
                root,
                config.bootstrap_eval_metrics,
            ),
            best_checkpoint=_resolve(
                root,
                config.best_checkpoint_path,
            ),
            best_metrics=_resolve(
                root,
                config.best_metrics_path,
            ),
            tuning_csv=(
                None
                if tuning_csv is None
                else _resolve(root, tuning_csv)
            ),
            tuning_raw_dir=(
                None
                if tuning_raw_dir is None
                else _resolve(root, tuning_raw_dir)
            ),
        )

    @property
    def replay_dir(self) -> Path:
        return self.run_dir / "replay_buffer"

    @property
    def replay_manifest(self) -> Path:
        return self.replay_dir / "buffer_manifest.json"

    @property
    def physical_split_manifest(self) -> Path:
        return self.run_dir / "physical_split_manifest.json"

    @property
    def physical_validation_snapshot(self) -> Path:
        return self.run_dir / "physical_validation_examples.csv"

    @property
    def physical_validation_snapshot_metadata(self) -> Path:
        return self.run_dir / "physical_validation_examples.metadata.json"

    @property
    def final_test_dir(self) -> Path:
        return self.run_dir / "final_test"

    @property
    def final_test_report(self) -> Path:
        return self.final_test_dir / "final_test_report.json"

    @property
    def learning_curve(self) -> Path:
        return self.run_dir / "learning_curve.csv"

    @property
    def resolved_config(self) -> Path:
        return self.run_dir / "self_play_loop.resolved.yaml"

    def iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iter_{iteration:03d}"

    def iteration_completion_marker(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / ITERATION_COMPLETION_FILENAME

    def replay_iteration_file(self, iteration: int) -> Path:
        return self.replay_dir / f"buffer_iter_{iteration:03d}.jsonl.gz"


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


def _validate_curriculum_sampling(metadata_path: Path) -> None:
    metadata = load_json(metadata_path)
    report_path = metadata_path.parent / _CURRICULUM_REPORT_FILENAME
    hashes = metadata.get("hashes")
    extra = metadata.get("extra")

    has_curriculum_fields = (
        isinstance(hashes, Mapping)
        and _CURRICULUM_HASH_KEY in hashes
    ) or (
        isinstance(extra, Mapping)
        and any(
            key in extra
            for key in (
                _CURRICULUM_HASH_KEY,
                "curriculum_sampling",
                "curriculum_sampling_path",
            )
        )
    )
    if not report_path.exists() and not has_curriculum_fields:
        return

    if not isinstance(hashes, Mapping):
        raise ValueError(
            f"metadata.json hashes must contain curriculum data: {metadata_path}"
        )
    if not isinstance(extra, Mapping):
        raise ValueError(
            f"metadata.json extra must contain curriculum data: {metadata_path}"
        )

    expected_hash = hashes.get(_CURRICULUM_HASH_KEY)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "metadata.json is missing curriculum sampling hash: "
            f"{metadata_path}"
        )
    if extra.get(_CURRICULUM_HASH_KEY) != expected_hash:
        raise ValueError(
            "metadata.json curriculum sampling hashes disagree: "
            f"{metadata_path}"
        )

    _require_file(report_path, label=_CURRICULUM_REPORT_FILENAME)
    actual_hash = sha256_file(report_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Corrupt curriculum sampling report: "
            f"{report_path}"
        )

    stored_path = extra.get("curriculum_sampling_path")
    if (
        not isinstance(stored_path, str)
        or Path(stored_path).name != _CURRICULUM_REPORT_FILENAME
    ):
        raise ValueError(
            "metadata.json has invalid curriculum sampling path: "
            f"{metadata_path}"
        )

    stored_report = extra.get("curriculum_sampling")
    if not isinstance(stored_report, Mapping):
        raise ValueError(
            "metadata.json is missing curriculum sampling payload: "
            f"{metadata_path}"
        )

    report = load_json(report_path)
    if dict(stored_report) != report:
        raise ValueError(
            "metadata.json curriculum sampling payload does not match report: "
            f"{report_path}"
        )
    if int(report.get("iteration", -1)) != int(metadata.get("iteration", -1)):
        raise ValueError(
            "Curriculum sampling report iteration does not match metadata: "
            f"{report_path}"
        )


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
    report_path = attach_checkpoint_selection_provenance(
        metadata_path=Path(metadata_path),
        learning_curve_path=Path(learning_curve_path),
        iteration=int(iteration),
    )

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
    _validate_curriculum_sampling(metadata_path)
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

    marker_path = save_json(payload, path)
    if report_path is None:
        return marker_path

    payload = load_json(marker_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            f"Completion marker artifacts must be an object: {marker_path}"
        )
    artifacts[CHECKPOINT_SELECTION_HASH_KEY] = sha256_file(report_path)
    save_json(payload, marker_path)
    return marker_path


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

    _validate_curriculum_sampling(iteration_dir / "metadata.json")

    metadata_path = iteration_dir / "metadata.json"
    report_path = validate_checkpoint_selection_provenance(metadata_path)
    if report_path is None:
        return marker

    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            "Completion marker artifacts must be an object: "
            f"{iteration_dir}"
        )
    expected_hash = artifacts.get(CHECKPOINT_SELECTION_HASH_KEY)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "Completion marker is missing checkpoint selection hash: "
            f"{iteration_dir}"
        )
    actual_path = iteration_dir / CHECKPOINT_SELECTION_REPORT
    if actual_path != report_path or sha256_file(actual_path) != expected_hash:
        raise ValueError(
            "Corrupt completed iteration checkpoint selection report: "
            f"{actual_path}"
        )
    return marker
