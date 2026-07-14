from __future__ import annotations

from pathlib import Path

from grid_topology_ai.self_play.paths import SelfPlayPaths


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_inputs(
    paths: SelfPlayPaths,
    *,
    require_bootstrap: bool,
) -> tuple[str, ...]:
    _require_file(
        paths.pool_transitions_csv,
        "Pool transitions CSV",
    )
    _require_directory(
        paths.pool_raw_dir,
        "Pool raw directory",
    )
    _require_file(
        paths.eval_csv,
        "Evaluation transitions CSV",
    )
    _require_directory(
        paths.eval_raw_dir,
        "Evaluation raw directory",
    )

    bootstrap_files = (
        (
            paths.bootstrap_checkpoint,
            "Bootstrap checkpoint",
        ),
        (
            paths.bootstrap_metrics,
            "Bootstrap evaluation metrics",
        ),
    )

    if require_bootstrap:
        for path, label in bootstrap_files:
            _require_file(path, label)
        return ()

    warnings = [
        f"{label} is missing: {path}"
        for path, label in bootstrap_files
        if not path.is_file()
    ]

    return tuple(warnings)


def validate_resume_artifacts(
    paths: SelfPlayPaths,
) -> None:
    required_files = (
        (paths.best_checkpoint, "Resume best checkpoint"),
        (paths.best_metrics, "Resume best metrics"),
        (paths.pool_metadata, "Resume pool metadata"),
        (paths.replay_manifest, "Resume replay manifest"),
    )

    for path, label in required_files:
        _require_file(path, label)