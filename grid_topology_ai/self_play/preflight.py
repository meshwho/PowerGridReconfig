from __future__ import annotations

from pathlib import Path

from grid_topology_ai.evaluation.checkpoint import (
    load_scenario_ids,
)
from grid_topology_ai.self_play.dataset_isolation import (
    validate_physical_dataset_isolation,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_disjoint_scenario_ids(
    first_csv: Path,
    first_label: str,
    second_csv: Path,
    second_label: str,
) -> None:
    first_ids = set(
        load_scenario_ids(
            first_csv,
            limit=None,
        )
    )
    second_ids = set(
        load_scenario_ids(
            second_csv,
            limit=None,
        )
    )

    overlap = sorted(first_ids & second_ids)
    if not overlap:
        return

    raise ValueError(
        f"{first_label} and {second_label} "
        "scenario IDs overlap: "
        f"{overlap[:20]}"
    )


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
    _require_file(
        paths.final_test_csv,
        "Final-test transitions CSV",
    )
    _require_directory(
        paths.final_test_raw_dir,
        "Final-test raw directory",
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

    _require_disjoint_scenario_ids(
        paths.pool_transitions_csv,
        "Pool",
        paths.eval_csv,
        "Evaluation",
    )
    _require_disjoint_scenario_ids(
        paths.pool_transitions_csv,
        "Pool",
        paths.final_test_csv,
        "final-test",
    )
    _require_disjoint_scenario_ids(
        paths.eval_csv,
        "Evaluation",
        paths.final_test_csv,
        "final-test",
    )

    validate_physical_dataset_isolation(
        pool_transitions_csv=paths.pool_transitions_csv,
        pool_raw_dir=paths.pool_raw_dir,
        eval_transitions_csv=paths.eval_csv,
        eval_raw_dir=paths.eval_raw_dir,
        final_test_transitions_csv=paths.final_test_csv,
        final_test_raw_dir=paths.final_test_raw_dir,
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
        (
            paths.physical_split_manifest,
            "Resume physical split manifest",
        ),
    )

    for path, label in required_files:
        _require_file(path, label)
