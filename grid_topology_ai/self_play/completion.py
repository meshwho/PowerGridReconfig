from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any

from grid_topology_ai.self_play import _completion_base as _base
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.checkpoint_provenance import (
    CHECKPOINT_SELECTION_HASH_KEY,
    CHECKPOINT_SELECTION_REPORT,
    attach_checkpoint_selection_provenance,
    validate_checkpoint_selection_provenance,
)


_BASE_EXPORTS = tuple(
    name for name in dir(_base) if not name.startswith("__")
)
for _name in _BASE_EXPORTS:
    globals()[_name] = getattr(_base, _name)


def _bound_arguments(function, args, kwargs) -> dict[str, Any]:
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


@wraps(_base.write_iteration_completion_marker)
def write_iteration_completion_marker(*args: Any, **kwargs: Any) -> Path:
    arguments = _bound_arguments(
        _base.write_iteration_completion_marker,
        args,
        kwargs,
    )
    report_path = attach_checkpoint_selection_provenance(
        metadata_path=Path(arguments["metadata_path"]),
        learning_curve_path=Path(arguments["learning_curve_path"]),
        iteration=int(arguments["iteration"]),
    )
    marker_path = Path(
        _base.write_iteration_completion_marker(*args, **kwargs)
    )
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


@wraps(_base.validate_iteration_completion)
def validate_iteration_completion(*args: Any, **kwargs: Any):
    arguments = _bound_arguments(
        _base.validate_iteration_completion,
        args,
        kwargs,
    )
    marker = _base.validate_iteration_completion(*args, **kwargs)
    iteration_dir = Path(arguments["iteration_dir"])
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


globals()["write_iteration_completion_marker"] = (
    write_iteration_completion_marker
)
globals()["validate_iteration_completion"] = validate_iteration_completion


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


__all__ = [
    name
    for name in _BASE_EXPORTS
    if not name.startswith("_")
]
