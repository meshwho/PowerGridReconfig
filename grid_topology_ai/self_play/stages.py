from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from grid_topology_ai.self_play import _stages_base as _base
from grid_topology_ai.training.validation_diagnostics import (
    annotate_example_difficulty,
    validation_diagnostic_options,
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


def _install_stage_overrides() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for name in _BASE_EXPORTS:
        if name in {"run_generate", "run_train", "run_evaluate"}:
            continue
        previous[name] = getattr(_base, name)
        setattr(_base, name, globals()[name])
    return previous


def _restore_stage_overrides(previous: dict[str, Any]) -> None:
    for name, value in previous.items():
        setattr(_base, name, value)


def _calibration_bins(output_dir: Path) -> int:
    config_path = output_dir.parent / "self_play_loop.resolved.yaml"
    if not config_path.is_file():
        return 10

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved self-play config must be a mapping: {config_path}")
    section = payload.get("checkpoint_selection", {})
    if not isinstance(section, dict):
        raise ValueError(
            "checkpoint_selection must be a mapping in resolved config: "
            f"{config_path}"
        )
    value = section.get("calibration_bins", 10)
    if isinstance(value, bool):
        raise ValueError("checkpoint_selection.calibration_bins must be positive.")
    bins = int(value)
    if bins <= 0 or float(value) != float(bins):
        raise ValueError("checkpoint_selection.calibration_bins must be positive.")
    return bins


@wraps(_base.run_generate)
def run_generate(*args: Any, **kwargs: Any) -> Path:
    arguments = _bound_arguments(_base.run_generate, args, kwargs)
    previous = _install_stage_overrides()
    try:
        examples_csv = _base.run_generate(*args, **kwargs)
        output_dir = Path(arguments["output_dir"])
        annotate_example_difficulty(
            examples_csv=examples_csv,
            transitions_csv=output_dir / "selected_transitions.csv",
        )
        return examples_csv
    finally:
        _restore_stage_overrides(previous)


@wraps(_base.run_train)
def run_train(*args: Any, **kwargs: Any) -> Path:
    arguments = _bound_arguments(_base.run_train, args, kwargs)
    bins = _calibration_bins(Path(arguments["output_dir"]))
    previous = _install_stage_overrides()
    try:
        with validation_diagnostic_options(calibration_bins=bins):
            return _base.run_train(*args, **kwargs)
    finally:
        _restore_stage_overrides(previous)


@wraps(_base.run_evaluate)
def run_evaluate(*args: Any, **kwargs: Any):
    previous = _install_stage_overrides()
    try:
        return _base.run_evaluate(*args, **kwargs)
    finally:
        _restore_stage_overrides(previous)


globals()["run_generate"] = run_generate
globals()["run_train"] = run_train
globals()["run_evaluate"] = run_evaluate


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


__all__ = [
    name
    for name in _BASE_EXPORTS
    if not name.startswith("_")
]
