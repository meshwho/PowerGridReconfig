from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play import _stages_base as _base
from grid_topology_ai.self_play.checkpoint_arena import (
    select_checkpoint_in_tuning_arena,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
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


def _resolved_self_play_config(
    output_dir: Path,
) -> SelfPlayConfig | None:
    config_path = output_dir.parent / "self_play_loop.resolved.yaml"
    if not config_path.is_file():
        return None

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Resolved self-play config must be a mapping: {config_path}"
        )
    return SelfPlayConfig.from_mapping(payload)


@wraps(_base.run_generate)
def run_generate(*args: Any, **kwargs: Any) -> Path:
    arguments = _bound_arguments(_base.run_generate, args, kwargs)
    previous = _install_stage_overrides()
    try:
        examples_csv = _base.run_generate(*args, **kwargs)
        output_dir = Path(arguments["output_dir"])
        columns = set(pd.read_csv(examples_csv, nrows=0).columns)
        if "scenario_id" in columns:
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
    output_dir = Path(arguments["output_dir"])
    self_play_config = _resolved_self_play_config(output_dir)
    training_config = arguments["config"]
    if (
        self_play_config is None
        and bool(training_config.save_multiple_best)
    ):
        config_path = output_dir.parent / "self_play_loop.resolved.yaml"
        raise RuntimeError(
            "Multiple checkpoint candidates require the resolved self-play "
            f"config before training: {config_path}"
        )

    bins = (
        10
        if self_play_config is None
        else int(
            self_play_config.checkpoint_selection.calibration_bins
        )
    )
    previous = _install_stage_overrides()
    try:
        with validation_diagnostic_options(calibration_bins=bins):
            checkpoint = Path(_base.run_train(*args, **kwargs))
    finally:
        _restore_stage_overrides(previous)

    if (
        self_play_config is None
        or not self_play_config.checkpoint_selection.enabled
    ):
        return checkpoint

    project_root = Path(arguments["project_root"])
    paths = SelfPlayPaths.from_config(
        self_play_config,
        project_root,
    )
    if paths.tuning_csv is None or paths.tuning_raw_dir is None:
        raise RuntimeError(
            "Enabled checkpoint selection has no resolved tuning paths."
        )

    selection = select_checkpoint_in_tuning_arena(
        canonical_checkpoint=checkpoint,
        project_root=project_root,
        output_dir=output_dir / "checkpoint_selection",
        config=self_play_config.checkpoint_selection,
        physics_config=arguments["physics_config"],
        tuning_csv=paths.tuning_csv,
        tuning_raw_dir=paths.tuning_raw_dir,
        excluded_csvs={
            "self-play pool": paths.pool_transitions_csv,
            "evaluation set": paths.eval_csv,
            "final test set": paths.final_test_csv,
        },
        evaluate=run_evaluate,
    )
    print("\nClosed-loop checkpoint arena:")
    print(f"  candidates: {selection.candidate_count}")
    print(f"  winner:     {selection.selected_source}")
    print(
        f"  {selection.metric_name}: "
        f"{selection.metric_value:.6f}"
    )
    print(f"  report:     {selection.report_path}")
    return selection.checkpoint


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
