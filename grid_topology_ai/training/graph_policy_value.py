from __future__ import annotations

from typing import Any

from grid_topology_ai.training import _graph_policy_value_base as _base
from grid_topology_ai.training.validation_diagnostics import (
    attach_validation_metrics,
    evaluate_one_epoch as _evaluate_one_epoch_diagnostics,
    log_epoch_metrics as _log_epoch_diagnostics,
)


_BASE_EXPORTS = tuple(
    name for name in dir(_base) if not name.startswith("__")
)
for _name in _BASE_EXPORTS:
    globals()[_name] = getattr(_base, _name)

_legacy_make_checkpoint = _base.make_checkpoint
_legacy_log_epoch_metrics = _base.log_epoch_metrics
_legacy_build_model = _base._build_model
_legacy_train_graph_policy_value_model = (
    _base.train_graph_policy_value_model
)


def make_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
    checkpoint = _legacy_make_checkpoint(*args, **kwargs)
    return attach_validation_metrics(checkpoint)


def log_epoch_metrics(**kwargs: Any) -> None:
    _log_epoch_diagnostics(
        _legacy_log_epoch_metrics,
        **kwargs,
    )


def _install_training_overrides() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for name in _BASE_EXPORTS:
        if name == "train_graph_policy_value_model":
            continue
        previous[name] = getattr(_base, name)
        setattr(_base, name, globals()[name])
    return previous


def _build_model(*args: Any, **kwargs: Any):
    previous = _install_training_overrides()
    try:
        return _legacy_build_model(*args, **kwargs)
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def train_graph_policy_value_model(request: TrainingRequest):
    previous = _install_training_overrides()
    try:
        return _legacy_train_graph_policy_value_model(request)
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


# Override the base implementations exported above.
globals()["evaluate_one_epoch"] = _evaluate_one_epoch_diagnostics
globals()["_build_model"] = _build_model
globals()["make_checkpoint"] = make_checkpoint
globals()["log_epoch_metrics"] = log_epoch_metrics
globals()["train_graph_policy_value_model"] = train_graph_policy_value_model


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


__all__ = [
    name
    for name in _BASE_EXPORTS
    if not name.startswith("_")
]
