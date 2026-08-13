from __future__ import annotations

import inspect
from typing import Any

import torch

from grid_topology_ai.training import _graph_policy_value_base as _base
from grid_topology_ai.training.checkpoint_candidates import (
    checkpoint_candidate_tracking,
    record_validation_candidates,
    register_training_dataset,
)
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


class _AmpSafeGraphPolicyValueNetV2(_base.GraphPolicyValueNetV2):
    @staticmethod
    def _segment_softmax(
        scores: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if scores.ndim != 1:
            raise ValueError(
                "Segmented softmax expects a 1D score tensor."
            )

        if batch_index.shape != scores.shape:
            raise ValueError(
                "batch_index must match scores."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        work_dtype = (
            torch.float32
            if scores.dtype in (torch.float16, torch.bfloat16)
            else scores.dtype
        )
        work_scores = scores.to(dtype=work_dtype)
        index = batch_index.long()

        maxima = work_scores.new_full(
            (batch_size,),
            torch.finfo(work_scores.dtype).min,
        )
        maxima.scatter_reduce_(
            dim=0,
            index=index,
            src=work_scores,
            reduce="amax",
            include_self=True,
        )

        exponentials = torch.exp(
            work_scores - maxima[index]
        )
        denominators = exponentials.new_zeros(
            batch_size
        )
        denominators.index_add_(
            0,
            index,
            exponentials,
        )

        weights = (
            exponentials
            / denominators[index].clamp_min(1e-12)
        )
        return weights.to(dtype=scores.dtype)


# The training loop is where CUDA autocast is enabled. Keep the model contract
# unchanged while using float32 for the softmax reductions that autocast may
# promote independently from the surrounding half-precision tensors.
globals()["GraphPolicyValueNetV2"] = _AmpSafeGraphPolicyValueNetV2

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


def evaluate_one_epoch(*args: Any, **kwargs: Any) -> dict[str, float]:
    bound = inspect.signature(
        _evaluate_one_epoch_diagnostics
    ).bind(*args, **kwargs)
    bound.apply_defaults()
    metrics = _evaluate_one_epoch_diagnostics(*args, **kwargs)
    loader = bound.arguments["loader"]
    record_validation_candidates(
        model=bound.arguments["model"],
        validation_dataset=loader.dataset,
        metrics=metrics,
        device=bound.arguments["device"],
        use_amp=bound.arguments["use_amp"],
    )
    return metrics


def _install_training_overrides() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for name in _BASE_EXPORTS:
        if name == "train_graph_policy_value_model":
            continue
        previous[name] = getattr(_base, name)
        setattr(_base, name, globals()[name])
    return previous


def _build_model(*args: Any, **kwargs: Any):
    bound = inspect.signature(_legacy_build_model).bind(*args, **kwargs)
    bound.apply_defaults()
    register_training_dataset(bound.arguments["dataset"])

    previous = _install_training_overrides()
    try:
        return _legacy_build_model(*args, **kwargs)
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def train_graph_policy_value_model(request: TrainingRequest):
    with checkpoint_candidate_tracking(request):
        previous = _install_training_overrides()
        try:
            return _legacy_train_graph_policy_value_model(request)
        finally:
            for name, value in previous.items():
                setattr(_base, name, value)


# Override the base implementations exported above.
globals()["evaluate_one_epoch"] = evaluate_one_epoch
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
