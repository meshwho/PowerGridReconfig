from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping

import math
import torch

from grid_topology_ai.models.graph_self_play_dataset import (
    GraphSelfPlayDataset,
)
from grid_topology_ai.training.checkpoints import (
    checkpoint_variant_path,
    make_checkpoint,
)

if TYPE_CHECKING:
    from grid_topology_ai.training.graph_policy_value import TrainingRequest


_CANDIDATE_SELECTORS = (
    (
        "policy_loss",
        "best_policy_loss",
        "validation_policy_loss",
    ),
    (
        "value_loss",
        "best_value_loss",
        "validation_value_loss",
    ),
    (
        "value_calibration_error",
        "best_calibration",
        "validation_value_calibration_error",
    ),
)


@dataclass(slots=True)
class _CandidateTrackingState:
    request: TrainingRequest
    training_dataset: GraphSelfPlayDataset | None = None
    epoch: int = 0
    best_values: dict[str, float] = field(
        default_factory=lambda: {
            metric_name: float("inf")
            for metric_name, _, _ in _CANDIDATE_SELECTORS
        }
    )


_TRACKING_STATE: ContextVar[_CandidateTrackingState | None] = ContextVar(
    "checkpoint_candidate_tracking_state",
    default=None,
)


@contextmanager
def checkpoint_candidate_tracking(
    request: TrainingRequest,
) -> Iterator[None]:
    if not request.config.save_multiple_best:
        yield
        return

    token = _TRACKING_STATE.set(
        _CandidateTrackingState(request=request)
    )
    try:
        yield
    finally:
        _TRACKING_STATE.reset(token)


def register_training_dataset(
    dataset: GraphSelfPlayDataset,
) -> None:
    state = _TRACKING_STATE.get()
    if state is not None:
        state.training_dataset = dataset


def _normalization_metadata(
    request: TrainingRequest,
) -> dict[str, object]:
    from_init = request.init_checkpoint is not None
    return {
        "normalization_contract_version": 1,
        "normalization_source": (
            "init_checkpoint" if from_init else "training_dataset"
        ),
        "normalization_frozen_from_init_checkpoint": from_init,
        "normalization_source_checkpoint": (
            None
            if request.init_checkpoint is None
            else str(request.init_checkpoint)
        ),
    }


def _finite_metric(
    metrics: Mapping[str, object],
    name: str,
) -> float | None:
    if name not in metrics:
        return None
    try:
        value = float(metrics[name])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _save_candidate(
    *,
    path: Path,
    model: torch.nn.Module,
    training_dataset: GraphSelfPlayDataset,
    validation_dataset: GraphSelfPlayDataset,
    request: TrainingRequest,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    metric_name: str,
    selector_name: str,
    selector_value: float,
    val_metrics: Mapping[str, object],
) -> None:
    checkpoint = make_checkpoint(
        model=model,
        dataset=training_dataset,
        request=request,
        device=device,
        use_amp=use_amp,
        normalization_metadata=_normalization_metadata(request),
        validation_dataset=validation_dataset,
    )
    checkpoint["saved_epoch"] = int(epoch)
    checkpoint["selector_name"] = selector_name
    checkpoint["selector_value"] = float(selector_value)
    checkpoint["checkpoint_selection_metric"] = metric_name
    checkpoint["val_metrics"] = {
        key: float(value)
        for key, value in val_metrics.items()
        if isinstance(value, (int, float))
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def record_validation_candidates(
    *,
    model: torch.nn.Module,
    validation_dataset: GraphSelfPlayDataset,
    metrics: Mapping[str, object],
    device: torch.device,
    use_amp: bool,
) -> None:
    state = _TRACKING_STATE.get()
    if state is None:
        return
    if state.training_dataset is None:
        raise RuntimeError(
            "Checkpoint candidate tracking has no training dataset."
        )

    state.epoch += 1
    for metric_key, variant_name, selection_metric in _CANDIDATE_SELECTORS:
        value = _finite_metric(metrics, metric_key)
        if value is None or value >= state.best_values[metric_key]:
            continue

        state.best_values[metric_key] = value
        _save_candidate(
            path=checkpoint_variant_path(
                state.request.output_path,
                variant_name,
            ),
            model=model,
            training_dataset=state.training_dataset,
            validation_dataset=validation_dataset,
            request=state.request,
            device=device,
            use_amp=use_amp,
            epoch=state.epoch,
            metric_name=selection_metric,
            selector_name=f"val_{metric_key}",
            selector_value=value,
            val_metrics=metrics,
        )
