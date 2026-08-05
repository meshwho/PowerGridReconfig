from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.models.graph_self_play_dataset import (
    GraphSelfPlayDataset,
)
from grid_topology_ai.training import checkpoint_candidates
from grid_topology_ai.training import graph_policy_value as training_api
from grid_topology_ai.training.graph_policy_value import TrainingRequest


def _request(
    tmp_path: Path,
    *,
    save_multiple_best: bool = True,
) -> TrainingRequest:
    return TrainingRequest(
        project_root=tmp_path,
        examples_csv=tmp_path / "train.csv",
        output_path=tmp_path / "candidate_checkpoint.pt",
        config=TrainingConfig(
            save_multiple_best=save_multiple_best,
        ),
    )


def test_candidate_tracker_saves_only_improving_objectives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        checkpoint_candidates,
        "_save_candidate",
        lambda **kwargs: saved.append(kwargs),
    )
    dataset = cast(GraphSelfPlayDataset, object())
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")

    with checkpoint_candidates.checkpoint_candidate_tracking(
        _request(tmp_path)
    ):
        checkpoint_candidates.register_training_dataset(dataset)
        checkpoint_candidates.record_validation_candidates(
            model=model,
            validation_dataset=dataset,
            metrics={
                "policy_loss": 0.50,
                "value_loss": 0.40,
                "value_calibration_error": 0.30,
            },
            device=device,
            use_amp=False,
        )
        checkpoint_candidates.record_validation_candidates(
            model=model,
            validation_dataset=dataset,
            metrics={
                "policy_loss": 0.55,
                "value_loss": 0.20,
                "value_calibration_error": 0.30,
            },
            device=device,
            use_amp=False,
        )

    assert [item["path"].name for item in saved] == [
        "candidate_checkpoint_best_policy_loss.pt",
        "candidate_checkpoint_best_value_loss.pt",
        "candidate_checkpoint_best_calibration.pt",
        "candidate_checkpoint_best_value_loss.pt",
    ]
    assert [item["epoch"] for item in saved] == [1, 1, 1, 2]
    assert [item["metric_name"] for item in saved] == [
        "validation_policy_loss",
        "validation_value_loss",
        "validation_value_calibration_error",
        "validation_value_loss",
    ]
    assert saved[-1]["selector_value"] == pytest.approx(0.20)


def test_candidate_tracker_is_disabled_without_multiple_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        checkpoint_candidates,
        "_save_candidate",
        lambda **kwargs: pytest.fail("candidate should not be saved"),
    )
    dataset = cast(GraphSelfPlayDataset, object())

    with checkpoint_candidates.checkpoint_candidate_tracking(
        _request(tmp_path, save_multiple_best=False)
    ):
        checkpoint_candidates.register_training_dataset(dataset)
        checkpoint_candidates.record_validation_candidates(
            model=torch.nn.Linear(1, 1),
            validation_dataset=dataset,
            metrics={
                "policy_loss": 0.10,
                "value_loss": 0.10,
                "value_calibration_error": 0.10,
            },
            device=torch.device("cpu"),
            use_amp=False,
        )


def test_validation_wrapper_forwards_metrics_to_candidate_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "loss": 0.50,
        "policy_loss": 0.20,
        "value_loss": 0.10,
        "value_calibration_error": 0.05,
    }

    def fake_evaluate(
        model,
        loader,
        value_loss_fn,
        device,
        use_amp,
        value_loss_weight,
    ) -> dict[str, float]:
        return expected

    monkeypatch.setattr(
        training_api,
        "_evaluate_one_epoch_diagnostics",
        fake_evaluate,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        training_api,
        "record_validation_candidates",
        lambda **kwargs: captured.update(kwargs),
    )

    dataset = object()
    loader = type("Loader", (), {"dataset": dataset})()
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")

    result = training_api.evaluate_one_epoch(
        model=model,
        loader=loader,
        value_loss_fn=torch.nn.MSELoss(),
        device=device,
        use_amp=False,
        value_loss_weight=1.0,
    )

    assert result is expected
    assert captured["model"] is model
    assert captured["validation_dataset"] is dataset
    assert captured["metrics"] is expected
    assert captured["device"] == device
    assert captured["use_amp"] is False
