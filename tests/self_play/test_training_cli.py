from __future__ import annotations

from pathlib import Path

import pytest

import grid_topology_ai.cli as light_cli
import grid_topology_ai.training as training_runtime
from grid_topology_ai.training.graph_policy_value import TrainingRequest


def _capture_request(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    captured: list[TrainingRequest],
) -> None:
    def fake_train(request: TrainingRequest) -> Path:
        captured.append(request)
        return checkpoint_path

    monkeypatch.setattr(
        training_runtime,
        "train_graph_policy_value_model",
        fake_train,
    )


def test_cli_builds_graph_v2_training_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TrainingRequest] = []
    checkpoint = tmp_path / "model.pt"
    _capture_request(monkeypatch, checkpoint, captured)

    examples = tmp_path / "examples.csv"
    val = tmp_path / "val.csv"
    init = tmp_path / "init.pt"
    metrics = tmp_path / "metrics.csv"

    assert light_cli.main(
        [
            "train",
            str(examples),
            "--output", str(checkpoint),
            "--init-checkpoint", str(init),
            "--validation", str(val),
            "--metrics-csv", str(metrics),
            "--epochs", "7",
            "--lr", "0.02",
            "--hidden-dim", "64",
            "--num-layers", "5",
            "--dropout", "0.3",
            "--batch-size", "9",
            "--value-loss-weight", "1.5",
            "--value-huber-delta", "0.25",
            "--device", "cpu",
            "--workers", "2",
            "--amp",
            "--save-best",
            "--no-tensorboard",
        ]
    ) == 0

    request = captured[0]
    assert request.examples_csv == examples
    assert request.output_path == checkpoint
    assert request.init_checkpoint == init
    assert request.resume_checkpoint is None
    assert request.validation_examples_csv == val
    assert request.metrics_csv == metrics
    assert request.use_amp is True
    assert request.save_best is True

    config = request.config
    assert not hasattr(config, "model_type")
    assert config.epochs == 7
    assert config.learning_rate == 0.02
    assert config.hidden_dim == 64
    assert config.num_layers == 5
    assert config.dropout == 0.3
    assert config.batch_size == 9
    assert config.value_loss_weight == 1.5
    assert config.value_huber_delta == 0.25
    assert config.device == "cpu"
    assert config.num_workers == 2
    assert config.no_tensorboard is True


def test_cli_wires_resume_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TrainingRequest] = []
    checkpoint = tmp_path / "model.pt"
    resume = tmp_path / "model_resume.pt"
    examples = tmp_path / "examples.csv"
    _capture_request(monkeypatch, checkpoint, captured)

    assert light_cli.main(
        [
            "train",
            str(examples),
            "--output", str(checkpoint),
            "--resume-checkpoint", str(resume),
            "--epochs", "9",
            "--device", "cpu",
            "--no-tensorboard",
        ]
    ) == 0

    request = captured[0]
    assert request.init_checkpoint is None
    assert request.resume_checkpoint == resume
    assert request.config.epochs == 9


def test_cli_no_longer_exposes_model_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        light_cli.main(["train", "--help"])
    assert exc.value.code == 0

    help_text = capsys.readouterr().out
    assert "--model-type" not in help_text
    assert "--resume-checkpoint" in help_text
