from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.training.graph_policy_value import TrainingRequest
from scripts.self_play import train_graph_baseline as train_cli


def _capture_request(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    captured: list[TrainingRequest],
) -> None:
    def fake_train_graph_policy_value_model(request: TrainingRequest) -> Path:
        captured.append(request)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        return checkpoint_path

    monkeypatch.setattr(
        train_cli,
        "train_graph_policy_value_model",
        fake_train_graph_policy_value_model,
    )


def test_cli_builds_training_request_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TrainingRequest] = []
    checkpoint = tmp_path / "out" / "model.pt"
    _capture_request(monkeypatch, checkpoint, captured)

    examples_csv = tmp_path / "examples.csv"
    init_checkpoint = tmp_path / "init.pt"
    val_csv = tmp_path / "val.csv"
    metrics_csv = tmp_path / "metrics.csv"
    tensorboard_dir = tmp_path / "tb"

    assert train_cli.main(
        [
            str(examples_csv),
            "--output",
            str(checkpoint),
            "--init-checkpoint",
            str(init_checkpoint),
            "--val-examples-csv",
            str(val_csv),
            "--tensorboard-log-dir",
            str(tensorboard_dir),
            "--run-name",
            "unit-run",
            "--metrics-csv",
            str(metrics_csv),
            "--amp",
            "--no-normalize-features",
            "--save-best",
        ]
    ) == 0

    request = captured[0]
    assert request.examples_csv == examples_csv
    assert request.output_path == checkpoint
    assert request.init_checkpoint == init_checkpoint
    assert request.validation_examples_csv == val_csv
    assert request.tensorboard_log_dir == tensorboard_dir
    assert request.run_name == "unit-run"
    assert request.metrics_csv == metrics_csv
    assert request.use_amp is True
    assert request.normalize_features is False
    assert request.save_best is True


def test_cli_builds_training_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TrainingRequest] = []
    _capture_request(monkeypatch, tmp_path / "model.pt", captured)

    assert train_cli.main(
        [
            str(tmp_path / "examples.csv"),
            "--epochs",
            "7",
            "--lr",
            "0.02",
            "--hidden-dim",
            "64",
            "--num-layers",
            "5",
            "--dropout",
            "0.3",
            "--batch-size",
            "9",
            "--value-loss-weight",
            "1.5",
            "--value-huber-delta",
            "0.25",
            "--device",
            "cpu",
            "--num-workers",
            "2",
            "--model-type",
            "graph_v2",
            "--save-multiple-best",
            "--no-tensorboard",
        ]
    ) == 0

    config = captured[0].config
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
    assert config.model_type == "graph_v2"
    assert config.save_multiple_best is True
    assert config.no_tensorboard is True


def test_cli_prints_checkpoint_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[TrainingRequest] = []
    checkpoint = tmp_path / "model.pt"
    _capture_request(monkeypatch, checkpoint, captured)

    assert train_cli.main([str(tmp_path / "examples.csv")]) == 0

    assert capsys.readouterr().out.strip() == str(checkpoint)


def test_cli_help_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_cli.build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--epochs" in output
    assert "--save-multiple-best" in output
    assert "--val-examples-csv" in output
