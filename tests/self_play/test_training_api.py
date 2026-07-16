from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.training import graph_policy_value as training_api
from grid_topology_ai.training.checkpoints import build_training_config_payload
from grid_topology_ai.training.graph_policy_value import (
    TrainingRequest,
    resolve_device,
    train_graph_policy_value_model,
)


class _FakeDataset:
    created: list[dict[str, Any]] = []

    def __init__(
        self,
        *,
        examples_csv: str | Path,
        normalize_features: bool,
        normalization_stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.examples_csv = Path(examples_csv)
        self.normalize_features = normalize_features
        self.normalization_stats = normalization_stats
        if normalization_stats is None:
            self.bus_feature_mean = np.array([1.0], dtype=np.float32)
            self.bus_feature_std = np.array([2.0], dtype=np.float32)
            self.branch_feature_mean = np.array([3.0], dtype=np.float32)
            self.branch_feature_std = np.array([4.0], dtype=np.float32)
        else:
            self.bus_feature_mean = np.array(normalization_stats["bus_feature_mean"], dtype=np.float32, copy=True)
            self.bus_feature_std = np.array(normalization_stats["bus_feature_std"], dtype=np.float32, copy=True)
            self.branch_feature_mean = np.array(normalization_stats["branch_feature_mean"], dtype=np.float32, copy=True)
            self.branch_feature_std = np.array(normalization_stats["branch_feature_std"], dtype=np.float32, copy=True)
        self.num_bus_features = 1
        self.num_branch_features = 1
        self.num_buses = 1
        self.num_branches = 1
        self.num_actions = 2
        self.examples = pd.DataFrame(
            {
                "state_path": ["state.npz"],
                "scenario_id": [1],
                "outcome_value_target": [0.0],
            }
        )
        type(self).created.append(
            {
                "examples_csv": self.examples_csv,
                "normalize_features": normalize_features,
                "normalization_stats": normalization_stats,
            }
        )

    def __len__(self) -> int:
        return 4

    def normalization_state_dict(self) -> dict[str, np.ndarray]:
        return {
            "bus_feature_mean": self.bus_feature_mean,
            "bus_feature_std": self.bus_feature_std,
            "branch_feature_mean": self.branch_feature_mean,
            "branch_feature_std": self.branch_feature_std,
        }


class _FakeModel(torch.nn.Module):
    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.model_type = "graph_policy_value_net_v2"
        type(self).created.append(kwargs)

    def to(self, device: torch.device) -> "_FakeModel":
        self.device = device
        return self

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((1, 2)), torch.zeros((1, 1))


def _request(tmp_path: Path, **kwargs: Any) -> TrainingRequest:
    examples_csv = tmp_path / "examples.csv"
    examples_csv.write_text("scenario_id,state_path,outcome_value_target\n1,state.npz,0\n")
    values = {
        "project_root": tmp_path,
        "examples_csv": examples_csv,
        "output_path": tmp_path / "model.pt",
        "config": TrainingConfig(epochs=1, batch_size=2),
    }
    values.update(kwargs)
    return TrainingRequest(**values)


def _patch_light_training(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    _FakeDataset.created = []
    _FakeModel.created = []
    monkeypatch.setattr(training_api, "GraphSelfPlayDataset", _FakeDataset)
    monkeypatch.setattr(training_api, "GraphPolicyValueNet", _FakeModel)
    monkeypatch.setattr(training_api, "GraphPolicyValueNetV2", _FakeModel)
    monkeypatch.setattr(training_api, "evaluate_training_samples", lambda **kwargs: None)
    monkeypatch.setattr(
        training_api,
        "setup_live_logging",
        lambda **kwargs: (None, captured["metrics_csv"]),
    )
    monkeypatch.setattr(training_api, "log_epoch_metrics", lambda **kwargs: None)

    def fake_validate_no_scenario_overlap(**kwargs: Any) -> None:
        captured["overlap_checked"] = kwargs

    monkeypatch.setattr(
        training_api,
        "validate_no_scenario_overlap",
        fake_validate_no_scenario_overlap,
    )

    def fake_load_initial_checkpoint_into_model(**kwargs: Any) -> None:
        captured["init_checkpoint"] = kwargs["checkpoint_path"]

    monkeypatch.setattr(
        training_api,
        "load_initial_checkpoint_into_model",
        fake_load_initial_checkpoint_into_model,
    )

    monkeypatch.setattr(
        training_api,
        "load_checkpoint_payload",
        lambda *args, **kwargs: {
            "bus_feature_mean": np.array([1.0], dtype=np.float32),
            "bus_feature_std": np.array([2.0], dtype=np.float32),
            "branch_feature_mean": np.array([3.0], dtype=np.float32),
            "branch_feature_std": np.array([4.0], dtype=np.float32),
            "model_state_dict": {},
        },
    )

    def fake_train_one_epoch(**kwargs: Any) -> tuple[float, float, float]:
        captured["train_kwargs"] = kwargs
        captured["learning_rate"] = kwargs["optimizer"].param_groups[0]["lr"]
        captured["huber_delta"] = kwargs["value_loss_fn"].delta
        captured["batch_size"] = kwargs["loader"].batch_size
        captured["num_workers"] = kwargs["loader"].num_workers
        return 1.0, 0.5, 0.25

    monkeypatch.setattr(training_api, "train_one_epoch", fake_train_one_epoch)

    monkeypatch.setattr(
        training_api,
        "evaluate_one_epoch",
        lambda **kwargs: {
            "loss": 0.5,
            "policy_loss": 0.2,
            "value_loss": 0.1,
            "top1": 0.3,
            "top3": 0.4,
            "top5": 0.5,
            "stop_acc": 0.6,
            "switch_acc": 0.7,
            "examples": 4.0,
        },
    )


def test_training_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.use_amp = True  # type: ignore[misc]

    assert not hasattr(request, "__dict__")


def test_missing_examples_csv_raises(tmp_path: Path) -> None:
    request = TrainingRequest(
        project_root=tmp_path,
        examples_csv=tmp_path / "missing.csv",
        output_path=tmp_path / "model.pt",
        config=TrainingConfig(),
    )

    with pytest.raises(FileNotFoundError):
        train_graph_policy_value_model(request)


def test_resolve_device_cpu_auto_and_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert str(resolve_device("cpu")) == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"

    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_api_uses_training_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    config = TrainingConfig(
        epochs=1,
        batch_size=7,
        learning_rate=0.0123,
        value_loss_weight=2.5,
        value_huber_delta=0.75,
        num_workers=0,
        device="cpu",
        model_type="graph_v2",
        hidden_dim=33,
        num_layers=4,
        dropout=0.2,
    )

    result = train_graph_policy_value_model(_request(tmp_path, config=config))

    assert result == tmp_path / "model.pt"
    assert captured["learning_rate"] == 0.0123
    assert captured["huber_delta"] == 0.75
    assert captured["train_kwargs"]["value_loss_weight"] == 2.5
    assert captured["batch_size"] == 4
    assert captured["num_workers"] == 0
    assert _FakeModel.created[0]["hidden_dim"] == 33
    assert _FakeModel.created[0]["num_layers"] == 4
    assert _FakeModel.created[0]["dropout"] == 0.2


def test_validation_uses_train_normalization_and_overlap_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("scenario_id,state_path,outcome_value_target\n2,state.npz,0\n")

    train_graph_policy_value_model(
        _request(tmp_path, validation_examples_csv=val_csv)
    )

    assert _FakeDataset.created[1]["examples_csv"] == val_csv
    stats = _FakeDataset.created[1]["normalization_stats"]
    assert stats is not None
    np.testing.assert_array_equal(
        stats["bus_feature_mean"],
        np.array([1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        stats["bus_feature_std"],
        np.array([2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        stats["branch_feature_mean"],
        np.array([3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        stats["branch_feature_std"],
        np.array([4.0], dtype=np.float32),
    )
    assert captured["overlap_checked"]["val_dataset"] is not None


def test_api_loads_init_checkpoint_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")

    train_graph_policy_value_model(
        _request(tmp_path, init_checkpoint=init_checkpoint)
    )

    assert captured["init_checkpoint"] == init_checkpoint


def test_build_model_uses_explicit_architecture_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeV1(_FakeModel):
        pass

    class FakeV2(_FakeModel):
        pass

    monkeypatch.setattr(training_api, "GraphPolicyValueNet", FakeV1)
    monkeypatch.setattr(training_api, "GraphPolicyValueNetV2", FakeV2)
    dataset = _FakeDataset(
        examples_csv=tmp_path / "examples.csv",
        normalize_features=True,
    )

    model_v1 = training_api._build_model(
        request=_request(
            tmp_path,
            config=TrainingConfig(model_type="graph_v1"),
        ),
        dataset=dataset,
        device=torch.device("cpu"),
    )
    model_v2 = training_api._build_model(
        request=_request(
            tmp_path,
            config=TrainingConfig(model_type="graph_v2"),
        ),
        dataset=dataset,
        device=torch.device("cpu"),
    )

    assert isinstance(model_v1, FakeV1)
    assert isinstance(model_v2, FakeV2)


def test_checkpoint_training_config_uses_legacy_keys(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        init_checkpoint=tmp_path / "init.pt",
        validation_examples_csv=tmp_path / "val.csv",
        use_amp=True,
        normalize_features=False,
        save_best=True,
        tensorboard_log_dir=tmp_path / "tb",
        run_name="run",
        metrics_csv=tmp_path / "metrics.csv",
        config=TrainingConfig(
            epochs=3,
            batch_size=5,
            learning_rate=0.01,
            save_multiple_best=True,
            no_tensorboard=True,
        ),
    )

    payload = build_training_config_payload(request)

    assert list(payload) == [
        "examples_csv",
        "seed",
        "epochs",
        "lr",
        "hidden_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "value_loss_weight",
        "value_huber_delta",
        "device",
        "amp",
        "num_workers",
        "no_normalize_features",
        "output",
        "init_checkpoint",
        "val_examples_csv",
        "save_best",
        "tensorboard_log_dir",
        "run_name",
        "no_tensorboard",
        "metrics_csv",
        "model_type",
        "save_multiple_best",
    ]
    assert "examples_per_iteration" not in payload


def test_package_modules_do_not_use_argparse_namespace() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("grid_topology_ai/training/graph_policy_value.py"),
            Path("grid_topology_ai/training/checkpoints.py"),
            Path("grid_topology_ai/training/metrics.py"),
        ]
    )

    assert "argparse.Namespace" not in text
    assert "scripts." not in text


def _sentinel_stats() -> dict[str, np.ndarray]:
    return {
        "bus_feature_mean": np.array([10.0], dtype=np.float32),
        "bus_feature_std": np.array([2.0], dtype=np.float32),
        "branch_feature_mean": np.array([100.0], dtype=np.float32),
        "branch_feature_std": np.array([10.0], dtype=np.float32),
    }


def test_fine_tuning_reuses_init_checkpoint_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    sentinel = _sentinel_stats()
    monkeypatch.setattr(training_api, "load_checkpoint_payload", lambda *a, **k: {**sentinel, "model_state_dict": {}})
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("scenario_id,state_path,outcome_value_target\n2,state.npz,0\n")

    train_graph_policy_value_model(
        _request(tmp_path, init_checkpoint=init_checkpoint, validation_examples_csv=val_csv)
    )

    assert _FakeDataset.created[0]["normalization_stats"] is not None
    assert _FakeDataset.created[1]["normalization_stats"] is not None
    for created in _FakeDataset.created[:2]:
        for key, expected in sentinel.items():
            np.testing.assert_array_equal(created["normalization_stats"][key], expected)
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    assert checkpoint["normalization_source"] == "init_checkpoint"
    assert checkpoint["normalization_frozen_from_init_checkpoint"] is True
    assert checkpoint["normalization_source_checkpoint"] == str(init_checkpoint)


def test_fine_tuning_does_not_recompute_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"; init_checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(training_api, "load_checkpoint_payload", lambda *a, **k: {**_sentinel_stats(), "model_state_dict": {}})
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("normalization was recomputed")
    monkeypatch.setattr(_FakeDataset, "_compute_feature_statistics", forbidden, raising=False)
    train_graph_policy_value_model(_request(tmp_path, init_checkpoint=init_checkpoint))


def test_training_from_scratch_computes_and_saves_training_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    train_graph_policy_value_model(_request(tmp_path))
    assert _FakeDataset.created[0]["normalization_stats"] is None
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    assert checkpoint["normalization_source"] == "training_dataset"
    assert checkpoint["normalization_frozen_from_init_checkpoint"] is False
    assert checkpoint["normalization_source_checkpoint"] is None


def test_fine_tuning_rejects_checkpoint_without_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"; init_checkpoint.write_bytes(b"checkpoint")
    payload = _sentinel_stats(); del payload["branch_feature_std"]
    monkeypatch.setattr(training_api, "load_checkpoint_payload", lambda *a, **k: payload)
    with pytest.raises(ValueError, match="branch_feature_std"):
        train_graph_policy_value_model(_request(tmp_path, init_checkpoint=init_checkpoint))
    assert not (tmp_path / "model.pt").exists()
    assert "train_kwargs" not in captured


def test_fine_tuning_rejects_normalization_feature_dimension_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"; init_checkpoint.write_bytes(b"checkpoint")
    bad = _sentinel_stats(); bad["bus_feature_mean"] = np.array([1.0, 2.0], dtype=np.float32); bad["bus_feature_std"] = np.array([1.0, 2.0], dtype=np.float32)
    monkeypatch.setattr(training_api, "load_checkpoint_payload", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="dimension mismatch.*bus_feature_mean"):
        train_graph_policy_value_model(_request(tmp_path, init_checkpoint=init_checkpoint))
    assert "train_kwargs" not in captured


def test_fine_tuning_loads_init_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"metrics_csv": tmp_path / "metrics.csv"}
    _patch_light_training(monkeypatch, captured)
    init_checkpoint = tmp_path / "init.pt"; init_checkpoint.write_bytes(b"checkpoint")
    calls = {"count": 0}
    def fake_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {**_sentinel_stats(), "model_state_dict": {}}
    monkeypatch.setattr(training_api, "load_checkpoint_payload", fake_load)
    train_graph_policy_value_model(_request(tmp_path, init_checkpoint=init_checkpoint))
    assert calls["count"] == 1
