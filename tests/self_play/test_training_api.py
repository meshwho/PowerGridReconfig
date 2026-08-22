from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.state.schema import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS
from grid_topology_ai.topology_actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    ActionSpaceConfig,
    build_branch_action_slots,
)
from grid_topology_ai.training import graph_policy_value as training_api
from grid_topology_ai.training.checkpoints import build_training_config_payload
from grid_topology_ai.training.graph_policy_value import (
    TrainingRequest,
    resolve_device,
    train_graph_policy_value_model,
)


class _Dataset:
    policy_layout = STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT
    num_bus_features = 4
    num_branch_features = 6
    action_layout_count = 2


class _Model(torch.nn.Module):
    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        type(self).created.append(kwargs)

    def to(self, device: torch.device):
        self.device = device
        return self


class _TinyTrainingDataset:
    def __init__(
        self,
        *,
        examples_csv: Path,
        normalize_features: bool = True,
        normalization_stats: dict[str, np.ndarray] | None = None,
        physics_config=None,
    ) -> None:
        self.examples_csv = Path(examples_csv)
        self.normalize_features = bool(normalize_features)
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
        self.topology_action_config = ActionSpaceConfig()
        self.action_layout = build_branch_action_slots((0, 1))
        self.policy_layout = STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT
        self.action_layout_count = 1
        self.num_bus_features = len(BUS_FEATURE_COLUMNS)
        self.num_branch_features = len(BRANCH_FEATURE_COLUMNS)
        self.examples = pd.DataFrame(
            {
                "scenario_id": [1, 2],
                "outcome_value_target": [0.25, -0.5],
            }
        )

        if normalization_stats is None:
            self._normalization = {
                "bus_feature_mean": np.zeros(
                    self.num_bus_features, dtype=np.float32
                ),
                "bus_feature_std": np.ones(
                    self.num_bus_features, dtype=np.float32
                ),
                "branch_feature_mean": np.zeros(
                    self.num_branch_features, dtype=np.float32
                ),
                "branch_feature_std": np.ones(
                    self.num_branch_features, dtype=np.float32
                ),
            }
        else:
            self._normalization = {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in normalization_stats.items()
            }

        self._samples = (
            self._sample(
                seed=101,
                scenario_id=1,
                target_policy=(0.0, 1.0, 0.0),
                target_value=0.25,
                edge_active=(True, True),
            ),
            self._sample(
                seed=202,
                scenario_id=2,
                target_policy=(0.5, 0.0, 0.5),
                target_value=-0.5,
                edge_active=(True, False),
            ),
        )

    def _sample(
        self,
        *,
        seed: int,
        scenario_id: int,
        target_policy: tuple[float, float, float],
        target_value: float,
        edge_active: tuple[bool, bool],
    ) -> dict[str, object]:
        generator = torch.Generator().manual_seed(seed)
        return {
            "bus_features": torch.randn(
                3,
                self.num_bus_features,
                generator=generator,
            ),
            "branch_features": torch.randn(
                2,
                self.num_branch_features,
                generator=generator,
            ),
            "edge_index": torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            "edge_active_mask": torch.tensor(
                edge_active,
                dtype=torch.bool,
            ),
            "action_mask": torch.tensor(
                [True, True, True],
                dtype=torch.bool,
            ),
            "target_policy": torch.tensor(
                target_policy,
                dtype=torch.float32,
            ),
            "target_value": torch.tensor(
                target_value,
                dtype=torch.float32,
            ),
            "scenario_id": scenario_id,
            "step": 0,
            "state_id": f"state-{scenario_id}",
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }

    def normalization_state_dict(self) -> dict[str, np.ndarray]:
        return {
            key: value.copy()
            for key, value in self._normalization.items()
        }


def _request(tmp_path: Path, **kwargs: Any) -> TrainingRequest:
    examples = tmp_path / "examples.csv"
    examples.write_text("scenario_id\n1\n", encoding="utf-8")
    values = {
        "project_root": tmp_path,
        "examples_csv": examples,
        "output_path": tmp_path / "model.pt",
        "config": TrainingConfig(epochs=1, batch_size=2),
    }
    values.update(kwargs)
    return TrainingRequest(**values)


def _resume_request(
    *,
    tmp_path: Path,
    examples: Path,
    output: Path,
    epochs: int,
    resume_checkpoint: Path | None = None,
) -> TrainingRequest:
    return TrainingRequest(
        project_root=tmp_path,
        examples_csv=examples,
        output_path=output,
        config=TrainingConfig(
            epochs=epochs,
            batch_size=2,
            learning_rate=1e-3,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            num_workers=0,
            device="cpu",
            no_tensorboard=True,
        ),
        resume_checkpoint=resume_checkpoint,
        save_best=True,
        seed=123,
    )


def _assert_same_model_state(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


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


def test_build_model_uses_graph_v2_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Model.created = []
    monkeypatch.setattr(training_api, "GraphPolicyValueNetV2", _Model)

    model = training_api._build_model(
        request=_request(
            tmp_path,
            config=TrainingConfig(
                hidden_dim=33,
                num_layers=4,
                dropout=0.2,
            ),
        ),
        dataset=_Dataset(),
        device=torch.device("cpu"),
    )

    assert isinstance(model, _Model)
    assert _Model.created == [
        {
            "num_bus_features": 4,
            "num_branch_features": 6,
            "hidden_dim": 33,
            "num_layers": 4,
            "dropout": 0.2,
        }
    ]


def test_checkpoint_training_config_has_no_architecture_knob(
    tmp_path: Path,
) -> None:
    payload = build_training_config_payload(
        _request(
            tmp_path,
            config=TrainingConfig(
                epochs=3,
                batch_size=5,
                learning_rate=0.01,
                save_multiple_best=True,
            ),
        )
    )

    assert "model_type" not in payload
    assert payload["epochs"] == 3
    assert payload["batch_size"] == 5
    assert payload["lr"] == 0.01
    assert payload["save_multiple_best"] is True


def test_cpu_training_resume_matches_continuous_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = tmp_path / "examples.csv"
    examples.write_text("scenario_id\n1\n2\n", encoding="utf-8")

    monkeypatch.setattr(
        training_api,
        "GraphSelfPlayDataset",
        _TinyTrainingDataset,
    )
    monkeypatch.setattr(
        training_api,
        "print_value_target_diagnostics",
        lambda diagnostics: None,
    )
    monkeypatch.setattr(
        training_api,
        "setup_live_logging",
        lambda *, request, output_path: (None, tmp_path / "metrics.csv"),
    )
    monkeypatch.setattr(
        training_api,
        "log_epoch_metrics",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        training_api,
        "evaluate_training_samples",
        lambda **kwargs: None,
    )

    continuous_output = tmp_path / "continuous.pt"
    train_graph_policy_value_model(
        _resume_request(
            tmp_path=tmp_path,
            examples=examples,
            output=continuous_output,
            epochs=2,
        )
    )

    split_output = tmp_path / "split.pt"
    train_graph_policy_value_model(
        _resume_request(
            tmp_path=tmp_path,
            examples=examples,
            output=split_output,
            epochs=1,
        )
    )
    split_resume = tmp_path / "split_resume.pt"
    first_progress = torch.load(
        split_resume,
        map_location="cpu",
        weights_only=False,
    )
    assert first_progress["completed_epoch"] == 1
    assert "optimizer_state_dict" in first_progress
    assert "rng_state" in first_progress
    assert "train_generator_state" in first_progress
    assert "best_model_state_dict" in first_progress

    train_graph_policy_value_model(
        _resume_request(
            tmp_path=tmp_path,
            examples=examples,
            output=split_output,
            epochs=2,
            resume_checkpoint=split_resume,
        )
    )

    continuous = torch.load(
        continuous_output,
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        split_output,
        map_location="cpu",
        weights_only=False,
    )
    _assert_same_model_state(
        continuous["model_state_dict"],
        resumed["model_state_dict"],
    )
    assert resumed["best_epoch"] == continuous["best_epoch"]
    assert resumed["best_metric"] == pytest.approx(
        continuous["best_metric"],
        rel=0.0,
        abs=0.0,
    )

    continuous_progress = torch.load(
        tmp_path / "continuous_resume.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_progress = torch.load(
        split_resume,
        map_location="cpu",
        weights_only=False,
    )
    assert continuous_progress["completed_epoch"] == 2
    assert resumed_progress["completed_epoch"] == 2
    _assert_same_model_state(
        continuous_progress["model_state_dict"],
        resumed_progress["model_state_dict"],
    )
