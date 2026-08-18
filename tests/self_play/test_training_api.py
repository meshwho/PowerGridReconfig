from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.topology_actions import STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT
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


def test_checkpoint_training_config_records_graph_v2(tmp_path: Path) -> None:
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

    assert payload["model_type"] == "graph_v2"
    assert payload["epochs"] == 3
    assert payload["batch_size"] == 5
    assert payload["lr"] == 0.01
    assert payload["save_multiple_best"] is True
