from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.training.checkpoints import make_checkpoint, save_checkpoint_now
from grid_topology_ai.training.graph_policy_value import TrainingRequest
from tests.topology_contract_helpers import fake_dataset_topology_fields


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def _dataset(tmp_path: Path) -> MagicMock:
    dataset = MagicMock()
    dataset.__len__.return_value = 1
    dataset.examples_csv = tmp_path / "examples.csv"
    dataset.examples = pd.DataFrame(
        {
            "state_path": ["state.npz"],
            "scenario_id": [1],
            "outcome_value_target": [0.0],
        }
    )
    dataset.physics_config = DEFAULT_PHYSICS_CONFIG
    dataset.num_bus_features = 1
    dataset.num_branch_features = 1
    dataset.num_branches = 1
    for name, value in fake_dataset_topology_fields(1).items():
        setattr(dataset, name, value)
    dataset.normalization_state_dict.return_value = {
        "bus_feature_mean": np.array([1.0], dtype=np.float32),
        "bus_feature_std": np.array([2.0], dtype=np.float32),
        "branch_feature_mean": np.array([3.0], dtype=np.float32),
        "branch_feature_std": np.array([4.0], dtype=np.float32),
    }
    return dataset


def _request(tmp_path: Path) -> TrainingRequest:
    examples_csv = tmp_path / "examples.csv"
    examples_csv.write_text(
        "scenario_id,state_path,outcome_value_target\n1,state.npz,0\n",
        encoding="utf-8",
    )
    return TrainingRequest(
        project_root=tmp_path,
        examples_csv=examples_csv,
        output_path=tmp_path / "model.pt",
        config=TrainingConfig(),
    )


@pytest.mark.parametrize(
    ("selector_name", "expected_metric"),
    [
        ("val_loss", "validation_loss"),
        ("val_top1", "validation_top1"),
        ("val_top5", "validation_top5"),
        ("val_switch", "validation_switch_accuracy"),
        ("policy_selection_score", "policy_selection_score"),
        ("last_epoch", "last_epoch"),
    ],
)
def test_checkpoint_variant_records_exact_selector_metric(
    tmp_path: Path,
    selector_name: str,
    expected_metric: str,
) -> None:
    path = tmp_path / f"{selector_name}.pt"
    dataset = _dataset(tmp_path)
    model = _Model()

    save_checkpoint_now(
        path=path,
        model=model,
        dataset=dataset,
        request=_request(tmp_path),
        device=torch.device("cpu"),
        use_amp=False,
        epoch=2,
        selector_name=selector_name,
        selector_value=0.25,
        val_metrics={"loss": 0.25},
        validation_dataset=dataset,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["selector_name"] == selector_name
    assert checkpoint["selector_value"] == 0.25
    assert checkpoint["saved_epoch"] == 2
    assert checkpoint["checkpoint_selection_metric"] == expected_metric


def test_unknown_checkpoint_selector_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    model = _Model()

    with pytest.raises(ValueError, match="Unknown checkpoint selector"):
        save_checkpoint_now(
            path=tmp_path / "bad.pt",
            model=model,
            dataset=dataset,
            request=_request(tmp_path),
            device=torch.device("cpu"),
            use_amp=False,
            epoch=1,
            selector_name="mystery",
            selector_value=1.0,
            val_metrics=None,
            validation_dataset=dataset,
        )


def test_make_checkpoint_model_state_dict_is_parameter_snapshot(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    model = _Model()
    with torch.no_grad():
        model.weight.fill_(2.0)

    checkpoint = make_checkpoint(
        model=model,
        dataset=dataset,
        request=_request(tmp_path),
        device=torch.device("cpu"),
        use_amp=False,
        validation_dataset=dataset,
    )

    with torch.no_grad():
        model.weight.fill_(9.0)

    assert float(checkpoint["model_state_dict"]["weight"]) == 2.0
