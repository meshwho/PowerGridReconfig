from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.training.checkpoints import save_checkpoint_now
from grid_topology_ai.training.graph_policy_value import TrainingRequest
from tests.self_play.test_training_api import _FakeDataset, _FakeModel


def _request(tmp_path: Path) -> TrainingRequest:
    examples_csv = tmp_path / "examples.csv"
    examples_csv.write_text("scenario_id,state_path,outcome_value_target\n1,state.npz,0\n")
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
    dataset = _FakeDataset(examples_csv=tmp_path / "examples.csv", normalize_features=True)
    model = _FakeModel()

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
    dataset = _FakeDataset(examples_csv=tmp_path / "examples.csv", normalize_features=True)
    model = _FakeModel()

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
