from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from grid_topology_ai.config.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import checkpoints


def _candidate(
    path: Path,
    *,
    loss: float,
    policy_loss: float,
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": checkpoints.sha256_file(path),
        "payload": {},
        "val_metrics": {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": 1.0,
            "value_calibration_error": 1.0,
        },
        "training_selector": "validation_loss",
        "saved_epoch": 1,
        "ranking_sources": [],
    }


def test_tuning_arena_minimizes_configured_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "candidate_checkpoint.pt"
    alternative = tmp_path / "candidate_checkpoint_best_policy_loss.pt"
    canonical.write_bytes(b"canonical")
    alternative.write_bytes(b"alternative")

    tuning_csv = tmp_path / "tuning.csv"
    pd.DataFrame({"scenario_id": [1]}).to_csv(tuning_csv, index=False)
    tuning_raw_dir = tmp_path / "raw"
    tuning_raw_dir.mkdir()

    monkeypatch.setattr(
        checkpoints,
        "_validate_tuning_independence",
        lambda **kwargs: (1,),
    )
    monkeypatch.setattr(
        checkpoints,
        "_load_candidates",
        lambda **kwargs: [
            _candidate(canonical, loss=0.1, policy_loss=0.9),
            _candidate(alternative, loss=0.9, policy_loss=0.1),
        ],
    )

    def fake_annotate(**kwargs: Any) -> None:
        Path(kwargs["destination"]).write_bytes(
            Path(kwargs["source"]).read_bytes()
        )

    monkeypatch.setattr(
        checkpoints,
        "_annotate_selected_checkpoint",
        fake_annotate,
    )

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        content = Path(kwargs["checkpoint"]).read_bytes()
        ungated = 0.4 if content == b"canonical" else 0.1
        constrained = 0.05 if content == b"canonical" else 0.9
        return {
            "primary_policy_mode": "ungated",
            "failed_scenario_rate_requested": ungated,
            "failed_scenarios": 0,
            "mode_metrics": {
                "ungated": {
                    "failed_scenario_rate_requested": ungated,
                    "failed_scenarios": 0,
                },
                "constrained": {
                    "failed_scenario_rate_requested": constrained,
                    "failed_scenarios": 0,
                },
            },
        }

    config = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        max_candidates=2,
        metric="failed_scenario_rate_requested",
        metric_direction="minimize",
    )
    result = checkpoints.select_checkpoint_in_tuning_arena(
        canonical_checkpoint=canonical,
        project_root=tmp_path,
        output_dir=tmp_path / "selection",
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        tuning_csv=tuning_csv,
        tuning_raw_dir=tuning_raw_dir,
        excluded_csvs={},
        evaluate=fake_evaluate,
    )

    assert result.selected_source == alternative
    assert result.metric_value == pytest.approx(0.1)
    assert canonical.read_bytes() == b"alternative"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["policy_mode"] == "ungated"
    assert report["metric_direction"] == "minimize"
    assert report["selected_source_checkpoint"] == str(alternative)
