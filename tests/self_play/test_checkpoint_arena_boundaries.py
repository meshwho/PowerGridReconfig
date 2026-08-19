from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.config import TrainingConfig
from grid_topology_ai.config.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import (
    checkpoint_arena,
    checkpoint_provenance,
    stages,
)


@pytest.mark.parametrize(
    "value",
    [1, 0, "true", "false", None],
)
def test_checkpoint_selection_enabled_requires_boolean(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        CheckpointSelectionConfig.from_mapping({"enabled": value})


def test_enabled_arena_requires_capacity_for_comparison() -> None:
    with pytest.raises(ValueError, match="max_candidates >= 2"):
        CheckpointSelectionConfig(
            enabled=True,
            tuning_csv=Path("tuning.csv"),
            tuning_raw_dir=Path("tuning_raw"),
            max_candidates=1,
        )


def test_tuning_arena_rejects_scenario_id_overlap(
    tmp_path: Path,
) -> None:
    tuning = tmp_path / "tuning.csv"
    pool = tmp_path / "pool.csv"
    evaluation = tmp_path / "evaluation.csv"
    final_test = tmp_path / "final_test.csv"

    pd.DataFrame(
        {
            "scenario_id": [1, 2],
            "source": ["tuning-a", "tuning-b"],
        }
    ).to_csv(tuning, index=False)
    pd.DataFrame(
        {
            "scenario_id": [2, 101],
            "source": ["pool-overlap", "pool-only"],
        }
    ).to_csv(pool, index=False)
    pd.DataFrame({"scenario_id": [201]}).to_csv(
        evaluation,
        index=False,
    )
    pd.DataFrame({"scenario_id": [301]}).to_csv(
        final_test,
        index=False,
    )

    with pytest.raises(ValueError, match="scenario-ID leakage"):
        checkpoint_arena._validate_tuning_independence(
            tuning_csv=tuning,
            excluded_csvs={
                "self-play pool": pool,
                "evaluation set": evaluation,
                "final test set": final_test,
            },
        )


def test_enabled_arena_rejects_training_request_without_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run" / "iter_001"
    output_dir.mkdir(parents=True)
    selection = CheckpointSelectionConfig(
        enabled=True,
        tuning_csv=Path("tuning.csv"),
        tuning_raw_dir=Path("tuning_raw"),
    )
    self_play_config = SimpleNamespace(
        checkpoint_selection=selection,
    )
    training_started = False

    def fake_train(_request) -> Path:
        nonlocal training_started
        training_started = True
        return output_dir / "candidate_checkpoint.pt"

    monkeypatch.setattr(
        stages,
        "_resolved_self_play_config",
        lambda path: self_play_config,
    )
    monkeypatch.setattr(
        stages,
        "train_graph_policy_value_model",
        fake_train,
    )

    with pytest.raises(
        RuntimeError,
        match="save_multiple_best=true",
    ):
        stages.run_train(
            project_root=tmp_path,
            examples_csv=tmp_path / "train.csv",
            validation_examples_csv=tmp_path / "validation.csv",
            init_checkpoint=tmp_path / "parent.pt",
            output_dir=output_dir,
            config=TrainingConfig(save_multiple_best=False),
            physics_config=DEFAULT_PHYSICS_CONFIG,
            iteration=1,
            seed=43,
        )

    assert training_started is False


def test_metadata_checkpoint_selection_flag_requires_boolean() -> None:
    metadata = {
        "config": {
            "checkpoint_selection": {
                "enabled": 1,
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="enabled must be a boolean",
    ):
        checkpoint_provenance._checkpoint_selection_required(metadata)
