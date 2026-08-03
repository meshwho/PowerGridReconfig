from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from grid_topology_ai.config import ReplayBufferConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play.iteration_split import (
    prepare_physical_iteration_split,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
    PhysicalLineage,
)
from grid_topology_ai.self_play.replay import RollingReplayBuffer


def _lineage(index: int) -> PhysicalLineage:
    return PhysicalLineage.build(
        base_case_id="case118",
        load_profile_id=f"load-{index}",
        contingency_family_id=[f"branch:{index}"],
    )


def _row(
    index: int,
    *,
    step: int = 0,
    scenario_id: int | None = None,
    replay_iteration: int = 1,
    difficulty: str = "medium",
    outcome: str = "solved",
) -> dict[str, object]:
    lineage = _lineage(index)
    return {
        "state_id": f"state-{index}-{scenario_id}-{step}",
        "episode_id": f"episode-{index}-{scenario_id or index}",
        "scenario_id": index if scenario_id is None else scenario_id,
        "step": step,
        "replay_iteration": replay_iteration,
        "difficulty_class": difficulty,
        "outcome_class": outcome,
        **lineage.as_dict(),
    }


def _paths(tmp_path: Path) -> SelfPlayPaths:
    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text("scenario_id\n1\n2\n", encoding="utf-8")
    return SelfPlayPaths(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        pool_transitions_csv=pool_csv,
        pool_raw_dir=tmp_path / "pool_raw",
        pool_metadata=tmp_path / "pool.json",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "eval_raw",
        final_test_csv=tmp_path / "test.csv",
        final_test_raw_dir=tmp_path / "test_raw",
        bootstrap_checkpoint=tmp_path / "bootstrap.pt",
        bootstrap_metrics=tmp_path / "bootstrap.json",
        best_checkpoint=tmp_path / "best.pt",
        best_metrics=tmp_path / "best.json",
    )


def _buffer(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> RollingReplayBuffer:
    buffer = RollingReplayBuffer(
        save_dir=tmp_path / "replay",
        config=ReplayBufferConfig(
            max_size=100,
            min_size_to_train=1,
            fresh_fraction=0.5,
            random_seed=17,
        ),
    )
    buffer.buffer = rows
    return buffer


def _prepare(
    *,
    replay: RollingReplayBuffer,
    paths: SelfPlayPaths,
    iteration: int,
    sampling_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    iter_dir = paths.iteration_dir(iteration)
    return prepare_physical_iteration_split(
        replay_buffer=replay,
        paths=paths,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        iteration=iteration,
        split_seed=17,
        sampling_seed=sampling_seed,
        validation_fraction=0.25,
        min_validation_lineages=1,
        n_examples=4,
        fresh_fraction=0.5,
        train_batch_path=iter_dir / "train_batch.csv",
        train_examples_path=iter_dir / "train_examples.csv",
        validation_examples_path=iter_dir / "validation_examples.csv",
        metadata_path=iter_dir / "train_validation_split.json",
    )


def test_iteration_split_samples_only_train_lineages(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    rows = [
        _row(index, step=step)
        for index in range(1, 5)
        for step in range(2)
    ]
    replay = _buffer(tmp_path, rows)

    batch_metadata, split_metadata = _prepare(
        replay=replay,
        paths=paths,
        iteration=1,
        sampling_seed=101,
    )

    train = pd.read_csv(paths.iteration_dir(1) / "train_examples.csv")
    validation = pd.read_csv(
        paths.iteration_dir(1) / "validation_examples.csv"
    )
    train_lineages = set(train[PHYSICAL_LINEAGE_FINGERPRINT_FIELD])
    validation_lineages = set(
        validation[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
    )
    manifest = json.loads(
        paths.physical_split_manifest.read_text(encoding="utf-8")
    )
    expected_train_lineages = {
        fingerprint
        for fingerprint, entry in manifest["assignments"].items()
        if entry["split"] == "train"
    }

    assert train_lineages.isdisjoint(validation_lineages)
    assert train_lineages <= expected_train_lineages
    assert validation_lineages.isdisjoint(expected_train_lineages)
    assert len(validation) == 2
    assert batch_metadata["n_examples"] == len(train)
    assert batch_metadata["eligible_physical_lineage_count"] == 3
    assert split_metadata["split_unit"] == "physical_lineage"
    assert split_metadata["validation_examples"] == 2
    assert split_metadata["persistent_manifest_path"] == str(
        paths.physical_split_manifest
    )
    assert paths.physical_split_manifest.is_file()


def test_iteration_split_reuses_assignments_across_iterations(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    replay = _buffer(
        tmp_path,
        [
            _row(index, step=step)
            for index in range(1, 5)
            for step in range(2)
        ],
    )
    _prepare(
        replay=replay,
        paths=paths,
        iteration=1,
        sampling_seed=101,
    )
    first = json.loads(
        paths.physical_split_manifest.read_text(encoding="utf-8")
    )
    original = {
        fingerprint: entry["split"]
        for fingerprint, entry in first["assignments"].items()
    }

    replay.buffer.extend(
        [
            _row(1, scenario_id=101, replay_iteration=2),
            _row(5, replay_iteration=2),
            _row(6, replay_iteration=2),
        ]
    )
    _, metadata = _prepare(
        replay=replay,
        paths=paths,
        iteration=2,
        sampling_seed=202,
    )
    second = json.loads(
        paths.physical_split_manifest.read_text(encoding="utf-8")
    )

    for fingerprint, split in original.items():
        assert second["assignments"][fingerprint]["split"] == split
    assert second["assignments"][_lineage(1).fingerprint][
        "scenario_ids"
    ] == [1, 101]
    assert metadata["new_assignments_this_iteration"] == 2
    assert metadata["split_seed"] == 17
    assert metadata["sampling_seed"] == 202


def test_validation_uses_all_replay_rows_for_its_lineage(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    replay = _buffer(
        tmp_path,
        [
            _row(index, step=step)
            for index in range(1, 5)
            for step in range(5)
        ],
    )

    _, metadata = _prepare(
        replay=replay,
        paths=paths,
        iteration=1,
        sampling_seed=31,
    )
    validation = pd.read_csv(
        paths.iteration_dir(1) / "validation_examples.csv"
    )

    assert metadata["validation_lineages"] == 1
    assert len(validation) == 5
    assert validation[PHYSICAL_LINEAGE_FINGERPRINT_FIELD].nunique() == 1


def test_iteration_split_rejects_missing_lineage_metadata(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    replay = _buffer(
        tmp_path,
        [
            {
                "state_id": "state-1",
                "episode_id": "episode-1",
                "scenario_id": 1,
                "replay_iteration": 1,
                "outcome_class": "solved",
            }
        ],
    )

    with pytest.raises(ValueError, match="physical lineage"):
        _prepare(
            replay=replay,
            paths=paths,
            iteration=1,
            sampling_seed=7,
        )
