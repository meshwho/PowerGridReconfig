from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai import cli
from grid_topology_ai.dataset import GraphSelfPlayDataset
from grid_topology_ai.self_play import example_validation
from grid_topology_ai.self_play.example_validation import resolve_example_state_path


def test_merge_teacher_split_writes_relative_state_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "teacher"
    monkeypatch.setattr(
        example_validation,
        "validate_examples_dataframe",
        lambda frame, *, source_path: None,
    )

    for index, difficulty in enumerate(("simple", "medium", "hard")):
        run_dir = output_dir / "train" / difficulty
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "scenario_id": [index],
                "step": [0],
                "state_id": [f"state_{difficulty}"],
            }
        ).to_csv(run_dir / "examples.csv", index=False)

    merged_path = cli._merge_teacher_split_examples(output_dir, "train")
    merged = pd.read_csv(merged_path)

    assert merged["state_path"].tolist() == [
        "train/simple/states/state_simple.npz",
        "train/medium/states/state_medium.npz",
        "train/hard/states/state_hard.npz",
    ]


def test_state_path_resolution_keeps_legacy_and_supports_merged_paths(
    tmp_path: Path,
) -> None:
    merged_csv = tmp_path / "teacher" / "examples_train.csv"
    merged_state = resolve_example_state_path(
        merged_csv,
        "state_1",
        "train/simple/states/state_1.npz",
    )
    assert merged_state == (
        merged_csv.parent / "train" / "simple" / "states" / "state_1.npz"
    )

    run_csv = tmp_path / "teacher" / "train" / "simple" / "examples.csv"
    assert resolve_example_state_path(run_csv, "state_1") == (
        run_csv.parent / "states" / "state_1.npz"
    )

    with pytest.raises(ValueError, match="safe relative path"):
        resolve_example_state_path(
            merged_csv,
            "state_1",
            "../states/state_1.npz",
        )


def test_graph_dataset_uses_row_state_path(tmp_path: Path) -> None:
    dataset = GraphSelfPlayDataset.__new__(GraphSelfPlayDataset)
    dataset.examples_csv = tmp_path / "teacher" / "examples_train.csv"
    row = pd.Series(
        {
            "state_id": "state_1",
            "state_path": "train/simple/states/state_1.npz",
        }
    )

    assert dataset._state_path(row) == (
        dataset.examples_csv.parent
        / "train"
        / "simple"
        / "states"
        / "state_1.npz"
    )
