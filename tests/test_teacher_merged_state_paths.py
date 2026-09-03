from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai import cli
from grid_topology_ai.dataset import GraphSelfPlayDataset
from grid_topology_ai.self_play import example_validation


def test_merge_teacher_split_preserves_direct_state_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "teacher"
    monkeypatch.setattr(
        example_validation,
        "validate_examples_dataframe",
        lambda frame, *, source_path: None,
    )

    expected_paths: list[str] = []
    for index, difficulty in enumerate(("simple", "medium", "hard")):
        run_dir = output_dir / "train" / difficulty
        run_dir.mkdir(parents=True)
        state_id = f"state_{difficulty}"
        state_path = run_dir / "states" / f"{state_id}.npz"
        pd.DataFrame(
            {
                "scenario_id": [index],
                "step": [0],
                "state_id": [state_id],
                "state_path": [str(state_path)],
            }
        ).to_csv(run_dir / "examples.csv", index=False)
        expected_paths.append(str(state_path))

    merged_path = cli._merge_teacher_split_examples(output_dir, "train")
    merged = pd.read_csv(merged_path)

    assert merged["state_path"].tolist() == expected_paths


def test_graph_dataset_uses_state_path_directly(tmp_path: Path) -> None:
    dataset = GraphSelfPlayDataset.__new__(GraphSelfPlayDataset)
    state_path = tmp_path / "teacher" / "train" / "simple" / "states" / "state_1.npz"

    assert dataset._state_path(str(state_path)) == state_path
