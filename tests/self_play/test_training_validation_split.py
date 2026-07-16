from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.self_play.stages import split_examples_by_scenario


def _write_examples(path: Path) -> None:
    pd.DataFrame(
        [
            {"scenario_id": 1, "state_id": "s1a", "step": 0},
            {"scenario_id": 1, "state_id": "s1b", "step": 1},
            {"scenario_id": 2, "state_id": "s2a", "step": 0},
            {"scenario_id": 3, "state_id": "s3a", "step": 0},
            {"scenario_id": 3, "state_id": "s3b", "step": 1},
            {"scenario_id": 4, "state_id": "s4a", "step": 0},
        ]
    ).to_csv(path, index=False)


def _split(tmp_path: Path, *, seed: int = 11, min_validation_scenarios: int = 1):
    source = tmp_path / "train_batch.csv"
    _write_examples(source)
    return split_examples_by_scenario(
        examples_csv=source,
        train_output_csv=tmp_path / "train_examples.csv",
        validation_output_csv=tmp_path / "validation_examples.csv",
        metadata_output_json=tmp_path / "train_validation_split.json",
        validation_fraction=0.25,
        min_validation_scenarios=min_validation_scenarios,
        seed=seed,
    )


def _read_ids(path: Path) -> set[int]:
    return set(pd.read_csv(path)["scenario_id"].astype(int).tolist())


def test_split_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = _split(tmp_path, seed=123)
    first_train = (tmp_path / "train_examples.csv").read_bytes()
    first_val = (tmp_path / "validation_examples.csv").read_bytes()
    second = _split(tmp_path, seed=123)
    assert first["validation_scenario_ids"] == second["validation_scenario_ids"]
    assert first == second
    assert (tmp_path / "train_examples.csv").read_bytes() == first_train
    assert (tmp_path / "validation_examples.csv").read_bytes() == first_val


def test_split_has_no_scenario_overlap(tmp_path: Path) -> None:
    _split(tmp_path)
    assert _read_ids(tmp_path / "train_examples.csv").isdisjoint(
        _read_ids(tmp_path / "validation_examples.csv")
    )


def test_all_rows_are_preserved_exactly_once(tmp_path: Path) -> None:
    _split(tmp_path)
    assert len(pd.read_csv(tmp_path / "train_examples.csv")) + len(
        pd.read_csv(tmp_path / "validation_examples.csv")
    ) == 6


def test_all_steps_of_scenario_remain_together(tmp_path: Path) -> None:
    _split(tmp_path)
    train_ids = _read_ids(tmp_path / "train_examples.csv")
    val_ids = _read_ids(tmp_path / "validation_examples.csv")
    assert 1 not in train_ids or 1 not in val_ids
    assert 3 not in train_ids or 3 not in val_ids


def test_split_preserves_original_row_order(tmp_path: Path) -> None:
    _split(tmp_path)
    source = pd.read_csv(tmp_path / "train_batch.csv")
    for name in ["train_examples.csv", "validation_examples.csv"]:
        df = pd.read_csv(tmp_path / name)
        ids = set(df["scenario_id"].astype(int).tolist())
        expected = source[source["scenario_id"].isin(ids)].reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_minimum_validation_scenario_count_is_respected(tmp_path: Path) -> None:
    metadata = _split(tmp_path, min_validation_scenarios=2)
    assert metadata["validation_scenarios"] == 2


def test_single_scenario_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    pd.DataFrame([{"scenario_id": 1}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="at least two"):
        split_examples_by_scenario(examples_csv=source, train_output_csv=tmp_path/"t.csv", validation_output_csv=tmp_path/"v.csv", metadata_output_json=tmp_path/"m.json", validation_fraction=0.2, min_validation_scenarios=1, seed=1)


def test_impossible_minimum_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "two.csv"
    pd.DataFrame([{"scenario_id": 1}, {"scenario_id": 2}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="leaves no training"):
        split_examples_by_scenario(examples_csv=source, train_output_csv=tmp_path/"t.csv", validation_output_csv=tmp_path/"v.csv", metadata_output_json=tmp_path/"m.json", validation_fraction=0.2, min_validation_scenarios=2, seed=1)


def test_missing_scenario_id_column_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    pd.DataFrame([{"x": 1}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="scenario_id"):
        split_examples_by_scenario(examples_csv=source, train_output_csv=tmp_path/"t.csv", validation_output_csv=tmp_path/"v.csv", metadata_output_json=tmp_path/"m.json", validation_fraction=0.2, min_validation_scenarios=1, seed=1)


def test_fractional_scenario_id_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "fraction.csv"
    pd.DataFrame([{"scenario_id": 1.5}, {"scenario_id": 2.0}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="integer-valued"):
        split_examples_by_scenario(examples_csv=source, train_output_csv=tmp_path/"t.csv", validation_output_csv=tmp_path/"v.csv", metadata_output_json=tmp_path/"m.json", validation_fraction=0.2, min_validation_scenarios=1, seed=1)


def test_invalid_validation_fraction_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "batch.csv"
    _write_examples(source)
    with pytest.raises(ValueError, match="validation_fraction"):
        split_examples_by_scenario(examples_csv=source, train_output_csv=tmp_path/"t.csv", validation_output_csv=tmp_path/"v.csv", metadata_output_json=tmp_path/"m.json", validation_fraction=0.0, min_validation_scenarios=1, seed=1)


def test_split_metadata_contains_hashes_and_counts(tmp_path: Path) -> None:
    metadata = _split(tmp_path)
    assert metadata["total_examples"] == 6
    assert metadata["train_csv_sha256"]
    assert metadata["validation_csv_sha256"]
    assert metadata["source_csv_sha256"]
