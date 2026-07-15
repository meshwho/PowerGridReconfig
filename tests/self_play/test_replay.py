from pathlib import Path

from grid_topology_ai.config import ReplayBufferConfig
from grid_topology_ai.self_play.replay import RollingReplayBuffer


def rows(prefix: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "state_id": f"{prefix}_{index}",
            "scenario_id": index,
        }
        for index in range(count)
    ]


def test_mixed_batch_respects_fresh_fraction(
    tmp_path: Path,
) -> None:
    buffer = RollingReplayBuffer(
        save_dir=tmp_path / "replay",
        config=ReplayBufferConfig(
            max_size=300,
            min_size_to_train=1,
            fresh_fraction=0.70,
            random_seed=42,
        ),
    )

    buffer.add_examples(rows("old", 100), iteration=1)
    buffer.add_examples(rows("fresh", 100), iteration=2)

    metadata = buffer.export_mixed_batch(
        output_path=tmp_path / "batch.csv",
        current_iteration=2,
        n_examples=100,
        seed=42,
    )

    assert metadata["n_examples"] == 100
    assert metadata["n_fresh"] == 70
    assert metadata["n_old"] == 30
    assert metadata["fresh_fraction_actual"] == 0.70


def test_reload_preserves_fifo_order(tmp_path: Path) -> None:
    save_dir = tmp_path / "replay"
    config = ReplayBufferConfig(
        max_size=4,
        min_size_to_train=1,
    )

    buffer = RollingReplayBuffer(
        save_dir=save_dir,
        config=config,
    )

    iteration_1 = rows("i1", 2)
    buffer.add_examples(iteration_1, iteration=1)
    buffer.save_iteration_file(iteration_1, iteration=1)
    buffer.save_manifest()

    iteration_2 = rows("i2", 3)
    buffer.add_examples(iteration_2, iteration=2)
    buffer.save_iteration_file(iteration_2, iteration=2)
    buffer.save_manifest()

    reloaded = RollingReplayBuffer(
        save_dir=save_dir,
        config=config,
    )

    state_ids = [
        str(row["state_id"])
        for row in reloaded.buffer
    ]

    assert state_ids == [
        "i1_1",
        "i2_0",
        "i2_1",
        "i2_2",
    ]


def test_rolling_replay_buffer_class_name_is_explicit() -> None:
    assert RollingReplayBuffer.__name__ == "RollingReplayBuffer"

import gzip
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


def _write_valid_state(path: Path) -> Path:
    np.savez(
        path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        action_mask=np.array([True, True], dtype=bool),
    )
    return path


def _valid_example_row(state_path: Path, *, state_id: str = "state-1") -> dict[str, object]:
    return {
        "state_path": str(state_path),
        "mcts_policy_json": '{"0": 0.25, "1": 0.75}',
        "scenario_id": 1,
        "step": 0,
        "state_id": state_id,
        "outcome_value_target": 1.0,
    }


def _write_examples_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _valid_csv(tmp_path: Path) -> Path:
    return _write_examples_csv(tmp_path / "examples.csv", [_valid_example_row(_write_valid_state(tmp_path / "s.npz"))])


def _invalid_csv(tmp_path: Path, *, name: str = "invalid.csv") -> Path:
    row = _valid_example_row(_write_valid_state(tmp_path / f"{name}.npz"))
    row["mcts_policy_json"] = "{}"
    return _write_examples_csv(tmp_path / name, [row])


def test_valid_csv_is_added_and_persisted(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay", config=ReplayBufferConfig(max_size=10, min_size_to_train=1))
    returned = buffer.add_and_save_from_csv(examples_csv=_valid_csv(tmp_path), iteration=1)
    assert len(returned) == 1
    assert len(buffer.buffer) == 1
    iter_file = tmp_path / "replay" / "buffer_iter_001.jsonl.gz"
    manifest = tmp_path / "replay" / "buffer_manifest.json"
    assert iter_file.exists()
    assert manifest.exists()
    with gzip.open(iter_file, "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["replay_iteration"] == 1


def test_invalid_csv_does_not_mutate_buffer(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay", config=ReplayBufferConfig(max_size=1, min_size_to_train=1))
    buffer.add_examples([{"state_id": "old"}], iteration=0)
    before = deepcopy(buffer.buffer)
    with pytest.raises(ValueError):
        buffer.add_and_save_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=1)
    assert buffer.buffer == before


def test_invalid_csv_does_not_create_iteration_file(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay")
    with pytest.raises(ValueError):
        buffer.add_and_save_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=1)
    assert not (tmp_path / "replay" / "buffer_iter_001.jsonl.gz").exists()


def test_invalid_csv_does_not_create_manifest(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay")
    with pytest.raises(ValueError):
        buffer.add_and_save_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=1)
    assert not (tmp_path / "replay" / "buffer_manifest.json").exists()


def test_invalid_csv_does_not_overwrite_existing_iteration_file(tmp_path: Path) -> None:
    save_dir = tmp_path / "replay"; save_dir.mkdir()
    existing = save_dir / "buffer_iter_002.jsonl.gz"
    known = b"known bytes"
    existing.write_bytes(known)
    buffer = RollingReplayBuffer(save_dir=save_dir)
    with pytest.raises(ValueError):
        buffer.add_and_save_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=2)
    assert existing.read_bytes() == known


def test_missing_state_file_does_not_mutate_replay(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay")
    buffer.add_examples([{"state_id": "old"}], iteration=0)
    before = deepcopy(buffer.buffer)
    csv = _write_examples_csv(tmp_path / "missing.csv", [_valid_example_row(tmp_path / "missing.npz")])
    with pytest.raises(FileNotFoundError):
        buffer.add_and_save_from_csv(examples_csv=csv, iteration=1)
    assert buffer.buffer == before


def test_invalid_policy_does_not_mutate_replay(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay")
    buffer.add_examples([{"state_id": "old"}], iteration=0)
    before = deepcopy(buffer.buffer)
    with pytest.raises(ValueError):
        buffer.add_and_save_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=1)
    assert buffer.buffer == before


def test_add_examples_from_csv_validates_before_mutation(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(save_dir=tmp_path / "replay")
    before = deepcopy(buffer.buffer)
    with pytest.raises(ValueError):
        buffer.add_examples_from_csv(examples_csv=_invalid_csv(tmp_path), iteration=1)
    assert buffer.buffer == before
