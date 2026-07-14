from pathlib import Path

from grid_topology_ai.self_play.replay_buffer_v2 import (
    ReplayBuffer,
    ReplayBufferConfig,
)


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
    buffer = ReplayBuffer(
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

    buffer = ReplayBuffer(
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

    reloaded = ReplayBuffer(
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