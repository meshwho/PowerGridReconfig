from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from grid_topology_ai.config import ReplayBufferConfig
from grid_topology_ai.self_play.replay import RollingReplayBuffer
from grid_topology_ai.self_play.replay_sampling import (
    AGE_DECAY,
    ERROR_PRIORITY_SCALE,
    SAMPLING_CONTRACT_VERSION,
)


def _row(
    state_id: str,
    *,
    episode_id: str,
    scenario_id: int,
    replay_iteration: int = 2,
    outcome: str = "solved",
    **extra: object,
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "replay_iteration": replay_iteration,
        "outcome_class": outcome,
        **extra,
    }


def _buffer(tmp_path: Path) -> RollingReplayBuffer:
    return RollingReplayBuffer(
        save_dir=tmp_path / "replay",
        config=ReplayBufferConfig(
            max_size=100,
            min_size_to_train=1,
            fresh_fraction=0.5,
            random_seed=17,
        ),
    )


def _priorities(
    buffer: RollingReplayBuffer,
    rows: list[dict[str, object]],
    *,
    current_iteration: int = 2,
) -> dict[str, float]:
    episodes, _ = buffer._episode_groups(
        rows,
        current_iteration=current_iteration,
        rng=np.random.default_rng(1),
    )
    return {
        str(episode["rows"][0]["episode_id"]): float(
            episode["priority"]
        )
        for episode in episodes
    }


def test_episode_sampling_does_not_favor_long_episodes(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        *[
            _row(
                f"long-{step}",
                episode_id="long",
                scenario_id=1,
            )
            for step in range(5)
        ],
        _row("short-a", episode_id="short-a", scenario_id=2),
        _row("short-b", episode_id="short-b", scenario_id=3),
    ]

    selected, metadata = buffer._sample_episode_rows(
        source,
        n_examples=3,
        current_iteration=2,
        rng=np.random.default_rng(4),
    )

    assert {str(row["episode_id"]) for row in selected} == {
        "long",
        "short-a",
        "short-b",
    }
    assert metadata["source_episodes"] == 3
    assert metadata["selected_episodes"] == 3


def test_episode_sampling_balances_outcome_and_difficulty_strata(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    buffer.set_scenario_metadata(
        {
            "scenarios": {
                "1": {"difficulty_class": "easy"},
                "2": {"difficulty_class": "easy"},
                "3": {"difficulty_class": "hard"},
                "4": {"difficulty_class": "hard"},
            }
        }
    )
    source = [
        _row("solved-1", episode_id="solved-1", scenario_id=1),
        _row("solved-2", episode_id="solved-2", scenario_id=2),
        _row(
            "handoff-1",
            episode_id="handoff-1",
            scenario_id=3,
            outcome="handoff_to_redispatch",
        ),
        _row(
            "handoff-2",
            episode_id="handoff-2",
            scenario_id=4,
            outcome="handoff_to_redispatch",
        ),
    ]

    _, metadata = buffer._sample_episode_rows(
        source,
        n_examples=2,
        current_iteration=2,
        rng=np.random.default_rng(8),
    )

    assert metadata["selected_strata"] == {
        "outcome=handoff_to_redispatch|difficulty=hard": 1,
        "outcome=solved|difficulty=easy": 1,
    }


def test_episode_sampling_is_deterministic_for_seed(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        _row(
            f"episode-{episode}-step-{step}",
            episode_id=f"episode-{episode}",
            scenario_id=episode,
        )
        for episode in range(4)
        for step in range(2)
    ]

    first, _ = buffer._sample_episode_rows(
        source,
        n_examples=5,
        current_iteration=2,
        rng=np.random.default_rng(19),
    )
    second, _ = buffer._sample_episode_rows(
        source,
        n_examples=5,
        current_iteration=2,
        rng=np.random.default_rng(19),
    )

    assert [row["state_id"] for row in first] == [
        row["state_id"] for row in second
    ]


def test_episode_priority_combines_age_and_error(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        _row(
            "old",
            episode_id="old",
            scenario_id=1,
            replay_iteration=1,
        ),
        _row(
            "fresh",
            episode_id="fresh",
            scenario_id=2,
        ),
        _row(
            "error",
            episode_id="error",
            scenario_id=3,
            value_error=9.0,
        ),
    ]

    priorities = _priorities(buffer, source)

    assert priorities["old"] == AGE_DECAY
    assert priorities["fresh"] == 1.0
    assert priorities["error"] == (
        1.0 + ERROR_PRIORITY_SCALE * 0.9
    )


def test_policy_loss_activates_priority_without_explicit_error(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        _row(
            "certain",
            episode_id="certain",
            scenario_id=1,
            selected_action_id=1,
            mcts_policy_json=json.dumps({"1": 1.0}),
        ),
        _row(
            "surprising",
            episode_id="surprising",
            scenario_id=2,
            selected_action_id=1,
            mcts_policy_json=json.dumps({"1": 0.1, "2": 0.9}),
        ),
    ]

    priorities = _priorities(buffer, source)
    policy_loss = -math.log(0.1)
    expected_score = policy_loss / (1.0 + policy_loss)

    assert priorities["certain"] == 1.0
    assert np.isclose(
        priorities["surprising"],
        1.0 + ERROR_PRIORITY_SCALE * expected_score,
    )


def test_explicit_error_takes_precedence_over_policy_loss(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        _row(
            "explicit",
            episode_id="explicit",
            scenario_id=1,
            selected_action_id=1,
            mcts_policy_json=json.dumps({"1": 0.01, "2": 0.99}),
            value_error=1.0,
        )
    ]

    priority = _priorities(buffer, source)["explicit"]

    assert np.isclose(
        priority,
        1.0 + ERROR_PRIORITY_SCALE * 0.5,
    )


def test_missing_or_invalid_policy_error_fallback_is_neutral(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    source = [
        _row(
            "missing",
            episode_id="missing",
            scenario_id=1,
        ),
        _row(
            "invalid-json",
            episode_id="invalid-json",
            scenario_id=2,
            selected_action_id=1,
            mcts_policy_json="not-json",
        ),
        _row(
            "missing-action",
            episode_id="missing-action",
            scenario_id=3,
            selected_action_id=3,
            mcts_policy_json=json.dumps({"1": 1.0}),
        ),
    ]

    priorities = _priorities(buffer, source)

    assert priorities == {
        "missing": 1.0,
        "invalid-json": 1.0,
        "missing-action": 1.0,
    }


def test_export_mixed_batch_records_sampling_contract(
    tmp_path: Path,
) -> None:
    buffer = _buffer(tmp_path)
    buffer.set_scenario_metadata(
        {
            "scenarios": {
                str(index): {
                    "difficulty_class": (
                        "easy" if index % 2 else "hard"
                    )
                }
                for index in range(1, 7)
            }
        }
    )
    buffer.buffer = [
        *[
            _row(
                f"old-{index}",
                episode_id=f"old-{index}",
                scenario_id=index,
                replay_iteration=1,
            )
            for index in range(1, 4)
        ],
        *[
            _row(
                f"fresh-{index}",
                episode_id=f"fresh-{index}",
                scenario_id=index,
                replay_iteration=2,
            )
            for index in range(4, 7)
        ],
    ]
    output_path = tmp_path / "train_batch.csv"

    metadata = buffer.export_mixed_batch(
        output_path=output_path,
        current_iteration=2,
        n_examples=4,
        fresh_fraction=0.5,
        seed=23,
    )

    assert metadata["n_fresh"] == 2
    assert metadata["n_old"] == 2
    assert metadata["sampling_contract_version"] == (
        SAMPLING_CONTRACT_VERSION
    )
    assert metadata["sampling_unit"] == "episode_then_state"
    assert metadata["sampling_strata"] == [
        "outcome",
        "difficulty",
    ]
    assert metadata["scenario_metadata_count"] == 6
    assert metadata["error_priority_source"] == (
        "explicit_error_or_selected_action_policy_loss"
    )
    assert metadata["fresh_sampling"]["selected_episodes"] == 2
    assert metadata["old_sampling"]["selected_episodes"] == 2

    frame = pd.read_csv(output_path)
    assert len(frame) == 4

    metadata_path = output_path.with_suffix(".metadata.json")
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["sampling_contract_version"] == (
        SAMPLING_CONTRACT_VERSION
    )
    assert persisted["error_priority_source"] == (
        metadata["error_priority_source"]
    )
    assert persisted["fresh_sampling"] == metadata["fresh_sampling"]
    assert persisted["old_sampling"] == metadata["old_sampling"]
