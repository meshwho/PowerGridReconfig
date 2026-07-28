from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    _scenario_seed,
)
from grid_topology_ai.self_play.iteration import _self_play_seeds


def test_iteration_streams_are_reproducible_and_distinct() -> None:
    first = _self_play_seeds(
        base_seed=42,
        iteration=3,
    )
    repeated = _self_play_seeds(
        base_seed=42,
        iteration=3,
    )

    assert first == repeated
    assert len(
        {
            first.scenario_sampling,
            first.mcts,
            first.action_sampling,
        }
    ) == 3
    assert all(
        isinstance(value, int) and value >= 0
        for value in (
            first.scenario_sampling,
            first.mcts,
            first.action_sampling,
        )
    )


def test_iteration_number_changes_each_random_stream() -> None:
    first = _self_play_seeds(
        base_seed=42,
        iteration=1,
    )
    second = _self_play_seeds(
        base_seed=42,
        iteration=2,
    )

    assert first.scenario_sampling != second.scenario_sampling
    assert first.mcts != second.mcts
    assert first.action_sampling != second.action_sampling


def test_scenario_seed_depends_only_on_stream_and_scenario() -> None:
    stream_seed = 123456

    assert _scenario_seed(stream_seed, 7) == _scenario_seed(
        stream_seed,
        7,
    )
    assert _scenario_seed(stream_seed, 7) != _scenario_seed(
        stream_seed,
        8,
    )
    assert _scenario_seed(stream_seed, 7) != _scenario_seed(
        stream_seed + 1,
        7,
    )


def test_scenario_order_does_not_change_assigned_seeds() -> None:
    stream_seed = 987654

    forward = {
        scenario_id: _scenario_seed(
            stream_seed,
            scenario_id,
        )
        for scenario_id in (1, 2, 3, 4)
    }
    reversed_order = {
        scenario_id: _scenario_seed(
            stream_seed,
            scenario_id,
        )
        for scenario_id in (4, 3, 2, 1)
    }

    assert forward == reversed_order


def test_action_sampling_does_not_consume_mcts_random_stream() -> None:
    streams = _self_play_seeds(
        base_seed=17,
        iteration=4,
    )
    scenario_id = 12
    mcts_seed = _scenario_seed(
        streams.mcts,
        scenario_id,
    )
    action_seed = _scenario_seed(
        streams.action_sampling,
        scenario_id,
    )

    expected_mcts_values = np.random.default_rng(
        mcts_seed
    ).integers(
        0,
        1_000_000,
        size=8,
    )

    action_rng = np.random.default_rng(
        action_seed
    )
    action_rng.random(10_000)

    actual_mcts_values = np.random.default_rng(
        mcts_seed
    ).integers(
        0,
        1_000_000,
        size=8,
    )

    np.testing.assert_array_equal(
        actual_mcts_values,
        expected_mcts_values,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("mcts_seed", True),
        ("mcts_seed", -1),
        ("mcts_seed", 1.5),
        ("mcts_seed", "1"),
        ("mcts_seed", None),
        ("action_seed", True),
        ("action_seed", -1),
        ("action_seed", 1.5),
        ("action_seed", "1"),
        ("action_seed", None),
    ],
)
def test_generation_request_rejects_invalid_stream_seed(
    field_name: str,
    value: object,
    tmp_path: Path,
) -> None:
    kwargs: dict[str, object] = {
        "raw_dir": tmp_path / "raw",
        "transitions_csv": tmp_path / "transitions.csv",
        "output_dir": tmp_path / "out",
        "checkpoint": None,
        "config": GenerationConfig(),
        "mcts_seed": 10,
        "action_seed": 20,
        "clear_cache_between_scenarios": False,
    }
    kwargs[field_name] = value

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a non-negative integer",
    ):
        GenerationRequest(
            **kwargs  # type: ignore[arg-type]
        )


def test_generation_request_normalizes_numpy_integer_seeds(
    tmp_path: Path,
) -> None:
    request = GenerationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=tmp_path / "transitions.csv",
        output_dir=tmp_path / "out",
        checkpoint=None,
        config=GenerationConfig(),
        mcts_seed=np.int64(10),
        action_seed=np.int64(20),
        clear_cache_between_scenarios=False,
    )

    assert request.mcts_seed == 10
    assert type(request.mcts_seed) is int
    assert request.action_seed == 20
    assert type(request.action_seed) is int
