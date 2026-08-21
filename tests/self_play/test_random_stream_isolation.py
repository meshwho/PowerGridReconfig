from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play.generation import GenerationRequest, _scenario_seed


def test_scenario_seed_is_reproducible_order_independent_and_distinct() -> None:
    stream = 987654
    forward = {sid: _scenario_seed(stream, sid) for sid in (1, 2, 3, 4)}
    reverse = {sid: _scenario_seed(stream, sid) for sid in (4, 3, 2, 1)}
    assert forward == reverse
    assert len(set(forward.values())) == 4
    assert _scenario_seed(stream + 1, 1) != forward[1]


def test_action_sampling_does_not_consume_mcts_stream() -> None:
    mcts_seed = _scenario_seed(17, 12)
    action_seed = _scenario_seed(18, 12)
    expected = np.random.default_rng(mcts_seed).integers(0, 1_000_000, 8)
    np.random.default_rng(action_seed).random(10_000)
    actual = np.random.default_rng(mcts_seed).integers(0, 1_000_000, 8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("field_name", ["mcts_seed", "action_seed"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_generation_request_rejects_invalid_stream_seed(
    field_name: str, value: object, tmp_path: Path
) -> None:
    kwargs: dict[str, object] = dict(
        raw_dir=tmp_path / "raw", transitions_csv=tmp_path / "transitions.csv",
        output_dir=tmp_path / "out", checkpoint=None, config=GenerationConfig(),
        mcts_seed=10, action_seed=20, clear_cache_between_scenarios=False,
    )
    kwargs[field_name] = value
    with pytest.raises(ValueError, match=rf"{field_name} must be a non-negative integer"):
        GenerationRequest(**kwargs)  # type: ignore[arg-type]


def test_generation_request_normalizes_numpy_integer_fields(tmp_path: Path) -> None:
    request = GenerationRequest(
        raw_dir=tmp_path / "raw", transitions_csv=tmp_path / "transitions.csv",
        output_dir=tmp_path / "out", checkpoint=None, config=GenerationConfig(),
        mcts_seed=np.int64(10), action_seed=np.int64(20),
        clear_cache_between_scenarios=False, workers=np.int64(2),
    )
    assert (request.mcts_seed, request.action_seed, request.workers) == (10, 20, 2)
    assert all(type(value) is int for value in (
        request.mcts_seed, request.action_seed, request.workers
    ))
