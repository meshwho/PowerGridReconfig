import copy

import pytest

from grid_topology_ai.self_play.pool_sampling import compute_priority, refresh_priorities, sample_from_pool


def _scenario(*, priority=1.0, attempts=0, solve_rate=0.0, last=0, difficulty="medium", **extra):
    scenario = {
        "difficulty_class": difficulty,
        "times_attempted": attempts,
        "times_solved": 0,
        "solve_rate": solve_rate,
        "last_attempted_iter": last,
        "last_solved_iter": None,
        "avg_steps_when_solved": None,
        "priority": priority,
    }
    scenario.update(extra)
    return scenario


def _metadata(size=5):
    return {
        "schema_version": 2,
        "last_updated_iteration": 0,
        "scenarios": {str(i): _scenario(priority=float(i)) for i in range(1, size + 1)},
    }


def test_sampling_is_deterministic_for_same_seed() -> None:
    assert sample_from_pool(copy.deepcopy(_metadata()), 3, seed=7) == sample_from_pool(copy.deepcopy(_metadata()), 3, seed=7)


def test_sampling_changes_with_seed_when_pool_allows_it() -> None:
    metadata = _metadata(8)
    assert sample_from_pool(metadata, 4, seed=1) != sample_from_pool(metadata, 4, seed=2)


def test_sampling_returns_unique_ids() -> None:
    chosen = sample_from_pool(_metadata(), 5, seed=3)
    assert len(chosen) == len(set(chosen))


def test_sampling_respects_requested_count() -> None:
    assert len(sample_from_pool(_metadata(3), 10, seed=3)) == 3


def test_sampling_rejects_invalid_count() -> None:
    with pytest.raises(ValueError):
        sample_from_pool(_metadata(3), -1, seed=3)


def test_priority_prefers_unsolved_or_hard_scenarios() -> None:
    unsolved = compute_priority(0.0, 0, 0, 0, "medium")
    hard_frontier = compute_priority(0.5, 1, 0, 0, "hard")
    easy_solved = compute_priority(1.0, 2, 0, 0, "simple")

    assert unsolved > easy_solved
    assert hard_frontier > unsolved


def test_stale_priorities_are_refreshed_for_all_scenarios() -> None:
    metadata = {"scenarios": {"1": _scenario(attempts=1, solve_rate=0.5, last=1, priority=1.0), "2": _scenario(attempts=1, solve_rate=0.5, last=1, priority=1.0)}}

    refresh_priorities(metadata, current_iter=6)

    assert metadata["scenarios"]["1"]["priority"] == metadata["scenarios"]["2"]["priority"]
    assert metadata["scenarios"]["1"]["priority"] > 1.0


def test_sampling_does_not_mutate_unrelated_metadata_fields() -> None:
    metadata = {"scenarios": {"1": _scenario(priority=1.0, custom="kept"), "2": _scenario(priority=1.0)}}

    sample_from_pool(metadata, 1, seed=1)

    assert metadata["scenarios"]["1"]["custom"] == "kept"


def test_sampling_preserves_scenario_order_before_rng() -> None:
    metadata = {"scenarios": {"10": _scenario(priority=1.0), "2": _scenario(priority=1.0), "7": _scenario(priority=1.0)}}

    assert sample_from_pool(metadata, 3, seed=0) == [2, 10, 7]


def test_sampling_regression_fixed_seed_priorities_and_uniqueness() -> None:
    scenarios = {}
    rows = [
        (0.0, 0, 0, "simple"),
        (0.2, 2, 1, "medium"),
        (0.5, 3, 2, "hard"),
        (1.0, 4, 0, "medium"),
        (0.75, 1, 1, "simple"),
        (0.1, 5, 3, "hard"),
    ]
    for idx, (solve_rate, attempts, last, difficulty) in enumerate(rows, 1):
        scenarios[str(idx)] = _scenario(
            attempts=attempts,
            solve_rate=solve_rate,
            last=last,
            difficulty=difficulty,
            priority=compute_priority(solve_rate, attempts, last, 3, difficulty),
        )
    metadata = {"schema_version": 2, "last_updated_iteration": 3, "scenarios": scenarios}

    chosen = sample_from_pool(copy.deepcopy(metadata), 3, seed=123)
    refresh_priorities(metadata, current_iter=7)

    assert chosen == [3, 1, 2]
    assert len(chosen) == len(set(chosen))
    assert {key: round(value["priority"], 6) for key, value in metadata["scenarios"].items()} == {
        "1": 0.8,
        "2": 0.94,
        "3": 1.56,
        "4": 0.3,
        "5": 0.84,
        "6": 0.72,
    }
