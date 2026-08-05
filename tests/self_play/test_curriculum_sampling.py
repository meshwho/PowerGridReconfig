import copy

import pytest

from grid_topology_ai.config.pool import CurriculumSamplingConfig
from grid_topology_ai.self_play.pool_sampling import (
    compute_priority_breakdown,
    refresh_priorities,
    sample_curriculum_from_pool,
)


def _scenario(
    *,
    difficulty: str = "medium",
    attempts: int = 4,
    solved: int = 2,
    solve_rate: float = 0.5,
    last_attempted: int = 5,
    learning_progress: float = 0.2,
    uncertainty: float = 0.5,
) -> dict[str, object]:
    return {
        "difficulty_class": difficulty,
        "times_attempted": attempts,
        "times_solved": solved,
        "solve_rate": solve_rate,
        "last_attempted_iter": last_attempted,
        "last_solved_iter": last_attempted if solved else None,
        "avg_steps_when_solved": 4.0 if solved else None,
        "last_iteration_solve_rate": solve_rate,
        "solve_rate_delta": 0.0,
        "learning_progress": learning_progress,
        "uncertainty": uncertainty,
        "staleness": 0.0,
        "priority": 0.05,
    }


def _metadata(
    scenarios: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "transitions_csv": "transitions.csv",
        "last_updated_iteration": 5,
        "scenarios": scenarios,
    }


def test_priority_breakdown_uses_all_configured_components() -> None:
    config = CurriculumSamplingConfig(
        learning_progress_weight=2.0,
        uncertainty_weight=3.0,
        staleness_weight=4.0,
        frontier_weight=0.5,
        priority_floor=0.2,
    )

    breakdown = compute_priority_breakdown(
        solve_rate=0.5,
        learning_progress=0.25,
        uncertainty=0.4,
        staleness=0.5,
        difficulty_class="hard",
        config=config,
    )

    assert breakdown.learning_progress == pytest.approx(0.5)
    assert breakdown.uncertainty == pytest.approx(1.2)
    assert breakdown.staleness == pytest.approx(2.0)
    assert breakdown.frontier == pytest.approx(0.5)
    assert breakdown.difficulty_bonus == pytest.approx(0.2)
    assert breakdown.total == pytest.approx(4.6)


def test_priority_increases_with_each_learning_signal() -> None:
    base = compute_priority_breakdown(
        solve_rate=0.0,
        learning_progress=0.0,
        uncertainty=0.0,
        staleness=0.0,
        difficulty_class="simple",
    ).total

    progress = compute_priority_breakdown(
        solve_rate=0.0,
        learning_progress=1.0,
        uncertainty=0.0,
        staleness=0.0,
        difficulty_class="simple",
    ).total
    uncertainty = compute_priority_breakdown(
        solve_rate=0.0,
        learning_progress=0.0,
        uncertainty=1.0,
        staleness=0.0,
        difficulty_class="simple",
    ).total
    staleness = compute_priority_breakdown(
        solve_rate=0.0,
        learning_progress=0.0,
        uncertainty=0.0,
        staleness=1.0,
        difficulty_class="simple",
    ).total
    frontier = compute_priority_breakdown(
        solve_rate=0.5,
        learning_progress=0.0,
        uncertainty=0.0,
        staleness=0.0,
        difficulty_class="simple",
    ).total
    hard = compute_priority_breakdown(
        solve_rate=0.0,
        learning_progress=0.0,
        uncertainty=0.0,
        staleness=0.0,
        difficulty_class="hard",
    ).total

    assert base == pytest.approx(CurriculumSamplingConfig().priority_floor)
    assert progress > base
    assert uncertainty > base
    assert staleness > base
    assert frontier > base
    assert hard > base


def test_refresh_priorities_records_auditable_components() -> None:
    metadata = _metadata(
        {
            "1": _scenario(
                difficulty="hard",
                attempts=3,
                solved=1,
                solve_rate=0.5,
                last_attempted=1,
                learning_progress=0.3,
                uncertainty=0.4,
            )
        }
    )
    config = CurriculumSamplingConfig(stale_after_iterations=3)

    refresh_priorities(
        metadata,
        current_iter=4,
        config=config,
    )
    scenario = metadata["scenarios"]["1"]
    components = scenario["priority_components"]

    assert scenario["staleness"] == 1.0
    assert components["learning_progress"] == pytest.approx(0.3)
    assert components["uncertainty"] == pytest.approx(0.3)
    assert components["staleness"] == pytest.approx(0.5)
    assert components["frontier"] == pytest.approx(0.35)
    assert components["difficulty_bonus"] == pytest.approx(0.2)
    assert components["total"] == pytest.approx(scenario["priority"])


def test_curriculum_sampling_is_deterministic_and_unique() -> None:
    scenarios = {
        str(scenario_id): _scenario(
            difficulty=(
                "hard"
                if scenario_id % 3 == 0
                else "simple"
                if scenario_id % 3 == 1
                else "medium"
            ),
            attempts=4,
            solved=0 if scenario_id % 4 == 0 else 2,
            solve_rate=0.0 if scenario_id % 4 == 0 else 0.5,
            learning_progress=scenario_id / 20.0,
        )
        for scenario_id in range(1, 13)
    }
    metadata = _metadata(scenarios)

    first = sample_curriculum_from_pool(
        copy.deepcopy(metadata),
        n=7,
        seed=91,
        current_iter=8,
    )
    second = sample_curriculum_from_pool(
        copy.deepcopy(metadata),
        n=7,
        seed=91,
        current_iter=8,
    )

    assert first == second
    assert len(first.scenario_ids) == 7
    assert len(first.scenario_ids) == len(set(first.scenario_ids))


def test_hard_never_solved_scenario_counts_toward_both_quotas() -> None:
    metadata = _metadata(
        {
            "1": _scenario(
                difficulty="hard",
                attempts=3,
                solved=0,
                solve_rate=0.0,
            ),
            "2": _scenario(solve_rate=1.0, solved=4),
            "3": _scenario(solve_rate=1.0, solved=4),
            "4": _scenario(solve_rate=1.0, solved=4),
        }
    )
    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.25,
        hard_min_fraction=0.25,
        simple_max_fraction=1.0,
        frontier_max_fraction=1.0,
    )

    sample = sample_curriculum_from_pool(
        metadata,
        n=4,
        seed=3,
        current_iter=6,
        config=config,
    )

    assert 1 in sample.scenario_ids
    assert sample.report["never_solved"]["target"] == 1
    assert sample.report["never_solved"]["selected"] == 1
    assert sample.report["hard"]["target"] == 1
    assert sample.report["hard"]["selected"] == 1
    assert sample.report["never_solved"]["shortfall"] == 0
    assert sample.report["hard"]["shortfall"] == 0


def test_available_minimum_quotas_are_enforced() -> None:
    scenarios = {
        "1": _scenario(attempts=3, solved=0, solve_rate=0.0),
        "2": _scenario(
            difficulty="simple",
            attempts=3,
            solved=0,
            solve_rate=0.0,
        ),
        "3": _scenario(
            difficulty="hard",
            attempts=3,
            solved=0,
            solve_rate=0.0,
        ),
        "4": _scenario(difficulty="hard", solved=3, solve_rate=1.0),
        "5": _scenario(difficulty="hard", solved=3, solve_rate=1.0),
    }
    for scenario_id in range(6, 13):
        scenarios[str(scenario_id)] = _scenario(
            solved=4,
            solve_rate=1.0,
        )

    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.25,
        hard_min_fraction=0.25,
        simple_max_fraction=1.0,
        frontier_max_fraction=1.0,
    )
    sample = sample_curriculum_from_pool(
        _metadata(scenarios),
        n=8,
        seed=17,
        current_iter=7,
        config=config,
    )

    assert sample.report["never_solved"]["selected"] >= 2
    assert sample.report["hard"]["selected"] >= 2
    assert sample.report["never_solved"]["shortfall"] == 0
    assert sample.report["hard"]["shortfall"] == 0


def test_simple_and_frontier_caps_hold_when_alternatives_exist() -> None:
    scenarios: dict[str, dict[str, object]] = {}

    for scenario_id in range(1, 7):
        scenarios[str(scenario_id)] = _scenario(
            difficulty="simple",
            solved=4,
            solve_rate=1.0,
            learning_progress=1.0,
        )
    for scenario_id in range(7, 13):
        scenarios[str(scenario_id)] = _scenario(
            difficulty="medium",
            solved=2,
            solve_rate=0.5,
            learning_progress=1.0,
        )
    for scenario_id in range(13, 21):
        scenarios[str(scenario_id)] = _scenario(
            difficulty="medium",
            solved=4,
            solve_rate=1.0,
            learning_progress=0.0,
        )

    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.0,
        hard_min_fraction=0.0,
        simple_max_fraction=0.20,
        frontier_max_fraction=0.30,
    )
    sample = sample_curriculum_from_pool(
        _metadata(scenarios),
        n=10,
        seed=21,
        current_iter=7,
        config=config,
    )

    assert sample.report["simple"]["limit"] == 2
    assert sample.report["simple"]["selected"] <= 2
    assert sample.report["frontier"]["limit"] == 3
    assert sample.report["frontier"]["selected"] <= 3
    assert sample.report["cap_relaxations"] == []


def test_quota_shortfall_is_reported_when_candidates_are_scarce() -> None:
    metadata = _metadata(
        {
            "1": _scenario(
                difficulty="hard",
                attempts=3,
                solved=0,
                solve_rate=0.0,
            ),
            "2": _scenario(
                difficulty="hard",
                solved=3,
                solve_rate=1.0,
            ),
            "3": _scenario(solved=4, solve_rate=1.0),
            "4": _scenario(solved=4, solve_rate=1.0),
            "5": _scenario(solved=4, solve_rate=1.0),
            "6": _scenario(solved=4, solve_rate=1.0),
        }
    )
    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.50,
        hard_min_fraction=0.50,
        simple_max_fraction=1.0,
        frontier_max_fraction=1.0,
    )

    sample = sample_curriculum_from_pool(
        metadata,
        n=6,
        seed=4,
        current_iter=7,
        config=config,
    )

    assert sample.report["never_solved"] == {
        "available": 1,
        "target": 3,
        "selected": 1,
        "shortfall": 2,
        "fraction": pytest.approx(1 / 6),
    }
    assert sample.report["hard"]["available"] == 2
    assert sample.report["hard"]["target"] == 3
    assert sample.report["hard"]["selected"] == 2
    assert sample.report["hard"]["shortfall"] == 1


def test_caps_are_relaxed_only_to_finish_an_otherwise_impossible_batch() -> None:
    scenarios = {
        str(scenario_id): _scenario(
            difficulty="simple",
            solved=2,
            solve_rate=0.5,
        )
        for scenario_id in range(1, 5)
    }
    config = CurriculumSamplingConfig(
        never_solved_min_fraction=0.0,
        hard_min_fraction=0.0,
        simple_max_fraction=0.25,
        frontier_max_fraction=0.25,
    )

    sample = sample_curriculum_from_pool(
        _metadata(scenarios),
        n=4,
        seed=2,
        current_iter=7,
        config=config,
    )

    assert len(sample.scenario_ids) == 4
    assert len(sample.scenario_ids) == len(set(sample.scenario_ids))
    assert sample.report["simple"]["limit"] == 1
    assert sample.report["frontier"]["limit"] == 1
    assert sample.report["cap_relaxations"] == [
        "frontier_max_fraction",
        "simple_max_fraction",
    ]
