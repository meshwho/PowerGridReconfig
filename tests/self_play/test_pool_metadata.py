from grid_topology_ai.self_play.pool_metadata import (
    compute_priority,
    update_pool_metadata,
)


def scenario(
    *,
    attempts: int,
    solved: int,
    solve_rate: float,
    last_attempted: int,
) -> dict[str, object]:
    return {
        "difficulty_class": "medium",
        "times_attempted": attempts,
        "times_solved": solved,
        "solve_rate": solve_rate,
        "last_attempted_iter": last_attempted,
        "last_solved_iter": None,
        "avg_steps_when_solved": None,
        "priority": compute_priority(
            solve_rate=solve_rate,
            times_attempted=attempts,
            last_attempted_iter=last_attempted,
            current_iter=last_attempted,
            difficulty_class="medium",
        ),
    }


def test_refreshes_priority_for_stale_scenario() -> None:
    metadata = {
        "schema_version": 2,
        "last_updated_iteration": 1,
        "scenarios": {
            "1": scenario(
                attempts=1,
                solved=0,
                solve_rate=0.50,
                last_attempted=1,
            ),
            "2": scenario(
                attempts=0,
                solved=0,
                solve_rate=0.0,
                last_attempted=0,
            ),
        },
    }

    before = float(metadata["scenarios"]["1"]["priority"])

    update_pool_metadata(
        pool_metadata=metadata,
        episode_results=[
            {
                "scenario_id": 2,
                "solved": False,
                "steps": 1,
            }
        ],
        current_iter=3,
    )

    after = float(metadata["scenarios"]["1"]["priority"])

    assert after > before
    assert metadata["last_updated_iteration"] == 3


def test_updates_scenario_statistics() -> None:
    metadata = {
        "schema_version": 2,
        "last_updated_iteration": 0,
        "scenarios": {
            "1": scenario(
                attempts=0,
                solved=0,
                solve_rate=0.0,
                last_attempted=0,
            ),
        },
    }

    update_pool_metadata(
        pool_metadata=metadata,
        episode_results=[
            {
                "scenario_id": 1,
                "solved": True,
                "steps": 2,
            }
        ],
        current_iter=1,
    )

    result = metadata["scenarios"]["1"]

    assert result["times_attempted"] == 1
    assert result["times_solved"] == 1
    assert result["solve_rate"] == 1.0
    assert result["last_attempted_iter"] == 1
    assert result["last_solved_iter"] == 1
    assert result["avg_steps_when_solved"] == 2.0