import json
import math
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.self_play.pool_state import (
    SCHEMA_VERSION,
    initialize_pool_metadata,
    load_json,
    update_pool_metadata,
)


def _transitions_csv(tmp_path: Path, scenario_id: int = 7) -> Path:
    path = tmp_path / "transitions.csv"
    pd.DataFrame(
        [{"scenario_id": scenario_id, "difficulty_class": "hard"}]
    ).to_csv(path, index=False)
    return path


def _v3_scenario(
    *,
    attempts: int = 10,
    solved: int = 5,
    solve_rate: float = 0.5,
    last_attempted_iter: int = 2,
    last_iteration_solve_rate: float | None = 0.5,
    learning_progress: float = 0.2,
) -> dict[str, object]:
    return {
        "difficulty_class": "hard",
        "times_attempted": attempts,
        "times_solved": solved,
        "solve_rate": solve_rate,
        "last_attempted_iter": last_attempted_iter,
        "last_solved_iter": 2,
        "avg_steps_when_solved": 4.5,
        "last_iteration_solve_rate": last_iteration_solve_rate,
        "solve_rate_delta": 0.0,
        "learning_progress": learning_progress,
        "uncertainty": 1.0,
        "staleness": 0.0,
        "priority": 0.5,
    }


def _metadata(scenario: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "transitions_csv": "transitions.csv",
        "last_updated_iteration": 2,
        "scenarios": {"7": scenario},
    }


def _beta_uncertainty(attempts: int, solved: int) -> float:
    alpha = solved + 1.0
    beta = attempts - solved + 1.0
    total = alpha + beta
    variance = alpha * beta / (total * total * (total + 1.0))
    return min(math.sqrt(variance) / math.sqrt(1.0 / 12.0), 1.0)


def test_v2_migration_preserves_historical_statistics(tmp_path: Path) -> None:
    transitions = _transitions_csv(tmp_path)
    path = tmp_path / "pool_metadata.json"
    legacy_scenario = {
        "difficulty_class": "hard",
        "times_attempted": 10,
        "times_solved": 3,
        "solve_rate": 0.3,
        "last_attempted_iter": 2,
        "last_solved_iter": 1,
        "avg_steps_when_solved": 5.5,
        "priority": 0.42,
    }
    legacy = {
        "schema_version": 2,
        "transitions_csv": str(transitions),
        "last_updated_iteration": 4,
        "scenarios": {"7": legacy_scenario},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = initialize_pool_metadata(
        transitions,
        path,
        current_iter=8,
        stale_after_iterations=4,
    )
    scenario = migrated["scenarios"]["7"]

    assert migrated["schema_version"] == SCHEMA_VERSION
    assert scenario["times_attempted"] == 10
    assert scenario["times_solved"] == 3
    assert scenario["solve_rate"] == 0.3
    assert scenario["last_attempted_iter"] == 2
    assert scenario["last_solved_iter"] == 1
    assert scenario["avg_steps_when_solved"] == 5.5
    assert scenario["priority"] == 0.42
    assert scenario["last_iteration_solve_rate"] is None
    assert scenario["solve_rate_delta"] == 0.0
    assert scenario["learning_progress"] == 0.0
    assert scenario["uncertainty"] == pytest.approx(
        _beta_uncertainty(10, 3)
    )
    assert scenario["staleness"] == 1.0
    assert load_json(path) == migrated


def test_loading_metadata_uses_configured_staleness_threshold(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(tmp_path)
    path = tmp_path / "pool_metadata.json"
    metadata = _metadata(
        _v3_scenario(last_attempted_iter=4)
    )
    metadata["last_updated_iteration"] = 6
    path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = initialize_pool_metadata(
        transitions,
        path,
        current_iter=6,
        stale_after_iterations=4,
    )

    assert loaded["scenarios"]["7"]["staleness"] == 0.5


def test_learning_progress_tracks_absolute_solve_rate_change() -> None:
    metadata = _metadata(_v3_scenario())
    results = [
        {"scenario_id": 7, "solved": solved, "steps": 3}
        for solved in (True, True, True, False)
    ]

    update_pool_metadata(
        metadata,
        results,
        current_iter=3,
        selected_scenario_ids=[7],
        ema_alpha=0.30,
    )
    scenario = metadata["scenarios"]["7"]

    assert scenario["last_iteration_solve_rate"] == 0.75
    assert scenario["solve_rate_delta"] == 0.25
    assert scenario["learning_progress"] == pytest.approx(0.215)
    assert scenario["solve_rate"] == pytest.approx(0.575)

    regression = [
        {"scenario_id": 7, "solved": solved, "steps": 3}
        for solved in (True, False, False, False)
    ]
    update_pool_metadata(
        metadata,
        regression,
        current_iter=4,
        selected_scenario_ids=[7],
        ema_alpha=0.30,
    )
    scenario = metadata["scenarios"]["7"]

    assert scenario["last_iteration_solve_rate"] == 0.25
    assert scenario["solve_rate_delta"] == pytest.approx(-0.325)
    assert scenario["learning_progress"] == pytest.approx(0.248)


def test_uncertainty_decreases_as_attempts_accumulate() -> None:
    metadata = _metadata(
        _v3_scenario(
            attempts=0,
            solved=0,
            solve_rate=0.0,
            last_attempted_iter=0,
            last_iteration_solve_rate=None,
            learning_progress=0.0,
        )
    )

    first_batch = [
        {"scenario_id": 7, "solved": False, "steps": 2}
        for _ in range(20)
    ]
    update_pool_metadata(metadata, first_batch, current_iter=1)
    first_uncertainty = metadata["scenarios"]["7"]["uncertainty"]

    second_batch = [
        {"scenario_id": 7, "solved": False, "steps": 2}
        for _ in range(100)
    ]
    update_pool_metadata(metadata, second_batch, current_iter=2)
    second_uncertainty = metadata["scenarios"]["7"]["uncertainty"]

    assert 0.0 < second_uncertainty < first_uncertainty < 1.0


def test_empty_results_still_update_all_selected_scenarios() -> None:
    metadata = _metadata(
        _v3_scenario(
            attempts=0,
            solved=0,
            solve_rate=0.0,
            last_attempted_iter=0,
            last_iteration_solve_rate=None,
            learning_progress=0.0,
        )
    )
    metadata["scenarios"]["8"] = _v3_scenario(
        attempts=0,
        solved=0,
        solve_rate=0.0,
        last_attempted_iter=0,
        last_iteration_solve_rate=None,
        learning_progress=0.0,
    )

    update_pool_metadata(
        metadata,
        [],
        current_iter=5,
        selected_scenario_ids=[7, 8],
    )

    for scenario_id in ("7", "8"):
        scenario = metadata["scenarios"][scenario_id]
        assert scenario["times_attempted"] == 1
        assert scenario["times_solved"] == 0
        assert scenario["last_attempted_iter"] == 5
        assert scenario["solve_rate_delta"] == 0.0
        assert scenario["staleness"] == 0.0
        assert 0.0 < scenario["uncertainty"] < 1.0


def test_loading_sanitizes_non_finite_learning_signals(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(tmp_path)
    path = tmp_path / "pool_metadata.json"
    scenario = _v3_scenario(attempts=4, solved=2)
    scenario.update(
        {
            "last_iteration_solve_rate": "nan",
            "solve_rate_delta": float("inf"),
            "learning_progress": float("-inf"),
            "uncertainty": float("nan"),
            "staleness": float("nan"),
        }
    )
    path.write_text(
        json.dumps(_metadata(scenario)),
        encoding="utf-8",
    )

    loaded = initialize_pool_metadata(
        transitions,
        path,
        current_iter=5,
        stale_after_iterations=3,
    )
    loaded_scenario = loaded["scenarios"]["7"]

    assert loaded_scenario["last_iteration_solve_rate"] == 0.0
    assert loaded_scenario["solve_rate_delta"] == 0.0
    assert loaded_scenario["learning_progress"] == 0.0
    for field in ("uncertainty", "staleness"):
        assert math.isfinite(float(loaded_scenario[field]))
        assert 0.0 <= float(loaded_scenario[field]) <= 1.0


def test_pool_state_rejects_unsupported_schema(tmp_path: Path) -> None:
    transitions = _transitions_csv(tmp_path)
    path = tmp_path / "pool_metadata.json"
    metadata = _metadata(_v3_scenario())
    metadata["schema_version"] = 1
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported pool metadata"):
        initialize_pool_metadata(transitions, path)


@pytest.mark.parametrize("value", [0, -1])
def test_pool_state_rejects_invalid_staleness_threshold(
    tmp_path: Path,
    value: int,
) -> None:
    transitions = _transitions_csv(tmp_path)

    with pytest.raises(ValueError, match="stale_after_iterations"):
        initialize_pool_metadata(
            transitions,
            tmp_path / "pool_metadata.json",
            stale_after_iterations=value,
        )
