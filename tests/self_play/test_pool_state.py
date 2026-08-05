import json
from pathlib import Path

import pandas as pd

from grid_topology_ai.self_play.pool_sampling import (
    compute_priority,
)
from grid_topology_ai.self_play.pool_state import (
    SCHEMA_VERSION,
    initialize_pool_metadata,
    load_json,
    update_and_save_pool_metadata,
    update_pool_metadata,
)


def _transitions_csv(
    tmp_path: Path,
    rows: list[tuple[int, str]],
) -> Path:
    path = tmp_path / "transitions.csv"
    pd.DataFrame(
        rows,
        columns=["scenario_id", "difficulty_class"],
    ).to_csv(path, index=False)
    return path


def _scenario(
    *,
    attempts=0,
    solved=0,
    solve_rate=0.0,
    last_attempted=0,
    difficulty="medium",
):
    return {
        "difficulty_class": difficulty,
        "times_attempted": attempts,
        "times_solved": solved,
        "solve_rate": solve_rate,
        "last_attempted_iter": last_attempted,
        "last_solved_iter": None,
        "avg_steps_when_solved": None,
        "last_iteration_solve_rate": None,
        "solve_rate_delta": 0.0,
        "learning_progress": 0.0,
        "uncertainty": 1.0,
        "staleness": 1.0 if attempts == 0 else 0.0,
        "priority": compute_priority(
            solve_rate,
            attempts,
            last_attempted,
            last_attempted,
            difficulty,
        ),
    }


def _metadata():
    return {
        "schema_version": SCHEMA_VERSION,
        "transitions_csv": "transitions.csv",
        "last_updated_iteration": 0,
        "scenarios": {
            "1": _scenario(),
            "2": _scenario(difficulty="hard"),
            "3": _scenario(difficulty="simple"),
        },
    }


def test_initialize_pool_metadata_preserves_schema(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(
        tmp_path,
        [(2, "hard"), (1, "simple"), (2, "hard")],
    )
    metadata = initialize_pool_metadata(
        transitions,
        tmp_path / "pool_metadata.json",
    )

    assert set(metadata) == {
        "schema_version",
        "transitions_csv",
        "last_updated_iteration",
        "scenarios",
    }
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert list(metadata["scenarios"]) == ["1", "2"]
    assert set(metadata["scenarios"]["1"]) == {
        "difficulty_class",
        "times_attempted",
        "times_solved",
        "solve_rate",
        "last_attempted_iter",
        "last_solved_iter",
        "avg_steps_when_solved",
        "last_iteration_solve_rate",
        "solve_rate_delta",
        "learning_progress",
        "uncertainty",
        "staleness",
        "priority",
    }


def test_initialize_pool_metadata_does_not_overwrite_existing_file(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(
        tmp_path,
        [(1, "simple")],
    )
    path = tmp_path / "pool_metadata.json"
    existing = {
        "schema_version": SCHEMA_VERSION,
        "last_updated_iteration": 9,
        "scenarios": {"99": _scenario()},
    }
    path.write_text(json.dumps(existing), encoding="utf-8")

    assert (
        initialize_pool_metadata(
            transitions,
            path,
            overwrite=False,
        )
        == existing
    )


def test_initialize_pool_metadata_overwrites_when_requested(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(
        tmp_path,
        [(1, "simple")],
    )
    path = tmp_path / "pool_metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios": {"99": _scenario()},
            }
        ),
        encoding="utf-8",
    )

    metadata = initialize_pool_metadata(
        transitions,
        path,
        overwrite=True,
    )

    assert list(metadata["scenarios"]) == ["1"]


def test_update_pool_metadata_tracks_attempts_and_solves() -> None:
    metadata = _metadata()

    update_pool_metadata(
        metadata,
        [{"scenario_id": 1, "solved": True, "steps": 2}],
        4,
    )

    scenario = metadata["scenarios"]["1"]
    assert scenario["times_attempted"] == 1
    assert scenario["times_solved"] == 1
    assert scenario["last_attempted_iter"] == 4
    assert scenario["last_solved_iter"] == 4
    assert scenario["avg_steps_when_solved"] == 2.0
    assert scenario["last_iteration_solve_rate"] == 1.0
    assert scenario["solve_rate_delta"] == 1.0
    assert scenario["learning_progress"] == 1.0
    assert scenario["staleness"] == 0.0
    assert 0.0 < scenario["uncertainty"] < 1.0


def test_selected_scenario_without_examples_is_still_attempted() -> None:
    metadata = _metadata()

    update_pool_metadata(
        metadata,
        [{"scenario_id": 1, "solved": False, "steps": 3}],
        5,
        selected_scenario_ids=[1, 2],
    )

    missing = metadata["scenarios"]["2"]
    assert missing["times_attempted"] == 1
    assert missing["last_attempted_iter"] == 5
    assert missing["times_solved"] == 0
    assert missing["staleness"] == 0.0
    assert missing["uncertainty"] < 1.0


def test_update_preserves_unselected_scenarios() -> None:
    metadata = _metadata()

    update_pool_metadata(
        metadata,
        [{"scenario_id": 1, "solved": False, "steps": 3}],
        5,
        selected_scenario_ids=[1],
    )

    untouched = metadata["scenarios"]["3"]
    assert untouched["times_attempted"] == 0
    assert untouched["last_attempted_iter"] == 0
    assert untouched["staleness"] == 1.0


def test_update_and_save_pool_metadata_persists_result(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    path = tmp_path / "pool_metadata.json"

    returned = update_and_save_pool_metadata(
        metadata,
        [{"scenario_id": 1, "solved": True, "steps": 2}],
        2,
        path,
        selected_scenario_ids=[1],
    )

    assert returned["last_updated_iteration"] == 2
    assert returned["schema_version"] == SCHEMA_VERSION
    assert load_json(path) == returned


def test_pool_state_migrates_existing_v2_metadata(
    tmp_path: Path,
) -> None:
    transitions = _transitions_csv(
        tmp_path,
        [(1, "simple")],
    )
    path = tmp_path / "pool_metadata.json"
    legacy = _metadata()
    legacy["schema_version"] = 2

    for scenario in legacy["scenarios"].values():
        for field in (
            "last_iteration_solve_rate",
            "solve_rate_delta",
            "learning_progress",
            "uncertainty",
            "staleness",
        ):
            scenario.pop(field)

    path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = initialize_pool_metadata(
        transitions,
        path,
    )

    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["scenarios"]["1"]["times_attempted"] == 0
    assert migrated["scenarios"]["1"]["uncertainty"] == 1.0
    assert migrated["scenarios"]["1"]["staleness"] == 1.0
    assert load_json(path) == migrated
