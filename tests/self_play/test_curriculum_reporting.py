from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from grid_topology_ai.config.pool import CurriculumSamplingConfig
from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    sha256_file,
)
from grid_topology_ai.self_play.curriculum_reporting import (
    persist_curriculum_pool_state,
    prepare_curriculum_sampling,
    record_curriculum_sampling,
)


def _scenario(
    *,
    difficulty: str,
    attempts: int,
    solved: int,
    solve_rate: float,
    last_attempted: int,
    learning_progress: float,
    uncertainty: float,
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


def _pool_metadata() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "transitions_csv": "transitions.csv",
        "last_updated_iteration": 5,
        "scenarios": {
            "1": _scenario(
                difficulty="hard",
                attempts=0,
                solved=0,
                solve_rate=0.0,
                last_attempted=0,
                learning_progress=0.1,
                uncertainty=1.0,
            ),
            "2": _scenario(
                difficulty="simple",
                attempts=4,
                solved=0,
                solve_rate=0.0,
                last_attempted=1,
                learning_progress=0.2,
                uncertainty=0.8,
            ),
            "3": _scenario(
                difficulty="medium",
                attempts=4,
                solved=2,
                solve_rate=0.5,
                last_attempted=4,
                learning_progress=0.3,
                uncertainty=0.6,
            ),
            "4": _scenario(
                difficulty="hard",
                attempts=4,
                solved=4,
                solve_rate=1.0,
                last_attempted=5,
                learning_progress=0.4,
                uncertainty=0.4,
            ),
        },
    }


def _config() -> CurriculumSamplingConfig:
    return CurriculumSamplingConfig(
        never_solved_min_fraction=0.25,
        hard_min_fraction=0.50,
        simple_max_fraction=1.0,
        frontier_max_fraction=1.0,
        stale_after_iterations=3,
    )


def _prepare(tmp_path: Path):
    metadata = _pool_metadata()
    report_path = tmp_path / "iteration_0006" / "curriculum_sampling.json"
    prepared = prepare_curriculum_sampling(
        pool_metadata=metadata,
        n_scenarios=4,
        current_iter=6,
        scenario_sampling_seed=123,
        config=_config(),
        report_path=report_path,
    )
    assert prepared is not None
    return metadata, prepared, report_path


def test_prepare_curriculum_sampling_writes_coverage_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata, prepared, report_path = _prepare(tmp_path)

    assert report_path.is_file()
    assert load_json(report_path) == prepared.report
    assert set(prepared.scenario_ids) == {1, 2, 3, 4}

    report = prepared.report
    assert report["iteration"] == 6
    assert report["scenario_sampling_seed"] == 123
    assert report["pool_count"] == 4
    assert report["pool_by_difficulty"] == {
        "hard": 2,
        "medium": 1,
        "simple": 1,
    }
    assert report["selected_by_difficulty"] == {
        "hard": 2,
        "medium": 1,
        "simple": 1,
    }

    assert report["never_solved"] == {
        "available": 1,
        "target": 1,
        "selected": 1,
        "shortfall": 0,
        "fraction": 0.25,
        "min_fraction": 0.25,
    }
    assert report["hard"] == {
        "available": 2,
        "target": 2,
        "selected": 2,
        "shortfall": 0,
        "fraction": 0.5,
        "min_fraction": 0.5,
    }
    assert report["simple"]["selected"] == 1
    assert report["simple"]["fraction"] == 0.25
    assert report["simple"]["max_fraction"] == 1.0
    assert report["frontier"]["selected"] == 1
    assert report["frontier"]["fraction"] == 0.25
    assert report["frontier"]["max_fraction"] == 1.0
    assert report["cap_relaxations"] == []

    unvisited = report["unvisited_after_n_iterations"]
    assert unvisited["threshold"] == 3
    assert unvisited["pool_count"] == 2
    assert unvisited["pool_fraction"] == 0.5
    assert unvisited["selected_count"] == 2
    assert unvisited["selected_fraction"] == 0.5
    assert unvisited["by_difficulty"] == {
        "hard": {"pool_count": 1, "selected_count": 1},
        "medium": {"pool_count": 0, "selected_count": 0},
        "simple": {"pool_count": 1, "selected_count": 1},
    }

    means = report["selected_signal_means"]
    assert means["priority"] == pytest.approx(1.4125)
    assert means["learning_progress"] == pytest.approx(0.25)
    assert means["uncertainty"] == pytest.approx(0.70)
    assert means["staleness"] == pytest.approx(0.75)

    output = capsys.readouterr().out
    assert "Curriculum sample: 4 scenarios" in output
    assert "never-solved: 1 / target 1" in output
    assert "hard:         2 / target 2" in output
    assert "stale >= 3:" in output
    assert str(report_path) in output

    assert metadata["curriculum_sampling"] == asdict(_config())
    assert metadata["scenarios"]["1"]["priority"] == 0.05
    assert "priority_components" not in metadata["scenarios"]["1"]


def test_record_curriculum_sampling_updates_metadata_and_curve(
    tmp_path: Path,
) -> None:
    _, prepared, report_path = _prepare(tmp_path)
    metadata_path = tmp_path / "iteration_0006" / "metadata.json"
    save_json(
        {
            "iteration": 6,
            "hashes": {"existing_sha256": "kept"},
            "extra": {"existing": "kept"},
        },
        metadata_path,
    )
    learning_curve_row: dict[str, object] = {
        "iteration": 6,
        "n_fresh": 4,
    }

    returned = record_curriculum_sampling(
        prepared=prepared,
        selected_scenario_ids=prepared.scenario_ids,
        iteration_metadata_path=metadata_path,
        learning_curve_row=learning_curve_row,
    )

    assert returned == prepared.report
    report_sha256 = sha256_file(report_path)
    metadata = load_json(metadata_path)
    assert metadata["hashes"] == {
        "existing_sha256": "kept",
        "curriculum_sampling_sha256": report_sha256,
    }
    assert metadata["extra"]["existing"] == "kept"
    assert metadata["extra"]["curriculum_sampling_path"] == str(
        report_path
    )
    assert metadata["extra"]["curriculum_sampling_sha256"] == (
        report_sha256
    )
    assert metadata["extra"]["curriculum_sampling"] == prepared.report

    assert learning_curve_row == {
        "iteration": 6,
        "n_fresh": 4,
        "curriculum_never_solved_fraction": 0.25,
        "curriculum_hard_fraction": 0.5,
        "curriculum_simple_fraction": 0.25,
        "curriculum_frontier_fraction": 0.25,
        "curriculum_unvisited_pool_fraction": 0.5,
        "curriculum_unvisited_selected_fraction": 0.5,
        "curriculum_never_solved_shortfall": 0,
        "curriculum_hard_shortfall": 0,
        "curriculum_cap_relaxation_count": 0,
        "curriculum_mean_priority": pytest.approx(1.4125),
        "curriculum_mean_learning_progress": pytest.approx(0.25),
        "curriculum_mean_uncertainty": pytest.approx(0.70),
        "curriculum_mean_staleness": pytest.approx(0.75),
    }


def test_record_rejects_diagnostics_for_different_scenario_ids(
    tmp_path: Path,
) -> None:
    _, prepared, _ = _prepare(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="do not match the sampled scenario IDs",
    ):
        record_curriculum_sampling(
            prepared=prepared,
            selected_scenario_ids=tuple(reversed(prepared.scenario_ids)),
            iteration_metadata_path=tmp_path / "metadata.json",
            learning_curve_row={"iteration": 6},
        )


def test_record_requires_complete_iteration_metadata(tmp_path: Path) -> None:
    _, prepared, _ = _prepare(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    save_json({"extra": []}, metadata_path)

    with pytest.raises(ValueError, match="metadata is incomplete"):
        record_curriculum_sampling(
            prepared=prepared,
            selected_scenario_ids=prepared.scenario_ids,
            iteration_metadata_path=metadata_path,
            learning_curve_row={"iteration": 6},
        )


def test_legacy_pool_skips_curriculum_report(tmp_path: Path) -> None:
    metadata = _pool_metadata()
    metadata["schema_version"] = 2
    report_path = tmp_path / "curriculum_sampling.json"

    prepared = prepare_curriculum_sampling(
        pool_metadata=metadata,
        n_scenarios=4,
        current_iter=6,
        scenario_sampling_seed=123,
        config=_config(),
        report_path=report_path,
    )

    assert prepared is None
    assert not report_path.exists()
    assert metadata["curriculum_sampling"] == asdict(_config())


def test_persist_curriculum_pool_state_refreshes_and_saves(
    tmp_path: Path,
) -> None:
    metadata = _pool_metadata()
    path = tmp_path / "pool_metadata.json"

    persist_curriculum_pool_state(
        pool_metadata=metadata,
        current_iter=6,
        config=_config(),
        path=path,
    )

    assert path.is_file()
    assert load_json(path) == metadata
    assert metadata["curriculum_sampling"] == asdict(_config())

    first = metadata["scenarios"]["1"]
    assert first["staleness"] == 1.0
    assert first["priority"] == pytest.approx(1.6)
    assert first["priority_components"] == {
        "total": pytest.approx(1.6),
        "learning_progress": pytest.approx(0.1),
        "uncertainty": pytest.approx(0.75),
        "staleness": pytest.approx(0.5),
        "frontier": pytest.approx(0.0),
        "difficulty_bonus": pytest.approx(0.2),
    }
