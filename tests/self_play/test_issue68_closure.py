from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from grid_topology_ai.config import (
    EvaluationConfig,
    ReplayBufferConfig,
    SelfPlayConfig,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import final_test as final_test_module
from grid_topology_ai.self_play import lineage_artifacts
from grid_topology_ai.self_play import pipeline as pipeline_module
from grid_topology_ai.self_play.final_test import (
    FinalTestEvaluation,
    load_final_test_evaluation,
    run_final_test_evaluation,
)
from grid_topology_ai.self_play.iteration_split import (
    prepare_physical_iteration_split,
)
from grid_topology_ai.self_play.paths import SelfPlayPaths
from grid_topology_ai.self_play.physical_lineage import (
    PHYSICAL_LINEAGE_FINGERPRINT_FIELD,
    PhysicalLineage,
)
from grid_topology_ai.self_play.replay import RollingReplayBuffer
from grid_topology_ai.self_play.split_integrity import (
    physical_split_source_hashes,
)


def _lineage(index: int) -> PhysicalLineage:
    return PhysicalLineage.build(
        base_case_id="case118",
        load_profile_id=f"load-{index}",
        contingency_family_id=[f"branch:{index}"],
    )


def _row(
    index: int,
    *,
    scenario_id: int | None = None,
    step: int = 0,
    replay_iteration: int = 1,
) -> dict[str, object]:
    lineage = _lineage(index)
    resolved_scenario = index if scenario_id is None else scenario_id
    return {
        "state_id": f"state-{index}-{resolved_scenario}-{step}",
        "episode_id": f"episode-{index}-{resolved_scenario}",
        "scenario_id": resolved_scenario,
        "step": step,
        "replay_iteration": replay_iteration,
        "difficulty_class": "medium",
        "outcome_class": "solved",
        **lineage.as_dict(),
    }


def _paths(tmp_path: Path) -> SelfPlayPaths:
    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text("scenario_id\n1\n2\n3\n4\n", encoding="utf-8")
    return SelfPlayPaths(
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        pool_transitions_csv=pool_csv,
        pool_raw_dir=tmp_path / "pool_raw",
        pool_metadata=tmp_path / "run" / "pool.json",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "eval_raw",
        final_test_csv=tmp_path / "final.csv",
        final_test_raw_dir=tmp_path / "final_raw",
        bootstrap_checkpoint=tmp_path / "bootstrap.pt",
        bootstrap_metrics=tmp_path / "bootstrap.json",
        best_checkpoint=tmp_path / "run" / "best.pt",
        best_metrics=tmp_path / "run" / "best.json",
    )


def _buffer(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> RollingReplayBuffer:
    replay = RollingReplayBuffer(
        save_dir=tmp_path / "replay",
        config=ReplayBufferConfig(
            max_size=100,
            min_size_to_train=1,
            fresh_fraction=0.5,
            random_seed=7,
        ),
    )
    replay.buffer = rows
    return replay


def _prepare(
    *,
    replay: RollingReplayBuffer,
    paths: SelfPlayPaths,
    iteration: int,
) -> dict[str, Any]:
    iteration_dir = paths.iteration_dir(iteration)
    _, metadata = prepare_physical_iteration_split(
        replay_buffer=replay,
        paths=paths,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        iteration=iteration,
        split_seed=17,
        sampling_seed=100 + iteration,
        validation_fraction=0.25,
        min_validation_lineages=1,
        n_examples=4,
        fresh_fraction=0.5,
        train_batch_path=iteration_dir / "train_batch.csv",
        train_examples_path=iteration_dir / "train_examples.csv",
        validation_examples_path=iteration_dir / "validation_examples.csv",
        metadata_path=iteration_dir / "train_validation_split.json",
    )
    return metadata


def test_validation_snapshot_survives_replay_eviction(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    rows = [
        _row(index, step=step)
        for index in range(1, 5)
        for step in range(2)
    ]
    replay = _buffer(tmp_path, rows)
    first_metadata = _prepare(
        replay=replay,
        paths=paths,
        iteration=1,
    )
    first_validation = pd.read_csv(paths.physical_validation_snapshot)
    validation_fingerprints = set(
        first_validation[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
    )
    assert validation_fingerprints

    replay.buffer = [
        row
        for row in rows
        if row[PHYSICAL_LINEAGE_FINGERPRINT_FIELD]
        not in validation_fingerprints
    ]
    second_metadata = _prepare(
        replay=replay,
        paths=paths,
        iteration=2,
    )
    second_validation = pd.read_csv(paths.physical_validation_snapshot)

    pd.testing.assert_frame_equal(first_validation, second_validation)
    assert first_metadata["validation_examples"] == len(first_validation)
    assert second_metadata["validation_examples"] == len(first_validation)
    assert second_metadata["active_validation_examples"] == 0
    assert second_metadata["active_validation_lineages"] == 0
    assert second_metadata["validation_snapshot_created_iteration"] == 1
    assert second_metadata["validation_snapshot_last_updated_iteration"] == 2


def test_changed_scenario_lineage_is_rejected_before_manifest_update(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    rows = [_row(index) for index in range(1, 5)]
    replay = _buffer(tmp_path, rows)
    _prepare(replay=replay, paths=paths, iteration=1)
    manifest_before = paths.physical_split_manifest.read_bytes()

    changed = {
        **_row(99, scenario_id=1, replay_iteration=2),
        **_lineage(99).as_dict(),
    }
    replay.buffer = [changed, _row(2, replay_iteration=2)]

    with pytest.raises(ValueError, match="changed physical lineage"):
        _prepare(replay=replay, paths=paths, iteration=2)

    assert paths.physical_split_manifest.read_bytes() == manifest_before


def test_physical_split_source_hashes_cover_raw_lineage_files(
    tmp_path: Path,
) -> None:
    transitions = tmp_path / "pool.csv"
    transitions.write_text("scenario_id\n1\n", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, content in {
        "bus_data.parquet": b"bus",
        "branch_data.parquet": b"branch",
        "gen_data.parquet": b"gen",
    }.items():
        (raw_dir / name).write_bytes(content)

    hashes = physical_split_source_hashes(
        transitions_csv=transitions,
        raw_dir=raw_dir,
    )

    assert set(hashes) == {
        "pool_transitions",
        "pool_raw:bus_data.parquet",
        "pool_raw:branch_data.parquet",
        "pool_raw:gen_data.parquet",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_scenario_grouping_normalizes_each_raw_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "scenario": [value // 2 for value in range(100)],
            "value": list(range(100)),
        }
    )
    calls = 0
    original = lineage_artifacts._coerce_scenario_id

    def counted(value: object, *, source: str) -> int:
        nonlocal calls
        calls += 1
        return original(value, source=source)

    monkeypatch.setattr(
        lineage_artifacts,
        "_coerce_scenario_id",
        counted,
    )
    groups = lineage_artifacts._scenario_groups(
        frame,
        range(50),
        source=Path("raw.parquet"),
    )

    assert calls == len(frame)
    assert len(groups) == 50
    assert all(len(group) == 2 for group in groups.values())


def _prepare_final_test_paths(tmp_path: Path) -> SelfPlayPaths:
    paths = _paths(tmp_path)
    paths.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.best_checkpoint.write_bytes(b"selected-best")
    paths.best_metrics.write_text(
        json.dumps({"selection_metric": 0.4}),
        encoding="utf-8",
    )
    paths.final_test_csv.write_text(
        "scenario_id\n7\n8\n",
        encoding="utf-8",
    )
    paths.final_test_raw_dir.mkdir(parents=True, exist_ok=True)
    return paths


def test_final_test_report_seals_and_reuses_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_final_test_paths(tmp_path)
    config = EvaluationConfig()
    monkeypatch.setattr(
        final_test_module,
        "require_metrics_pf_alg",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        final_test_module,
        "require_metrics_physics_config",
        lambda *args, **kwargs: None,
    )

    def fake_run_evaluate(**kwargs: Any) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"physically_secure_rate_requested": 0.5}
        (output_dir / config.output_json_name).write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        (output_dir / config.output_csv_name).write_text(
            "scenario_id,solved\n7,1\n8,0\n",
            encoding="utf-8",
        )
        return metrics

    monkeypatch.setattr(
        final_test_module,
        "run_evaluate",
        fake_run_evaluate,
    )
    first = run_final_test_evaluation(
        paths=paths,
        checkpoint=paths.best_checkpoint,
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["run_sealed"] is True

    loaded = load_final_test_evaluation(
        paths=paths,
        config=config,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    assert loaded is not None
    assert loaded.metrics == first.metrics

    with pytest.raises(FileExistsError, match="run is sealed"):
        run_final_test_evaluation(
            paths=paths,
            checkpoint=paths.best_checkpoint,
            config=config,
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )

    paths.final_test_csv.write_text(
        "scenario_id\n7\n8\n9\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transitions hash mismatch"):
        load_final_test_evaluation(
            paths=paths,
            config=config,
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )


def test_pipeline_rejects_more_iterations_after_final_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        SelfPlayConfig.load("configs/self_play_loop_pilot.yaml"),
        n_iterations=2,
    )
    paths = _paths(tmp_path)
    sealed = FinalTestEvaluation(
        metrics={"physically_secure_rate_requested": 0.5},
        metrics_path=tmp_path / "metrics.json",
        results_path=tmp_path / "results.csv",
        report_path=tmp_path / "report.json",
        checkpoint=paths.best_checkpoint,
    )
    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_state",
        lambda **kwargs: SimpleNamespace(
            start_iteration=2,
            completed_iterations=(1,),
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_resume_artifacts",
        lambda paths: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "initialize_best_state",
        lambda **kwargs: SimpleNamespace(
            checkpoint=paths.best_checkpoint,
            metrics={},
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_pf_alg",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_metrics_physics_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_final_test_evaluation",
        lambda **kwargs: sealed,
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_yaml",
        lambda **kwargs: pytest.fail("sealed run rewrote its config"),
    )

    with pytest.raises(RuntimeError, match="sealed by final-test"):
        pipeline_module.run_self_play_pipeline(
            pipeline_module.PipelineRequest(
                config=config,
                raw_config={},
                paths=paths,
                resume=True,
            )
        )