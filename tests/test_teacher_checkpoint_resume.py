from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import grid_topology_ai.teacher_runtime as teacher
import pytest


def _initialize_light_teacher_smoke() -> None:
    """Import the canonical runtime modules in a freshly spawned interpreter."""

    import grid_topology_ai.search.mcts  # noqa: F401
    import grid_topology_ai.search.teacher  # noqa: F401
    import grid_topology_ai.teacher_runtime  # noqa: F401


def _process_light_teacher_smoke(scenario_ids: list[int]) -> list[dict]:
    """Exercise importable Light teacher/search symbols across the worker boundary."""

    from grid_topology_ai.search.mcts import MCTSConfig
    from grid_topology_ai.search.teacher import ImpactBeamSearchConfig
    from grid_topology_ai.teacher_runtime import process_scenario_batch

    assert process_scenario_batch.__module__ == "grid_topology_ai.teacher_runtime"
    assert MCTSConfig(num_simulations=1).num_simulations == 1
    assert ImpactBeamSearchConfig(max_depth=1).max_depth == 1
    return [
        {
            "scenario_id": int(scenario_id),
            "worker_pid": os.getpid(),
        }
        for scenario_id in scenario_ids
    ]


def _completed_result(scenario_id: int) -> dict:
    return {
        "scenario_id": scenario_id,
        "ok": True,
        "reason": None,
        "rows": [{"scenario_id": scenario_id, "step": 0, "value": scenario_id * 10}],
    }


def test_real_spawn_workers_cross_canonical_light_teacher_boundary() -> None:
    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=_initialize_light_teacher_smoke,
        )
        for _ in range(2)
    ]
    try:
        futures = [
            executor.submit(
                teacher._run_timed_batch,
                _process_light_teacher_smoke,
                [scenario_id],
            )
            for executor, scenario_id in zip(executors, (202, 101))
        ]
        results = [future.result(timeout=60)[0][0] for future in futures]
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)

    assert len({result["worker_pid"] for result in results}) == 2
    assert sorted(result["scenario_id"] for result in results) == [101, 202]
    assert len({result["scenario_id"] for result in results}) == len(results)


def test_resume_restores_committed_work_and_ignores_incomplete_work(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    records = [
        {
            "scenario_id": 1,
            "ok": False,
            "reason": "exception",
            "rows": [],
        },
        {
            "scenario_id": 2,
            "ok": False,
            "reason": "no_teacher_action_found",
            "rows": [],
        },
        {
            "scenario_id": 3,
            "ok": True,
            "reason": None,
            "rows": [],
        },
        {
            "scenario_id": 4,
            "ok": True,
            "reason": None,
            "rows": [{"state_id": "state-4"}],
        },
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
        + '{"scenario_id":5,"ok":true',
        encoding="utf-8",
    )

    restored = teacher.load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3, 4, 5],
    )

    assert restored[1]["reason"] == "exception"
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert 3 not in restored
    assert restored[4]["ok"] is True
    assert 5 not in restored


def test_resume_ignores_scenarios_outside_current_run(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "scenario_id": 99,
                "ok": True,
                "reason": None,
                "rows": [{"state_id": "other-run"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert teacher.load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3],
    ) == {}


def test_teacher_restart_completes_only_pending_scenarios(tmp_path) -> None:
    scenario_ids = [30, 10, 20]
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    config_path = tmp_path / "teacher_checkpoint_config.json"
    config = {
        "scenario_ids": scenario_ids,
        "task_config": {"depth": 1, "beam_width": 2, "disable_cache": False},
        "source_identity": {"dataset": "restart-smoke"},
    }

    teacher.ensure_teacher_checkpoint_config(config_path, config)
    teacher.append_scenario_checkpoint(checkpoint_path, _completed_result(30))
    with checkpoint_path.open("ab") as checkpoint_file:
        checkpoint_file.write(b'{"scenario_id":10,"ok":true')

    resumed_config = dict(config)
    resumed_config["task_config"] = dict(config["task_config"], disable_cache=True)
    teacher.ensure_teacher_checkpoint_config(config_path, resumed_config)
    changed_config = dict(config)
    changed_config["task_config"] = dict(config["task_config"], depth=2)
    with pytest.raises(RuntimeError, match="does not match"):
        teacher.ensure_teacher_checkpoint_config(config_path, changed_config)

    restored = teacher.load_scenario_checkpoints(checkpoint_path, scenario_ids)
    assert sorted(restored) == [30]
    pending = [scenario_id for scenario_id in scenario_ids if scenario_id not in restored]
    assert pending == [10, 20]

    processed_after_restart = []
    for scenario_id in pending:
        processed_after_restart.append(scenario_id)
        teacher.append_scenario_checkpoint(
            checkpoint_path, _completed_result(scenario_id)
        )

    final = teacher.load_scenario_checkpoints(checkpoint_path, scenario_ids)
    rows, saved, skipped = teacher.collect_rows_from_checkpoints(final)
    uninterrupted = teacher.collect_rows_from_checkpoints(
        {scenario_id: _completed_result(scenario_id) for scenario_id in scenario_ids}
    )

    assert processed_after_restart == [10, 20]
    assert sorted(final) == [10, 20, 30]
    assert len(final) == len(scenario_ids)
    assert [row["scenario_id"] for row in rows] == [10, 20, 30]
    assert (rows, saved, skipped) == uninterrupted
