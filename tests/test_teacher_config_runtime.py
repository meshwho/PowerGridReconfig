from __future__ import annotations

import hashlib
import json

import pytest

from grid_topology_ai.teacher_config import (
    checkpoint_config_payload,
    ensure_teacher_checkpoint_config,
    load_teacher_task_config,
    semantic_teacher_task_config,
    split_teacher_task_config,
    teacher_run_id,
    teacher_source_identity,
)


def _task_config(**overrides):
    config = {
        "depth": 5,
        "beam_width": 20,
        "candidate_pool": 160,
        "top_k": 70,
        "physics_config_fingerprint": "physics-v1",
        "disable_cache": False,
        "clear_caches_every": 2,
        "max_worker_memory_mb": 1000.0,
        "min_free_system_memory_mb": 512.0,
        "memory_registry_max_age_sec": 120.0,
        "max_tasks_per_child": 0,
        "print_memory_events": False,
        "auto_worker_memory_mb": 1200.0,
        "auto_worker_memory_reserve_mb": 2048.0,
        "auto_worker_cpu_util_target": 85.0,
        "auto_worker_cpu_mode": "logical",
        "auto_worker_cpu_fraction": 0.85,
        "auto_worker_max": 4,
    }
    config.update(overrides)
    return config


def _checkpoint_config(task_config, *, source_identity=None):
    config = {
        "checkpoint_version": 2,
        "raw_dir": "D:/data/raw",
        "transitions_path": "D:/data/transitions.csv",
        "scenario_ids": [1, 2, 3],
        "task_config": task_config,
    }
    if source_identity is not None:
        config["source_identity"] = source_identity
    return config


def _source_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "bus_data.parquet").write_bytes(b"bus-v1")
    (raw_dir / "branch_data.parquet").write_bytes(b"branch-v1")
    (raw_dir / "gen_data.parquet").write_bytes(b"gen-v1")
    transitions = tmp_path / "transitions.csv"
    transitions.write_bytes(b"scenario\n1\n2\n3\n")
    return raw_dir, transitions


def test_task_config_split_keeps_cache_and_memory_controls_runtime_only():
    semantic, runtime = split_teacher_task_config(_task_config())

    assert semantic["depth"] == 5
    assert semantic["physics_config_fingerprint"] == "physics-v1"
    assert "disable_cache" not in semantic

    assert runtime["disable_cache"] is False
    assert runtime["clear_caches_every"] == 2
    assert runtime["max_worker_memory_mb"] == 1000.0
    assert runtime["min_free_system_memory_mb"] == 512.0
    assert runtime["max_tasks_per_child"] == 0
    assert runtime["auto_worker_cpu_mode"] == "logical"
    assert not (set(semantic) & set(runtime))


def test_cache_toggle_and_memory_changes_do_not_change_semantic_identity():
    left = _task_config()
    right = _task_config(
        disable_cache=True,
        clear_caches_every=0,
        max_worker_memory_mb=2400.0,
        min_free_system_memory_mb=2048.0,
        max_tasks_per_child=20,
        print_memory_events=True,
        auto_worker_memory_mb=1800.0,
        auto_worker_max=8,
    )

    assert semantic_teacher_task_config(left) == semantic_teacher_task_config(right)


def test_legacy_checkpoint_accepts_cache_and_runtime_changes_on_resume(tmp_path):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    original = _checkpoint_config(_task_config())
    config_path.write_text(json.dumps(original), encoding="utf-8")

    resumed = _checkpoint_config(
        _task_config(
            disable_cache=True,
            clear_caches_every=0,
            max_worker_memory_mb=2400.0,
            min_free_system_memory_mb=1024.0,
            max_tasks_per_child=10,
        )
    )

    ensure_teacher_checkpoint_config(config_path, resumed)
    assert json.loads(config_path.read_text(encoding="utf-8")) == original


def test_checkpoint_rejects_semantic_search_change(tmp_path):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(_task_config()),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        ensure_teacher_checkpoint_config(
            config_path,
            _checkpoint_config(_task_config(depth=6)),
        )


def test_teacher_source_identity_changes_with_raw_content(tmp_path):
    raw_dir, transitions = _source_files(tmp_path)
    original = teacher_source_identity(raw_dir, transitions)

    (raw_dir / "branch_data.parquet").write_bytes(b"branch-v2")
    changed = teacher_source_identity(raw_dir, transitions)

    assert changed != original


def test_teacher_source_identity_changes_with_transitions_content(tmp_path):
    raw_dir, transitions = _source_files(tmp_path)
    original = teacher_source_identity(raw_dir, transitions)

    transitions.write_bytes(b"scenario\n1\n2\n4\n")
    changed = teacher_source_identity(raw_dir, transitions)

    assert changed != original


def test_checkpoint_rejects_changed_source_content_at_same_paths(tmp_path):
    raw_dir, transitions = _source_files(tmp_path)
    config_path = tmp_path / "teacher_checkpoint_config.json"
    original_identity = teacher_source_identity(raw_dir, transitions)
    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(
            _task_config(),
            source_identity=original_identity,
        ),
    )

    (raw_dir / "bus_data.parquet").write_bytes(b"bus-v2")
    changed_identity = teacher_source_identity(raw_dir, transitions)

    with pytest.raises(RuntimeError, match="does not match"):
        ensure_teacher_checkpoint_config(
            config_path,
            _checkpoint_config(
                _task_config(),
                source_identity=changed_identity,
            ),
        )


def test_new_checkpoint_persists_cache_toggle_in_runtime_section(tmp_path):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(_task_config()),
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "task_config" not in payload
    assert payload["semantic_task_config"]["depth"] == 5
    assert "disable_cache" not in payload["semantic_task_config"]
    assert payload["runtime_task_config"]["disable_cache"] is False
    assert payload["runtime_task_config"]["max_worker_memory_mb"] == 1000.0


def test_new_run_id_is_independent_of_cache_and_runtime_controls(tmp_path):
    states_dir = tmp_path / "run" / "states"
    states_dir.mkdir(parents=True)

    first = teacher_run_id(states_dir, _task_config())
    second = teacher_run_id(
        states_dir,
        _task_config(
            disable_cache=True,
            clear_caches_every=0,
            max_worker_memory_mb=3000.0,
            min_free_system_memory_mb=4096.0,
            max_tasks_per_child=50,
        ),
    )

    assert first == second
    assert first != teacher_run_id(states_dir, _task_config(depth=6))


def test_legacy_checkpoint_preserves_historical_full_config_run_id(tmp_path):
    states_dir = tmp_path / "legacy" / "states"
    states_dir.mkdir(parents=True)
    legacy_task_config = _task_config(
        disable_cache=False,
        clear_caches_every=2,
        max_worker_memory_mb=1000.0,
    )
    checkpoint = _checkpoint_config(legacy_task_config)
    (states_dir.parent / "teacher_checkpoint_config.json").write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )

    payload = {
        "states_dir": str(states_dir.resolve()),
        "task_config": legacy_task_config,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = f"impact_teacher_{hashlib.sha256(encoded).hexdigest()[:24]}"

    resumed = teacher_run_id(
        states_dir,
        _task_config(
            disable_cache=True,
            clear_caches_every=0,
            max_worker_memory_mb=2400.0,
        ),
    )
    assert resumed == expected


def test_existing_split_checkpoint_migrates_cache_toggle_to_runtime_on_compare(
    tmp_path,
):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    old_split = {
        "checkpoint_version": 2,
        "raw_dir": "D:/data/raw",
        "transitions_path": "D:/data/transitions.csv",
        "scenario_ids": [1, 2, 3],
        "checkpoint_config_schema_version": 1,
        "semantic_task_config": {
            **semantic_teacher_task_config(_task_config()),
            "disable_cache": False,
        },
        "runtime_task_config": {
            "max_worker_memory_mb": 1000.0,
        },
    }
    config_path.write_text(json.dumps(old_split), encoding="utf-8")

    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(_task_config(disable_cache=True)),
    )


def test_split_checkpoint_task_config_can_be_loaded_for_diagnostics(tmp_path):
    path = tmp_path / "teacher_checkpoint_config.json"
    payload = checkpoint_config_payload(_checkpoint_config(_task_config()))
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_teacher_task_config(path)
    assert loaded["depth"] == 5
    assert loaded["disable_cache"] is False
    assert loaded["clear_caches_every"] == 2
    assert loaded["max_worker_memory_mb"] == 1000.0
