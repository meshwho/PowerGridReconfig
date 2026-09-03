from __future__ import annotations

import json

import pytest

from grid_topology_ai.search.teacher import (
    ensure_teacher_checkpoint_config,
    load_teacher_task_config,
    semantic_teacher_task_config,
    teacher_run_id,
    teacher_source_identity,
)


def _task_config(**overrides):
    config = {
        "depth": 5,
        "beam_width": 20,
        "candidate_pool": 160,
        "top_k": 70,
        "disable_cache": False,
    }
    config.update(overrides)
    return config


def _checkpoint_config(task_config, *, source_identity=None):
    return {
        "scenario_ids": [1, 2, 3],
        "task_config": task_config,
        "source_identity": source_identity or {"dataset": "test"},
    }


def _source_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "bus_data.parquet").write_bytes(b"bus-v1")
    (raw_dir / "branch_data.parquet").write_bytes(b"branch-v1")
    (raw_dir / "gen_data.parquet").write_bytes(b"gen-v1")
    transitions = tmp_path / "transitions.csv"
    transitions.write_bytes(b"scenario\n1\n2\n3\n")
    return raw_dir, transitions


def test_cache_toggle_is_not_part_of_semantic_task_config():
    semantic = semantic_teacher_task_config(_task_config())

    assert semantic["depth"] == 5
    assert semantic["beam_width"] == 20
    assert "disable_cache" not in semantic


def test_checkpoint_persists_current_light_identity(tmp_path):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    config = _checkpoint_config(_task_config())

    ensure_teacher_checkpoint_config(config_path, config)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload == {
        "scenario_ids": [1, 2, 3],
        "source_identity": {"dataset": "test"},
        "task_config": semantic_teacher_task_config(_task_config()),
    }


def test_checkpoint_allows_runtime_only_cache_change(tmp_path):
    config_path = tmp_path / "teacher_checkpoint_config.json"
    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(_task_config(disable_cache=False)),
    )

    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(_task_config(disable_cache=True)),
    )


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


def test_teacher_source_identity_tracks_current_input_files(tmp_path):
    raw_dir, transitions = _source_files(tmp_path)
    original = teacher_source_identity(raw_dir, transitions)

    (raw_dir / "branch_data.parquet").write_bytes(b"branch-v2-longer")
    changed_raw = teacher_source_identity(raw_dir, transitions)
    assert changed_raw != original

    transitions.write_bytes(b"scenario\n1\n2\n3\n4\n")
    changed_transitions = teacher_source_identity(raw_dir, transitions)
    assert changed_transitions != changed_raw


def test_teacher_source_identity_is_location_independent(tmp_path):
    first_raw, first_transitions = _source_files(tmp_path / "first")
    second_root = tmp_path / "second"
    second_raw = second_root / "raw"
    second_raw.mkdir(parents=True)
    for name in ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet"):
        (second_raw / name).write_bytes((first_raw / name).read_bytes())
    second_transitions = second_root / "renamed.csv"
    second_transitions.write_bytes(first_transitions.read_bytes())

    assert teacher_source_identity(first_raw, first_transitions) == teacher_source_identity(
        second_raw, second_transitions
    )


def test_checkpoint_rejects_changed_source_identity(tmp_path):
    raw_dir, transitions = _source_files(tmp_path)
    config_path = tmp_path / "teacher_checkpoint_config.json"

    ensure_teacher_checkpoint_config(
        config_path,
        _checkpoint_config(
            _task_config(),
            source_identity=teacher_source_identity(raw_dir, transitions),
        ),
    )

    (raw_dir / "bus_data.parquet").write_bytes(b"bus-v2-longer")

    with pytest.raises(RuntimeError, match="does not match"):
        ensure_teacher_checkpoint_config(
            config_path,
            _checkpoint_config(
                _task_config(),
                source_identity=teacher_source_identity(raw_dir, transitions),
            ),
        )


def test_run_id_depends_only_on_semantic_teacher_settings(tmp_path):
    states_dir = tmp_path / "run" / "states"
    states_dir.mkdir(parents=True)
    ensure_teacher_checkpoint_config(
        states_dir.parent / "teacher_checkpoint_config.json",
        _checkpoint_config(_task_config()),
    )

    first = teacher_run_id(states_dir, _task_config(disable_cache=False))
    second = teacher_run_id(states_dir, _task_config(disable_cache=True))

    assert first == second
    assert first != teacher_run_id(states_dir, _task_config(depth=6))


def test_run_id_is_location_independent_for_same_checkpoint_identity(tmp_path):
    config = _checkpoint_config(_task_config(), source_identity={"sha256": "same"})
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output_dir in (first, second):
        (output_dir / "states").mkdir(parents=True)
        ensure_teacher_checkpoint_config(
            output_dir / "teacher_checkpoint_config.json", config
        )

    assert teacher_run_id(first / "states", _task_config()) == teacher_run_id(
        second / "states", _task_config()
    )


def test_current_checkpoint_task_config_can_be_loaded(tmp_path):
    path = tmp_path / "teacher_checkpoint_config.json"
    ensure_teacher_checkpoint_config(path, _checkpoint_config(_task_config()))

    loaded = load_teacher_task_config(path)

    assert loaded == semantic_teacher_task_config(_task_config())
