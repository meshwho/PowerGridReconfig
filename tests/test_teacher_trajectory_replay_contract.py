from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import grid_topology_ai.teacher_runtime as teacher


def _teacher_args(**overrides):
    values = {
        "depth": 4,
        "beam_width": 10,
        "candidate_pool": 80,
        "top_k": 30,
        "redispatch_candidates_per_switch_count": 5,
        "gamma": 1.0,
        "pf_alg": 3,
        "pf_max_iter": 30,
        "max_steps": 5,
        "max_teacher_steps": 4,
        "soft_policy_temperature": 0.0,
        "use_soft_root_policy": False,
        "allow_hard_count_increase": False,
        "disable_cache": False,
        "use_lodf_screening": False,
        "lodf_screen_top_k": 0,
        "lodf_min_candidate_count": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_teacher_replay_keeps_beam_selected_steps_without_legacy_gates() -> None:
    source = inspect.getsource(teacher._generate_scenario)

    assert 'continue_reason = "selected_by_beam_search"' in source
    assert "step_improvement = float(safety_before - safety_after)" in source
    assert "_selected_teacher_replay_decision" not in source
    assert "min_continue_improvement" not in source
    assert "max_loading_increase_limit" not in source


def test_teacher_replay_rejects_power_flow_divergence() -> None:
    source = inspect.getsource(teacher._generate_scenario)

    assert "if next_state is None:" in source
    assert "power-flow failure during replay" in source


def test_teacher_replay_rejects_invalid_selected_action() -> None:
    mask = np.array([True, False, True], dtype=bool)

    with pytest.raises(RuntimeError, match="became invalid during replay"):
        teacher._selected_teacher_action_is_valid(mask, 1)

    assert teacher._selected_teacher_action_is_valid(mask, 2) is True


def test_teacher_search_depth_cannot_exceed_replay_horizon() -> None:
    config = teacher.make_task_config(
        _teacher_args(depth=6, max_teacher_steps=4)
    )

    assert config["depth"] == 4
    assert config["max_teacher_steps"] == 4
    assert config["redispatch_candidates_per_switch_count"] == 5


def test_teacher_search_depth_validation_rejects_empty_replay_horizon() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        teacher.make_task_config(_teacher_args(max_teacher_steps=0))


def test_teacher_redispatch_archive_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="redispatch_candidates_per_switch_count"):
        teacher.make_task_config(
            _teacher_args(redispatch_candidates_per_switch_count=0)
        )


def test_worker_batch_calls_canonical_scenario_generator_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_process(scenario_id: int):
        calls.append(int(scenario_id))
        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "reason": "test_failure",
        }

    monkeypatch.setattr(teacher, "process_one_scenario_fast", fake_process)

    results = teacher.process_scenario_batch([1, 2])

    assert calls == [1, 2]
    assert [result["scenario_id"] for result in results] == [1, 2]
