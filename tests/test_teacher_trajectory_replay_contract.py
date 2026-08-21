from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher


def _teacher_args(**overrides):
    values = {
        "depth": 4,
        "beam_width": 10,
        "candidate_pool": 80,
        "top_k": 30,
        "gamma": 1.0,
        "pf_alg": 3,
        "pf_max_iter": 30,
        "max_steps": 5,
        "max_teacher_steps": 4,
        "soft_policy_temperature": 0.0,
        "use_soft_root_policy": False,
        "allow_hard_count_increase": False,
        "disable_cache": False,
        "power_flow_failure_penalty": 1_000_000.0,
        "min_continue_improvement_with_hard": 100.0,
        "min_continue_improvement_without_hard": 150.0,
        "max_loading_increase_limit": 5.0,
        "add_handoff_example": False,
        "use_lodf_screening": False,
        "lodf_screen_top_k": 0,
        "lodf_min_candidate_count": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_teacher_replay_keeps_temporarily_worse_selected_step() -> None:
    state_before = SimpleNamespace(
        metrics={
            "num_hard_overloaded_branches": 0,
            "max_loading_percent": 105.0,
        }
    )
    state_after = SimpleNamespace(
        metrics={
            "num_hard_overloaded_branches": 1,
            "max_loading_percent": 125.0,
        }
    )
    task = {
        "allow_hard_count_increase": False,
        "min_continue_improvement_with_hard": 100.0,
        "min_continue_improvement_without_hard": 150.0,
        "max_loading_increase_limit": 5.0,
    }

    continue_action, reason, improvement = teacher._selected_teacher_replay_decision(
        safety_before=100.0,
        safety_after=140.0,
        state_before=state_before,
        state_after=state_after,
        task=task,
    )

    assert continue_action is True
    assert reason == "selected_by_beam_search"
    assert improvement == pytest.approx(-40.0)


def test_teacher_replay_rejects_power_flow_divergence() -> None:
    with pytest.raises(RuntimeError, match="power-flow failure during replay"):
        teacher._selected_teacher_replay_decision(
            safety_before=100.0,
            safety_after=1_000_100.0,
            state_before=object(),
            state_after=None,
            task={},
        )


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


def test_teacher_search_depth_validation_rejects_empty_replay_horizon() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        teacher.make_task_config(_teacher_args(max_teacher_steps=0))


def test_worker_batch_calls_canonical_scenario_generator_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_process(scenario_id: int):
        calls.append(int(scenario_id))
        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "reason": "no_teacher_action_found",
        }

    monkeypatch.setattr(teacher, "process_one_scenario_fast", fake_process)

    results = teacher.process_scenario_batch([1, 2])

    assert calls == [1, 2]
    assert [result["scenario_id"] for result in results] == [1, 2]
