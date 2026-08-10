from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from scripts.self_play import generate_impact_teacher_provenance as provenance


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

    continue_action, reason, improvement = (
        provenance._selected_teacher_replay_decision(
            safety_before=100.0,
            safety_after=140.0,
            state_before=state_before,
            state_after=state_after,
            task=task,
        )
    )

    assert continue_action is True
    assert reason == "selected_by_beam_search"
    assert improvement == pytest.approx(-40.0)


def test_teacher_replay_rejects_power_flow_divergence() -> None:
    with pytest.raises(RuntimeError, match="power-flow failure during replay"):
        provenance._selected_teacher_replay_decision(
            safety_before=100.0,
            safety_after=1_000_100.0,
            state_before=object(),
            state_after=None,
            task={},
        )


def test_teacher_replay_rejects_invalid_selected_action() -> None:
    mask = np.array([True, False, True], dtype=bool)

    with pytest.raises(RuntimeError, match="became invalid during replay"):
        provenance._selected_teacher_action_is_valid(mask, 1)

    assert provenance._selected_teacher_action_is_valid(mask, 2) is True


def test_teacher_search_depth_cannot_exceed_replay_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = {
        "depth": 6,
        "max_teacher_steps": 4,
        "physics_config": DEFAULT_PHYSICS_CONFIG.to_dict(),
    }
    monkeypatch.setattr(
        provenance,
        "_original_make_task_config",
        lambda args: dict(base_config),
    )

    config = provenance.make_task_config(SimpleNamespace())

    assert config["depth"] == 4
    assert config["max_teacher_steps"] == 4


def test_teacher_search_depth_validation_rejects_empty_replay_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = {
        "depth": 4,
        "max_teacher_steps": 0,
        "physics_config": DEFAULT_PHYSICS_CONFIG.to_dict(),
    }
    monkeypatch.setattr(
        provenance,
        "_original_make_task_config",
        lambda args: dict(base_config),
    )

    with pytest.raises(ValueError, match="must be positive"):
        provenance.make_task_config(SimpleNamespace())


def test_worker_installs_replay_contract_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        provenance,
        "_install_worker_instrumentation",
        lambda: events.append("instrumentation"),
    )
    monkeypatch.setattr(
        provenance,
        "_install_worker_replay_contract",
        lambda: events.append("replay_contract"),
    )
    monkeypatch.setattr(
        provenance,
        "_original_process_scenario_batch",
        lambda scenario_ids: [],
    )

    assert provenance.process_scenario_batch([1, 2]) == []
    assert events == ["instrumentation", "replay_contract"]
