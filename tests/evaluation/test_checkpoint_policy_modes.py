from __future__ import annotations

from types import SimpleNamespace

import pytest

from grid_topology_ai.evaluation import checkpoint
from grid_topology_ai.evaluation.policy_comparison import PolicyMode
from grid_topology_ai.termination import TerminationReason


def _metrics(*, secure: bool) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 95.0 if secure else 140.0,
        "num_overloaded_branches": 0 if secure else 1,
        "num_hard_overloaded_branches": 0 if secure else 1,
        "total_thermal_overload_mva": 0.0 if secure else 20.0,
        "num_outaged_branches": 1 if secure else 0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }


class _State:
    def __init__(self, *, secure: bool) -> None:
        self.metrics = _metrics(secure=secure)


class _Action:
    def __init__(self, action_id: int, branch_id: int | None) -> None:
        self.action_id = action_id
        self.branch_id = branch_id


class _Env:
    executed_action_ids: list[int] = []

    def __init__(self, **kwargs: object) -> None:
        self.current_state = _State(secure=False)
        self.done = False
        self.solved = False
        self.termination_reason = None

    def reset(self, scenario_id: int):
        return self.current_state

    def step(self, action: _Action):
        self.executed_action_ids.append(action.action_id)
        self.current_state = _State(secure=True)
        self.done = True
        self.solved = True
        self.termination_reason = TerminationReason.SOLVED
        return SimpleNamespace(reward=5.0, done=True, solved=True)


class _Planner:
    def search_from_env(self, env: _Env):
        actions = {
            1: _Action(1, 11),
            2: _Action(2, 22),
        }
        return SimpleNamespace(
            best_action_id=1,
            policy={1: 0.7, 2: 0.3},
            root=SimpleNamespace(actions_by_id=actions),
        )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allowed_action_ids: tuple[int, ...],
) -> None:
    _Env.executed_action_ids = []
    monkeypatch.setattr(checkpoint, "_ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(checkpoint, "TopologySwitchingEnv", _Env)
    monkeypatch.setattr(
        checkpoint,
        "analyze_root_branches",
        lambda **kwargs: SimpleNamespace(
            allowed_action_ids=allowed_action_ids,
        ),
    )
    monkeypatch.setattr(
        checkpoint,
        "make_do_nothing_action",
        lambda: _Action(0, None),
    )


def test_constrained_episode_executes_action_from_constrained_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, allowed_action_ids=(2,))

    row = checkpoint.run_episode(
        scenario_id=1,
        adapter=object(),
        backend=object(),
        action_space=object(),
        reward_fn=object(),
        planner=_Planner(),
        max_steps=2,
        gamma=0.95,
        use_continuation_gate=True,
        min_hard_improvement=0.0,
        min_soft_improvement=0.0,
        min_gate_visits=0,
        min_gate_visit_fraction=0.0,
        policy_mode=PolicyMode.CONSTRAINED,
    )

    assert _Env.executed_action_ids == [2]
    assert row["policy_mode"] == "constrained"
    assert row["actions"] == "[2]"
    assert row["constraint_changed_policy"] is True
    assert row["constraint_exhausted"] is False


def test_empty_constrained_support_terminates_without_action_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, allowed_action_ids=())

    row = checkpoint.run_episode(
        scenario_id=2,
        adapter=object(),
        backend=object(),
        action_space=object(),
        reward_fn=object(),
        planner=_Planner(),
        max_steps=2,
        gamma=0.95,
        use_continuation_gate=True,
        min_hard_improvement=0.0,
        min_soft_improvement=0.0,
        min_gate_visits=0,
        min_gate_visit_fraction=0.0,
        policy_mode=PolicyMode.CONSTRAINED,
    )

    assert _Env.executed_action_ids == []
    assert row["actions"] == "[]"
    assert row["constraint_exhausted"] is True
    assert row["empty_constrained_support_count"] == 1
    assert row["done"] is True
    assert row["solved"] is False
    assert row["termination_reason"] == "constraint_exhausted"
