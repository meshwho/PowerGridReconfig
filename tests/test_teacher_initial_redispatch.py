from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import grid_topology_ai.teacher_runtime as runtime
from grid_topology_ai.physics.redispatch import MinimalRedispatchResult
from grid_topology_ai.termination import TerminationReason


def _metrics(
    *,
    max_loading: float,
    overloaded: int,
    hard_overloaded: int,
) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": float(max_loading),
        "num_overloaded_branches": int(overloaded),
        "num_hard_overloaded_branches": int(hard_overloaded),
        "total_thermal_overload_mva": max(float(max_loading) - 100.0, 0.0),
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


class FakePhysicsConfig:
    def to_dict(self) -> dict[str, object]:
        return {"test": True}


class FakeStateStore:
    def __init__(self) -> None:
        self.metadata: list[dict[str, object]] = []

    def save_state(self, *, state, state_id, action_mask, extra_metadata):
        del state, action_mask
        self.metadata.append(dict(extra_metadata))
        return Path(f"/tmp/{state_id}.npz")


class FakeActionSpace:
    config = SimpleNamespace(to_contract_dict=lambda: {"test": True})

    @staticmethod
    def operational_action_mask(state):
        del state
        return np.array([False, True], dtype=bool)


class FakeEnv:
    initial_state = SimpleNamespace(
        safety=100.0,
        branch_ids=np.array([1], dtype=np.int64),
        metrics=_metrics(
            max_loading=140.0,
            overloaded=3,
            hard_overloaded=2,
        ),
    )
    after_state = SimpleNamespace(
        safety=50.0,
        branch_ids=np.array([1], dtype=np.int64),
        metrics=_metrics(
            max_loading=110.0,
            overloaded=1,
            hard_overloaded=0,
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.current_state = None
        self.done = False
        self.solved = False
        self.termination_reason = None

    def reset(self, scenario_id):
        del scenario_id
        self.current_state = self.initial_state
        self.done = False
        self.solved = False
        self.termination_reason = None
        return self.current_state

    def clone(self):
        cloned = type(self)()
        cloned.current_state = self.current_state
        cloned.done = self.done
        cloned.solved = self.solved
        cloned.termination_reason = self.termination_reason
        return cloned

    def operational_action_mask(self):
        return np.array([False, True], dtype=bool)

    def action_by_id(self, action_id):
        return int(action_id)

    def step(self, action):
        assert int(action) == 1
        self.current_state = self.after_state
        return SimpleNamespace(
            next_state=self.current_state,
            reward=0.0,
            done=False,
            solved=False,
            power_flow_success=True,
            info={"termination_reason": None},
        )


class FakePlanner:
    def __init__(self, config, physics_config=None) -> None:
        del config, physics_config

    def search(self, *, env, scenario_id):
        del env, scenario_id
        best = SimpleNamespace(
            action_ids=[1, 0],
            branch_ids=[1, None],
            safety_score=50.0,
            max_loading_percent=110.0,
            num_hard_overloaded=0,
            num_overloaded=1,
            total_hard_overload=0.0,
            squared_hard_overload=0.0,
            total_overload=10.0,
            short_sequence=lambda: "1 -> 0",
        )
        return SimpleNamespace(
            best_node=best,
            evaluated_actions=7,
        )


class FakeRootPlanner(FakePlanner):
    def search(self, *, env, scenario_id):
        del scenario_id
        state = env.current_state
        metrics = state.metrics
        best = SimpleNamespace(
            action_ids=[],
            branch_ids=[],
            safety_score=float(state.safety),
            max_loading_percent=float(metrics["max_loading_percent"]),
            num_hard_overloaded=int(metrics["num_hard_overloaded_branches"]),
            num_overloaded=int(metrics["num_overloaded_branches"]),
            total_hard_overload=0.0,
            squared_hard_overload=0.0,
            total_overload=float(metrics["total_thermal_overload_mva"]),
            short_sequence=lambda: "",
        )
        return SimpleNamespace(
            best_node=best,
            evaluated_actions=7,
        )


class FakeValidatedRootPlanner(FakeRootPlanner):
    def search(self, *, env, scenario_id):
        result = super().search(env=env, scenario_id=scenario_id)
        result.best_node.action_ids = [0]
        result.best_node.branch_ids = [None]
        return result


class FakeWorseJPlanner(FakePlanner):
    def search(self, *, env, scenario_id):
        result = super().search(env=env, scenario_id=scenario_id)
        result.best_node.safety_score = 120.0
        return result


def _task_config() -> dict[str, object]:
    return {
        "max_steps": 5,
        "depth": 4,
        "beam_width": 10,
        "candidate_pool": 20,
        "top_k": 10,
        "redispatch_candidates_per_switch_count": 5,
        "gamma": 1.0,
        "allow_hard_count_increase": False,
        "use_lodf_screening": False,
        "lodf_screen_top_k": 10,
        "lodf_min_candidate_count": 1,
        "soft_policy_temperature": 0.0,
        "max_teacher_steps": 4,
        "use_soft_root_policy": False,
    }


def _install_runtime(monkeypatch, state_store: FakeStateStore) -> None:
    context = {
        "adapter": object(),
        "backend": object(),
        "action_space": FakeActionSpace(),
        "reward_fn": object(),
        "physics_config": FakePhysicsConfig(),
        "state_store": state_store,
        "task_config": _task_config(),
    }
    monkeypatch.setattr(runtime, "_WORKER_CONTEXT", context)
    monkeypatch.setattr(runtime, "TopologySwitchingEnv", FakeEnv)
    monkeypatch.setattr(runtime, "ImpactBeamSearchPlanner", FakePlanner)
    monkeypatch.setattr(runtime, "_worker_run_id", lambda: "test-run")
    monkeypatch.setattr(
        runtime,
        "safety_score",
        lambda state, physics_config=None: float(state.safety),
    )
    monkeypatch.setattr(
        runtime,
        "state_utility",
        lambda state, physics_config=None: (
            1.0
            if runtime.assess_physical_state(state.metrics).physically_secure
            else 0.0
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_redispatch_aware_selection",
        lambda result, task_config, initial_redispatch_result=None: (result, {}),
    )
    monkeypatch.setattr(runtime, "_selection_provenance", lambda result, diagnostics: {})
    monkeypatch.setattr(
        runtime,
        "make_policy_from_final_beam",
        lambda result, temperature: ({1: 1.0}, {1: 1}),
    )


def _failed_redispatch() -> MinimalRedispatchResult:
    return MinimalRedispatchResult(
        opf_success=False,
        assessment=None,
        message="no feasible initial redispatch",
    )


def _validated_redispatch() -> MinimalRedispatchResult:
    return MinimalRedispatchResult(
        opf_success=True,
        assessment=runtime.assess_physical_state(
            _metrics(
                max_loading=90.0,
                overloaded=0,
                hard_overloaded=0,
            )
        ),
        message="validated",
        redispatch_l1_mw=12.5,
        redispatch_up_mw=6.4,
        redispatch_down_mw=6.1,
        redispatch_max_generator_delta_mw=4.2,
    )


def test_initial_redispatch_failure_stays_missing_in_rows_and_state_metadata(
    monkeypatch,
) -> None:
    state_store = FakeStateStore()
    _install_runtime(monkeypatch, state_store)
    calls: list[object] = []

    def fake_initial_redispatch(backend, state):
        del backend
        calls.append(state)
        return _failed_redispatch()

    monkeypatch.setattr(runtime, "run_minimal_ac_redispatch", fake_initial_redispatch)

    result = runtime._generate_scenario(17)
    runtime._SELECTION_PROVENANCE_BY_SCENARIO.pop(17, None)

    assert result["ok"] is True, result.get("traceback")
    assert calls == [FakeEnv.initial_state, FakeEnv.after_state]
    assert len(result["rows"]) == 2
    assert [row["state_path"] for row in result["rows"]] == [
        str(Path("/tmp/impact_teacher_scenario_000017_step_000.npz")),
        str(Path("/tmp/impact_teacher_scenario_000017_step_001.npz")),
    ]
    assert len(state_store.metadata) == 2

    for artifact in [*result["rows"], *state_store.metadata]:
        assert artifact["initial_redispatch_attempted"] is True
        assert artifact["initial_redispatch_opf_success"] is False
        assert artifact["initial_redispatch_validated"] is False
        assert artifact["initial_redispatch_l1_mw"] is None
        assert artifact["initial_redispatch_up_mw"] is None
        assert artifact["initial_redispatch_down_mw"] is None
        assert artifact["initial_redispatch_max_generator_delta_mw"] is None


def test_zero_switch_failed_redispatch_is_saved_as_max_steps_terminal_target(
    monkeypatch,
) -> None:
    state_store = FakeStateStore()
    _install_runtime(monkeypatch, state_store)
    monkeypatch.setattr(runtime, "ImpactBeamSearchPlanner", FakeRootPlanner)
    monkeypatch.setattr(
        runtime,
        "run_minimal_ac_redispatch",
        lambda backend, state: _failed_redispatch(),
    )

    def fail_if_topology_step_runs(self, action):
        del self, action
        raise AssertionError("0-switch terminal target must not execute an environment action")

    monkeypatch.setattr(FakeEnv, "step", fail_if_topology_step_runs)

    result = runtime._generate_scenario(23)
    runtime._SELECTION_PROVENANCE_BY_SCENARIO.pop(23, None)

    assert result["ok"] is True, result.get("traceback")
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["selected_action_id"] == 0
    assert row["selected_branch_id"] is None
    assert row["termination_reason"] == TerminationReason.MAX_STEPS_REACHED.value
    assert row["step_termination_reason"] == TerminationReason.MAX_STEPS_REACHED.value
    assert state_store.metadata[0]["teacher_decision_reason"] == (
        "terminal_redispatch_unavailable"
    )
    assert result["summary"]["first_action"] == 0
    assert result["summary"]["first_branch"] is None
    assert result["summary"]["handoff_added"] is False
    assert result["summary"]["handoff_reason"] is None


def test_zero_switch_secure_root_is_saved_as_solved_terminal_target(
    monkeypatch,
) -> None:
    state_store = FakeStateStore()
    _install_runtime(monkeypatch, state_store)
    secure_root = SimpleNamespace(
        safety=0.0,
        branch_ids=np.array([1], dtype=np.int64),
        metrics=_metrics(
            max_loading=90.0,
            overloaded=0,
            hard_overloaded=0,
        ),
    )
    monkeypatch.setattr(FakeEnv, "initial_state", secure_root)
    monkeypatch.setattr(runtime, "ImpactBeamSearchPlanner", FakeRootPlanner)
    monkeypatch.setattr(
        runtime,
        "run_minimal_ac_redispatch",
        lambda backend, state: _failed_redispatch(),
    )

    result = runtime._generate_scenario(24)
    runtime._SELECTION_PROVENANCE_BY_SCENARIO.pop(24, None)

    assert result["ok"] is True, result.get("traceback")
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["selected_action_id"] == 0
    assert row["solved"] is True
    assert row["termination_reason"] == TerminationReason.SOLVED.value
    assert row["step_termination_reason"] == TerminationReason.SOLVED.value
    assert state_store.metadata[0]["teacher_decision_reason"] == (
        "initial_topology_solved"
    )
    assert result["summary"]["handoff_added"] is False


def test_zero_switch_validated_redispatch_keeps_terminal_handoff(
    monkeypatch,
) -> None:
    state_store = FakeStateStore()
    _install_runtime(monkeypatch, state_store)
    monkeypatch.setattr(runtime, "ImpactBeamSearchPlanner", FakeValidatedRootPlanner)
    monkeypatch.setattr(
        runtime,
        "run_minimal_ac_redispatch",
        lambda backend, state: _validated_redispatch(),
    )

    def fail_if_topology_step_runs(self, action):
        del self, action
        raise AssertionError("validated root handoff must not execute an environment action")

    monkeypatch.setattr(FakeEnv, "step", fail_if_topology_step_runs)

    result = runtime._generate_scenario(25)
    runtime._SELECTION_PROVENANCE_BY_SCENARIO.pop(25, None)

    assert result["ok"] is True, result.get("traceback")
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["selected_action_id"] == 0
    assert row["selected_branch_id"] is None
    assert row["termination_reason"] == (
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER.value
    )
    assert state_store.metadata[0]["teacher_decision_reason"] == (
        "terminal_redispatch_selected"
    )
    assert result["summary"]["handoff_added"] is True
    assert result["summary"]["handoff_reason"] == "terminal_redispatch_selected"


def test_terminal_redispatch_selection_is_not_rejected_when_selected_j_is_worse(
    monkeypatch,
) -> None:
    state_store = FakeStateStore()
    _install_runtime(monkeypatch, state_store)
    monkeypatch.setattr(runtime, "ImpactBeamSearchPlanner", FakeWorseJPlanner)
    monkeypatch.setattr(
        runtime,
        "run_minimal_ac_redispatch",
        lambda backend, state: _failed_redispatch(),
    )

    result = runtime._generate_scenario(29)
    runtime._SELECTION_PROVENANCE_BY_SCENARIO.pop(29, None)

    assert result["ok"] is True, result.get("traceback")
    assert result["summary"]["teacher_final_safety"] == 120.0
    assert result["summary"]["total_safety_improvement"] == -20.0
    assert [row["selected_action_id"] for row in result["rows"]] == [1, 0]
    assert result["summary"]["handoff_reason"] == "terminal_redispatch_selected"


def test_initial_redispatch_diagnostics_preserve_validated_magnitudes() -> None:
    diagnostics = runtime._initial_redispatch_diagnostics(_validated_redispatch())

    assert diagnostics == {
        "initial_redispatch_attempted": True,
        "initial_redispatch_opf_success": True,
        "initial_redispatch_validated": True,
        "initial_redispatch_l1_mw": 12.5,
        "initial_redispatch_up_mw": 6.4,
        "initial_redispatch_down_mw": 6.1,
        "initial_redispatch_max_generator_delta_mw": 4.2,
    }
