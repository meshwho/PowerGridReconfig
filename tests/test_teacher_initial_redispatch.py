from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import grid_topology_ai.teacher_runtime as runtime
from grid_topology_ai.physics.redispatch import MinimalRedispatchResult


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
    @staticmethod
    def operational_action_mask(state):
        del state
        return np.array([False, True], dtype=bool)


class FakeEnv:
    initial_state = SimpleNamespace(
        safety=100.0,
        metrics={
            "max_loading_percent": 140.0,
            "num_hard_overloaded_branches": 2,
            "num_overloaded_branches": 3,
        },
    )
    after_state = SimpleNamespace(
        safety=50.0,
        metrics={
            "max_loading_percent": 110.0,
            "num_hard_overloaded_branches": 1,
            "num_overloaded_branches": 1,
        },
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
            num_hard_overloaded=1,
            num_overloaded=1,
            total_hard_overload=0.0,
            squared_hard_overload=0.0,
            total_overload=10.0,
        )
        return SimpleNamespace(
            best_node=best,
            evaluated_actions=7,
        )


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
        "min_safety_improvement": 0.0,
        "soft_policy_temperature": 0.0,
        "max_teacher_steps": 4,
        "use_soft_root_policy": False,
        "add_handoff_example": False,
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
    monkeypatch.setattr(
        runtime,
        "safety_score",
        lambda state, physics_config=None: float(state.safety),
    )
    monkeypatch.setattr(
        runtime,
        "_redispatch_aware_selection",
        lambda result, task_config: (result, {}),
    )
    monkeypatch.setattr(runtime, "_selection_provenance", lambda result, diagnostics: {})
    monkeypatch.setattr(
        runtime,
        "make_policy_from_final_beam",
        lambda result, temperature: ({1: 1.0}, {1: 1}),
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
        return MinimalRedispatchResult(
            opf_success=False,
            assessment=None,
            message="no feasible initial redispatch",
        )

    monkeypatch.setattr(runtime, "run_minimal_ac_redispatch", fake_initial_redispatch)

    result = runtime._generate_scenario(17)

    assert result["ok"] is True
    assert calls == [FakeEnv.initial_state]
    assert len(result["rows"]) == 2
    assert len(state_store.metadata) == 2

    for artifact in [*result["rows"], *state_store.metadata]:
        assert artifact["initial_redispatch_attempted"] is True
        assert artifact["initial_redispatch_opf_success"] is False
        assert artifact["initial_redispatch_validated"] is False
        assert artifact["initial_redispatch_l1_mw"] is None
        assert artifact["initial_redispatch_up_mw"] is None
        assert artifact["initial_redispatch_down_mw"] is None
        assert artifact["initial_redispatch_max_generator_delta_mw"] is None


def test_initial_redispatch_diagnostics_preserve_validated_magnitudes() -> None:
    redispatch = MinimalRedispatchResult(
        opf_success=True,
        assessment=SimpleNamespace(physically_secure=True),
        message="validated",
        redispatch_l1_mw=12.5,
        redispatch_up_mw=6.4,
        redispatch_down_mw=6.1,
        redispatch_max_generator_delta_mw=4.2,
    )

    diagnostics = runtime._initial_redispatch_diagnostics(redispatch)

    assert diagnostics == {
        "initial_redispatch_attempted": True,
        "initial_redispatch_opf_success": True,
        "initial_redispatch_validated": True,
        "initial_redispatch_l1_mw": 12.5,
        "initial_redispatch_up_mw": 6.4,
        "initial_redispatch_down_mw": 6.1,
        "initial_redispatch_max_generator_delta_mw": 4.2,
    }
