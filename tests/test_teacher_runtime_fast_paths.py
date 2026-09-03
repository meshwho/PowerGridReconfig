from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.actions import GridFMAction, GridFMActionSpace
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.physics import redispatch
from grid_topology_ai.power_flow.backend import (
    GridFMPowerFlowBackend,
    _GeneratorOperatingPointState,
)
from grid_topology_ai.power_flow.problem import (
    GeneratorOperatingPoint,
    build_power_flow_problem_from_state,
)
from grid_topology_ai.search.teacher import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchNode,
    ImpactBeamSearchPlanner,
    safety_score,
)
from grid_topology_ai.environment import TopologyStepResult
from grid_topology_ai.state import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)


def _metrics(*, loading: float) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": float(loading),
        "num_overloaded_branches": int(loading > 100.000001),
        "num_hard_overloaded_branches": int(loading > 120.000001),
        "total_thermal_overload_mva": max(float(loading) - 100.0, 0.0),
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


def _beam_state(*, loading: float) -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("br_status")] = 1.0
    branch_features[0, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = loading
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros(
            (2, len(BUS_FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        branch_features=branch_features,
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        branch_ids=np.asarray([10], dtype=np.int64),
        branch_status=np.asarray([1.0], dtype=np.float32),
        metrics=_metrics(loading=loading),
        outaged_branch_ids=[],
        bus_ids=np.asarray([0, 1], dtype=np.int64),
    )


class _BeamEnv:
    def __init__(self, before: GridFMState, after: GridFMState, calls: list[bool]):
        self.current_state = before
        self._after = after
        self._calls = calls

    def clone(self):
        return _BeamEnv(self.current_state, self._after, self._calls)

    def step(self, action, *, compute_reward: bool = True):
        self._calls.append(bool(compute_reward))
        self.current_state = self._after
        return TopologyStepResult(
            next_state=self._after,
            reward=0.0,
            done=False,
            solved=False,
            power_flow_success=True,
            action=action,
            reward_breakdown=None,
            power_flow_result=None,
            terminal_outcome_evidence=None,
            info={"termination_reason": None},
        )


def test_beam_expansion_skips_diagnostic_reward_without_changing_impact_score():
    before = _beam_state(loading=130.0)
    after = _beam_state(loading=110.0)
    calls: list[bool] = []
    env = _BeamEnv(before, after, calls)
    action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )
    planner = ImpactBeamSearchPlanner(
        ImpactBeamSearchConfig(gamma=1.0, switch_penalty=5.0)
    )
    before_safety = safety_score(
        before,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    after_safety = safety_score(
        after,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    node = ImpactBeamSearchNode(
        env=env,
        safety_score=before_safety,
        num_hard_overloaded=1,
        depth=0,
    )

    child = planner._expand_node(node, action)

    assert child is not None
    assert calls == [False]
    assert child.safety_score == pytest.approx(after_safety)
    expected_impact = before_safety - after_safety - 5.0 + 50.0
    assert child.impact_scores[-1] == pytest.approx(expected_impact)


def test_action_space_fast_mask_preserves_open_and_close_semantics():
    branch_features = np.zeros(
        (3, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = [1.0, 0.0, 1.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("loading_percent")] = [
        90.0,
        0.0,
        130.0,
    ]
    state = GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros(
            (2, len(BUS_FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        branch_features=branch_features,
        edge_index=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        branch_ids=np.asarray([10, 20, 30], dtype=np.int64),
        branch_status=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        metrics=_metrics(loading=130.0),
        outaged_branch_ids=[20],
        bus_ids=np.asarray([0, 1], dtype=np.int64),
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=100.0,
        closeable_branch_ids=(20,),
        enable_cache=False,
    )

    structural = action_space.structural_action_mask(state)
    operational = action_space.operational_action_mask(state)
    actions = action_space.valid_actions(state)

    np.testing.assert_array_equal(structural, [True, True, True, True])
    np.testing.assert_array_equal(operational, [True, False, True, True])
    assert [action.action_id for action in actions] == [0, 2, 3]
    assert actions[1].action_type == "switch_on_branch"
    assert actions[1].branch_id == 20
    assert actions[1].target_status == 1
    assert actions[2].action_type == "switch_off_branch"
    assert actions[2].branch_id == 30
    assert actions[2].target_status == 0


def _adapter_and_trusted_state():
    buses = []
    for bus_id, bus_type in ((10, "REF"), (20, "PQ")):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "load_scenario_idx": 0.0,
                "bus": bus_id,
                "Pd": 35.0 if bus_id == 20 else 0.0,
                "Qd": 8.0 if bus_id == 20 else 0.0,
                "Vm": 1.0,
                "Va": 0.0,
                "vn_kv": 110.0,
                "GS": 0.0,
                "BS": 0.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                bus_type: 1.0,
            }
        )
        buses.append(row)

    branch = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
    branch.update(
        {
            "scenario": 1,
            "load_scenario_idx": 0.0,
            "idx": 7,
            "from_bus": 10,
            "to_bus": 20,
            "r": 0.01,
            "x": 0.1,
            "b": 0.01,
            "tap": 0.0,
            "shift": 0.0,
            "rate_a": 100.0,
            "br_status": 1.0,
            "ang_min": -30.0,
            "ang_max": 30.0,
        }
    )
    generator = {
        "scenario": 1,
        "idx": 0,
        "bus": 10,
        "p_mw": 40.0,
        "q_mvar": 5.0,
        "min_p_mw": 0.0,
        "max_p_mw": 100.0,
        "min_q_mvar": -50.0,
        "max_q_mvar": 50.0,
        "in_service": 1.0,
    }
    adapter = SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame([branch]),
        gen_df=pd.DataFrame([generator]),
    )
    bus_df = adapter.bus_df.sort_values("bus").reset_index(drop=True)
    branch_df = adapter.branch_df.sort_values("idx").reset_index(drop=True)
    state = _GeneratorOperatingPointState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=bus_df[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        branch_features=branch_df[BRANCH_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        branch_ids=np.asarray([7], dtype=np.int64),
        branch_status=np.asarray([1.0], dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=np.asarray([10, 20], dtype=np.int64),
        generator_ids=np.asarray([0], dtype=np.int64),
        generator_p_mw=np.asarray([40.0], dtype=np.float64),
        generator_q_mvar=np.asarray([5.0], dtype=np.float64),
        generator_status=np.asarray([1.0], dtype=np.float64),
    )
    return adapter, state


def test_trusted_problem_builder_matches_canonical_problem_exactly():
    adapter, state = _adapter_and_trusted_state()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    template, _ = backend._scenario_problem_resources(1)
    action = GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=7,
        branch_pos=0,
    )

    trusted = backend._build_trusted_power_flow_problem(
        template=template,
        state=state,
        action=action,
        switched_off_branch_id=None,
    )
    canonical = build_power_flow_problem_from_state(
        template=template,
        state=state,
        branch_id=7,
        target_status=0,
        generator_operating_point=GeneratorOperatingPoint.from_state(state),
    )

    assert trusted.base_mva == canonical.base_mva
    np.testing.assert_array_equal(trusted.bus, canonical.bus)
    np.testing.assert_array_equal(trusted.branch, canonical.branch)
    np.testing.assert_array_equal(trusted.gen, canonical.gen)


def _redispatch_case() -> dict[str, object]:
    bus = np.zeros((1, 13), dtype=np.float64)
    branch = np.zeros((0, 13), dtype=np.float64)
    gen = np.zeros((1, 21), dtype=np.float64)
    gen[:, GEN_STATUS] = 1.0
    gen[:, PG] = 50.0
    gen[:, QG] = 5.0
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus,
        "branch": branch,
        "gen": gen,
    }


class _RedispatchBackend:
    enable_cache = False
    physics_config = object()

    def __init__(self, *, trusted: bool):
        self.trusted = bool(trusted)
        self.ppc = _redispatch_case()

    def _is_trusted_repeated_state(self, state) -> bool:
        del state
        return self.trusted

    def _build_ppc_from_state(self, state):
        del state
        return self.ppc, {}

    def _build_pp_options(self):
        return {}


def _install_redispatch_success(monkeypatch, *, input_calls: list[int]):
    def validate_input(*args, **kwargs):
        del args, kwargs
        input_calls.append(1)

    def runopf(case, options):
        del options
        return {
            "version": "2",
            "baseMVA": case["baseMVA"],
            "bus": np.array(case["bus"], copy=True),
            "branch": np.array(case["branch"], copy=True),
            "gen": np.array(case["gen"], copy=True),
            "success": True,
        }

    monkeypatch.setattr(redispatch, "validate_ppc_input", validate_input)
    monkeypatch.setattr(redispatch, "runopf", runopf)
    monkeypatch.setattr(
        redispatch,
        "calculate_physical_metrics_from_result",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        redispatch,
        "assess_physical_state",
        lambda metrics: SimpleNamespace(physically_secure=True),
    )


def test_trusted_redispatch_skips_repeated_input_validation(monkeypatch):
    input_calls: list[int] = []
    _install_redispatch_success(
        monkeypatch,
        input_calls=input_calls,
    )
    state = SimpleNamespace(
        scenario_id=5,
        generator_ids=np.asarray([0], dtype=np.int64),
    )

    result = redispatch.run_minimal_ac_redispatch(
        _RedispatchBackend(trusted=True),
        state,
    )

    assert result.validated is True
    assert input_calls == []


def test_untrusted_redispatch_keeps_input_validation(monkeypatch):
    input_calls: list[int] = []
    _install_redispatch_success(
        monkeypatch,
        input_calls=input_calls,
    )
    state = SimpleNamespace(
        scenario_id=5,
        generator_ids=np.asarray([0], dtype=np.int64),
    )

    result = redispatch.run_minimal_ac_redispatch(
        _RedispatchBackend(trusted=False),
        state,
    )

    assert result.validated is True
    assert input_calls == [1]
