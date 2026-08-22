from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.power_flow.backend import GridFMPowerFlowResult
from grid_topology_ai.search.continuation_gate import make_do_nothing_action
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


def _metrics(max_loading: float = 110.0) -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": float(max_loading),
        "num_overloaded_branches": 1,
        "num_hard_overloaded_branches": 0,
        "total_thermal_overload_mva": 4.0,
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


def _state(*, branch_active: bool = True) -> GridFMState:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("br_status"),
    ] = float(branch_active)
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("loading_percent"),
    ] = 110.0 if branch_active else 105.0

    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([int(branch_active)], dtype=np.int64),
        metrics=_metrics(110.0 if branch_active else 105.0),
        outaged_branch_ids=[] if branch_active else [10],
        bus_ids=np.array([100, 200], dtype=np.int64),
    )


def _switch_action() -> GridFMAction:
    return GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
    )


class _ActionSpace:
    def __init__(self) -> None:
        self.stop = make_do_nothing_action()
        self.switch = _switch_action()

    def build_all_actions(self, state: GridFMState) -> list[GridFMAction]:
        del state
        return [self.stop, self.switch]

    def valid_actions(self, state: GridFMState) -> list[GridFMAction]:
        del state
        return [self.stop, self.switch]

    def operational_action_mask(self, state: GridFMState) -> np.ndarray:
        del state
        return np.array([True, True], dtype=bool)


class _Reward:
    physics_config = DEFAULT_PHYSICS_CONFIG

    def compute(self, **kwargs) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(reward=0.0)


class _CachedBackend:
    def __init__(self) -> None:
        self.physics_config = DEFAULT_PHYSICS_CONFIG
        self._initial = _state()
        self._cache: dict[tuple[int, int, tuple[int, ...]], GridFMState] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.exact_cache_hits = 0

    def run_power_flow(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None = None,
    ) -> GridFMPowerFlowResult:
        return GridFMPowerFlowResult(
            success=True,
            scenario_id=int(scenario_id),
            switched_off_branch_id=switched_off_branch_id,
            next_state=self._initial,
            raw_result=None,
            message="initial",
        )

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> GridFMPowerFlowResult:
        del switched_off_branch_id
        assert action is not None
        key = (
            int(state.scenario_id),
            int(action.action_id),
            tuple(int(value) for value in state.branch_status),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.exact_cache_hits += 1
            next_state = cached
            message = "cache hit"
        else:
            self.cache_misses += 1
            next_state = _state(branch_active=False)
            self._cache[key] = next_state
            message = "solved"

        return GridFMPowerFlowResult(
            success=True,
            scenario_id=int(state.scenario_id),
            switched_off_branch_id=10,
            next_state=next_state,
            raw_result=None,
            message=message,
            switched_branch_id=10,
            target_status=0,
        )


def test_mcts_and_executed_self_play_step_share_power_flow_cache() -> None:
    backend = _CachedBackend()
    action_space = _ActionSpace()
    env = TopologySwitchingEnv(
        adapter=object(),
        backend=backend,
        action_space=action_space,
        reward_fn=_Reward(),
        max_steps=1,
    )
    env.reset(1)

    planner = MCTSPlanner(
        config=MCTSConfig(
            num_simulations=1,
            max_depth=1,
            top_k_actions=1,
            exploration_quota=0,
            stop_policy="never",
            use_root_dirichlet_noise=False,
            random_seed=7,
        ),
        evaluator=None,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    result = planner.search_from_env(env)

    assert result.best_action_id == 1
    assert backend.cache_misses == 1
    assert backend.cache_hits == 0
    assert result.root.env.backend is backend
    assert result.root.children[1].env.backend is backend

    step_result = env.step(result.root.actions_by_id[1])

    assert step_result.power_flow_success is True
    assert backend.cache_misses == 1
    assert backend.cache_hits == 1
    assert backend.exact_cache_hits == 1
