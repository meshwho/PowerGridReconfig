from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from pypower.api import ppoption
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_bus import PQ as BUS_TYPE_PQ
from pypower.idx_bus import PV as BUS_TYPE_PV
from pypower.idx_bus import REF as BUS_TYPE_REF
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)

from grid_topology_ai.cache import (
    DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES,
    ByteLRUCache,
    CachedPowerFlowFailure,
    CachedPowerFlowSuccess,
    ExactPowerFlowCache,
    PowerFlowCacheKey,
)
from grid_topology_ai.config import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    QLimitPolicy,
)
from grid_topology_ai.data import GridFMAdapter
from grid_topology_ai.state import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.physics.constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
)
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.power_flow import (
    InvalidPhysicalState,
    PowerFlowFailureKind,
    PowerFlowNotConverged,
)
from grid_topology_ai.power_flow.problem import (
    CanonicalPowerFlowProblem,
    GeneratorOperatingPoint,
    ScenarioPowerFlowTemplate,
    build_power_flow_problem_from_state,
    build_scenario_power_flow_template,
)
from grid_topology_ai.power_flow.solver import (
    get_power_flow_workload_counters,
    runpf,
)
from grid_topology_ai.state import GridFMStateBuilder
from grid_topology_ai.actions import GridFMAction


def pf_algorithm_name(pf_alg: int) -> str:
    names = {
        1: "newton_raphson",
        2: "fast_decoupled_xb",
        3: "fast_decoupled_bx",
        4: "gauss_seidel",
    }
    return names.get(int(pf_alg), f"unknown_{pf_alg}")


@dataclass(frozen=True)
class GridFMPowerFlowResult:
    """Result of applying one topology action and running AC power flow."""

    success: bool
    scenario_id: int
    switched_off_branch_id: int | None
    next_state: GridFMState | None
    raw_result: dict[str, Any] | None
    message: str
    failure_kind: PowerFlowFailureKind | None = None
    switched_branch_id: int | None = None
    target_status: int | None = None


_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_COL = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}
_GENERATOR_COLUMNS = (
    "bus",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
)
_MAX_PHYSICAL_METRICS_CACHE_BYTES = 8 * 1024 * 1024
_PHYSICAL_METRICS_ENTRY_BYTES = 4096


@dataclass(frozen=True)
class _GeneratorOperatingPointState(GridFMState):
    """Solved state carrying the exact per-generator operating point."""

    generator_ids: np.ndarray | None = None
    generator_p_mw: np.ndarray | None = None
    generator_q_mvar: np.ndarray | None = None
    generator_status: np.ndarray | None = None


# Preserve the public module path used by pickled results and type displays.
GridFMPowerFlowResult.__module__ = __name__


class GridFMPowerFlowBackend:
    """PYPOWER backend with a physically transparent exact-result cache."""

    def __init__(
        self,
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        enable_cache: bool = True,
        store_raw_result: bool = False,
        exact_cache_max_bytes: int = DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES,
    ) -> None:
        self._require_matching_physics_contract(adapter, physics_config)
        self.adapter = adapter
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")
        self.physics_config = physics_config
        self.enable_cache = bool(enable_cache)
        self.store_raw_result = bool(store_raw_result)
        self._state_builder = GridFMStateBuilder(
            physics_config=self.physics_config,
            result_metrics_calculator=calculate_physical_metrics_from_result,
        )
        exact_cache_max_bytes = int(exact_cache_max_bytes)
        self._exact_power_flow_cache = ExactPowerFlowCache(
            max_bytes=exact_cache_max_bytes
        )
        metrics_cache_max_bytes = min(
            max(exact_cache_max_bytes // 8, 0),
            _MAX_PHYSICAL_METRICS_CACHE_BYTES,
        )
        self._physical_metrics_cache: ByteLRUCache[
            bytes, dict[str, object]
        ] = ByteLRUCache(max_bytes=metrics_cache_max_bytes)
        self._active_problem_template: ScenarioPowerFlowTemplate | None = None
        self._active_problem_frames: dict[str, pd.DataFrame] | None = None
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0

    @property
    def base_mva(self) -> float:
        return self.physics_config.base_mva

    @property
    def max_iter(self) -> int:
        return self.physics_config.max_iterations

    @property
    def pf_alg(self) -> int:
        return self.physics_config.pf_alg

    def cache_info(self) -> dict[str, object]:
        info: dict[str, object] = dict(self._exact_power_flow_cache.info())
        info["enabled"] = bool(self.enable_cache)
        return info

    def performance_info(self) -> dict[str, object]:
        """Return exact-cache and PYPOWER workload counters."""

        info = dict(self.cache_info())
        misses = int(info["misses"])
        stock_calls = int(self.stock_runpf_calls)
        info.update(
            {
                "stock_runpf_calls": stock_calls,
                "q_limit_resolves": int(self.q_limit_resolves),
                "solves_per_cache_miss": (
                    float(stock_calls) / float(misses)
                    if misses > 0
                    else 0.0
                ),
            }
        )
        return info

    def reset_performance_counters(self) -> None:
        """Reset counters without discarding exact cached solutions."""

        self._exact_power_flow_cache.reset_counters()
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0

    def clear_cache(self) -> None:
        """Discard exact cached solutions while keeping scenario templates."""

        self._exact_power_flow_cache.clear(reset_counters=True)
        self._physical_metrics_cache.clear(reset_evictions=True)

    @staticmethod
    def _positive_cache_key(key: PowerFlowCacheKey | bytes) -> bytes:
        return key.positive if isinstance(key, PowerFlowCacheKey) else key

    def _cached_physical_metrics(
        self,
        key: PowerFlowCacheKey | bytes,
    ) -> dict[str, object] | None:
        metrics = self._physical_metrics_cache.get(
            self._positive_cache_key(key)
        )
        return None if metrics is None else dict(metrics)

    def _store_physical_metrics(
        self,
        key: PowerFlowCacheKey | bytes,
        metrics: dict[str, object],
    ) -> None:
        positive_key = self._positive_cache_key(key)
        self._physical_metrics_cache.put(
            positive_key,
            dict(metrics),
            size_bytes=_PHYSICAL_METRICS_ENTRY_BYTES + len(positive_key),
        )

    def _discard_physical_metrics(
        self,
        key: PowerFlowCacheKey | bytes,
    ) -> None:
        self._physical_metrics_cache.discard(self._positive_cache_key(key))

    @staticmethod
    def _require_matching_physics_contract(
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig,
    ) -> None:
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")

        adapter_config = getattr(adapter, "physics_config", None)
        if adapter_config is None:
            return
        if not isinstance(adapter_config, PhysicsConfig):
            raise TypeError("adapter.physics_config must be a PhysicsConfig.")
        if adapter_config.fingerprint() != physics_config.fingerprint():
            raise ValueError(
                "Adapter and power-flow backend must use the same "
                "PhysicsConfig fingerprint."
            )

    def _build_pp_options(self) -> dict[str, object]:
        config = self.physics_config
        options = ppoption(
            VERBOSE=0,
            OUT_ALL=0,
            PF_DC=False,
            PF_ALG=config.pf_alg,
            PF_TOL=config.pf_tolerance,
            PF_MAX_IT=config.max_iterations,
            PF_MAX_IT_FD=config.max_iterations,
            PF_MAX_IT_GS=config.max_iterations,
            ENFORCE_Q_LIMS=(
                1 if config.q_limit_policy is QLimitPolicy.ENFORCE else 0
            ),
        )
        options["OPF_VIOLATION"] = config.generator_q_tolerance_mvar
        return options

    @staticmethod
    def _require_usable_next_state(state: GridFMState) -> None:
        assessment = assess_physical_state(state.metrics)
        if not assessment.all_values_finite:
            raise InvalidPhysicalState(
                "Power-flow result contains non-finite mandatory physical values."
            )

        arrays = {
            "bus_features": state.bus_features,
            "branch_features": state.branch_features,
            "edge_index": state.edge_index,
            "branch_ids": state.branch_ids,
            "branch_status": state.branch_status,
        }
        for name, values in arrays.items():
            if not np.isfinite(np.asarray(values)).all():
                raise InvalidPhysicalState(
                    f"Power-flow result contains NaN or infinity in {name}."
                )

    @staticmethod
    def _is_trusted_repeated_state(state: GridFMState) -> bool:
        """Return whether the state was produced by this backend representation."""

        return isinstance(state, _GeneratorOperatingPointState)

    @staticmethod
    def _resolve_branch_status_action(
        *,
        action: GridFMAction | None,
        switched_off_branch_id: int | None,
    ) -> tuple[int | None, int | None]:
        if action is not None and switched_off_branch_id is not None:
            raise ValueError(
                "Pass either action or switched_off_branch_id, not both."
            )

        if action is None:
            if switched_off_branch_id is None:
                return None, None
            return int(switched_off_branch_id), 0

        if action.kind != "set_branch_status":
            raise ValueError(
                "Power-flow backend accepts only branch-status topology actions."
            )
        if action.branch_id is None or action.target_status is None:
            raise ValueError(
                "Branch-status action is missing its branch target or target_status."
            )
        return int(action.branch_id), int(action.target_status)

    @staticmethod
    def _apply_branch_status(
        branch_df: pd.DataFrame,
        *,
        branch_id: int,
        target_status: int,
        context: str,
    ) -> None:
        if target_status not in (0, 1):
            raise ValueError("target_status must be either 0 or 1.")

        mask = branch_df["idx"].astype(int) == int(branch_id)
        match_count = int(mask.sum())
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one branch id {branch_id} in {context}, "
                f"found {match_count}."
            )

        current_status = int(
            float(branch_df.loc[mask, "br_status"].iloc[0]) > 0.5
        )
        if current_status == target_status:
            raise ValueError(
                f"Branch id {branch_id} already has status {target_status} "
                f"in {context}."
            )
        branch_df.loc[mask, "br_status"] = float(target_status)

    def run_power_flow(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None = None,
    ) -> GridFMPowerFlowResult:
        """Run AC power flow for a source scenario."""

        try:
            ppc, frames = self._build_ppc(
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
            )
            result_ppc, metrics = self._solve_ppc(
                ppc,
                context=f"scenario={scenario_id}",
            )
            next_state = self._build_state_from_pypower_result(
                scenario_id=scenario_id,
                result_ppc=result_ppc,
                original_frames=frames,
                physical_metrics=metrics,
            )
            self._require_usable_next_state(next_state)
            return GridFMPowerFlowResult(
                success=True,
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
                next_state=next_state,
                raw_result=result_ppc,
                message="Power flow converged.",
            )
        except PowerFlowNotConverged as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
                next_state=None,
                raw_result=None,
                message=str(exc),
                failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
            )
        except InvalidPhysicalState as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
                next_state=None,
                raw_result=None,
                message=str(exc),
                failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
            )

    def _build_ppc(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        bus_df = self.adapter.bus_df[
            self.adapter.bus_df["scenario"] == scenario_id
        ].copy()
        branch_df = self.adapter.branch_df[
            self.adapter.branch_df["scenario"] == scenario_id
        ].copy()
        gen_df = self.adapter.gen_df[
            self.adapter.gen_df["scenario"] == scenario_id
        ].copy()

        if bus_df.empty:
            raise ValueError(f"Scenario {scenario_id} not found in bus_data.")
        if branch_df.empty:
            raise ValueError(f"Scenario {scenario_id} not found in branch_data.")
        if gen_df.empty:
            raise ValueError(f"Scenario {scenario_id} not found in gen_data.")

        bus_df = bus_df.sort_values("bus").reset_index(drop=True)
        branch_df = branch_df.sort_values("idx").reset_index(drop=True)
        gen_df = gen_df.sort_values("idx").reset_index(drop=True)

        if switched_off_branch_id is not None:
            mask = branch_df["idx"].astype(int) == int(switched_off_branch_id)
            if not mask.any():
                raise ValueError(
                    f"Branch id {switched_off_branch_id} not found "
                    f"in scenario {scenario_id}."
                )
            branch_df.loc[mask, "br_status"] = 0.0

        return (
            {
                "version": "2",
                "baseMVA": self.base_mva,
                "bus": self._build_bus_matrix(bus_df),
                "branch": self._build_branch_matrix(branch_df),
                "gen": self._build_gen_matrix(gen_df, bus_df),
            },
            {"bus": bus_df, "branch": branch_df, "gen": gen_df},
        )

    def _build_bus_matrix(self, bus_df: pd.DataFrame) -> np.ndarray:
        bus = np.zeros((len(bus_df), 13), dtype=float)
        bus[:, BUS_I] = bus_df["bus"].to_numpy(dtype=float)
        bus[:, BUS_TYPE] = self._infer_bus_types(bus_df)
        bus[:, PD] = bus_df["Pd"].to_numpy(dtype=float)
        bus[:, QD] = bus_df["Qd"].to_numpy(dtype=float)
        bus[:, GS] = bus_df["GS"].to_numpy(dtype=float) * self.base_mva
        bus[:, BS] = bus_df["BS"].to_numpy(dtype=float) * self.base_mva
        bus[:, BUS_AREA] = 1.0
        bus[:, VM] = bus_df["Vm"].to_numpy(dtype=float)
        bus[:, VA] = bus_df["Va"].to_numpy(dtype=float)
        bus[:, BASE_KV] = bus_df["vn_kv"].to_numpy(dtype=float)
        bus[:, ZONE] = 1.0
        bus[:, VMAX] = bus_df["max_vm_pu"].to_numpy(dtype=float)
        bus[:, VMIN] = bus_df["min_vm_pu"].to_numpy(dtype=float)
        return bus

    @staticmethod
    def _infer_bus_types(bus_df: pd.DataFrame) -> np.ndarray:
        bus_types = np.full(len(bus_df), BUS_TYPE_PQ, dtype=float)
        if "PV" in bus_df.columns:
            bus_types[
                bus_df["PV"].to_numpy(dtype=float) > 0.5
            ] = BUS_TYPE_PV
        if "REF" in bus_df.columns:
            bus_types[
                bus_df["REF"].to_numpy(dtype=float) > 0.5
            ] = BUS_TYPE_REF
        return bus_types

    def _build_branch_matrix(self, branch_df: pd.DataFrame) -> np.ndarray:
        branch = np.zeros((len(branch_df), 13), dtype=float)
        branch[:, F_BUS] = branch_df["from_bus"].to_numpy(dtype=float)
        branch[:, T_BUS] = branch_df["to_bus"].to_numpy(dtype=float)
        branch[:, BR_R] = branch_df["r"].to_numpy(dtype=float)
        branch[:, BR_X] = branch_df["x"].to_numpy(dtype=float)
        branch[:, BR_B] = branch_df["b"].to_numpy(dtype=float)
        rate_a = branch_df["rate_a"].to_numpy(dtype=float)
        branch[:, RATE_A] = rate_a
        branch[:, RATE_B] = rate_a
        branch[:, RATE_C] = rate_a
        branch[:, TAP] = branch_df["tap"].to_numpy(dtype=float)
        branch[:, SHIFT] = branch_df["shift"].to_numpy(dtype=float)
        branch[:, BR_STATUS] = branch_df["br_status"].to_numpy(dtype=float)
        branch[:, ANGMIN] = branch_df["ang_min"].to_numpy(dtype=float)
        branch[:, ANGMAX] = branch_df["ang_max"].to_numpy(dtype=float)
        return branch

    def _build_gen_matrix(
        self,
        gen_df: pd.DataFrame,
        bus_df: pd.DataFrame,
    ) -> np.ndarray:
        gen = np.zeros((len(gen_df), 21), dtype=float)
        gen[:, GEN_BUS] = gen_df["bus"].to_numpy(dtype=float)
        gen[:, PG] = gen_df["p_mw"].to_numpy(dtype=float)
        gen[:, QG] = gen_df["q_mvar"].to_numpy(dtype=float)
        gen[:, QMAX] = gen_df["max_q_mvar"].to_numpy(dtype=float)
        gen[:, QMIN] = gen_df["min_q_mvar"].to_numpy(dtype=float)

        bus_vm_by_id = dict(
            zip(
                bus_df["bus"].astype(int).values,
                bus_df["Vm"].astype(float).values,
            )
        )
        gen[:, VG] = [
            bus_vm_by_id.get(int(bus_id), 1.0)
            for bus_id in gen_df["bus"].values
        ]
        gen[:, MBASE] = self.base_mva
        gen[:, GEN_STATUS] = gen_df["in_service"].to_numpy(dtype=float)
        gen[:, PMAX] = gen_df["max_p_mw"].to_numpy(dtype=float)
        gen[:, PMIN] = gen_df["min_p_mw"].to_numpy(dtype=float)
        return gen

    @staticmethod
    def _generator_operating_point(
        state: GridFMState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        operating_point = GeneratorOperatingPoint.from_state(state)
        if operating_point is None:
            return None
        return (
            operating_point.generator_ids,
            operating_point.p_mw,
            operating_point.q_mvar,
            operating_point.status,
        )

    @staticmethod
    def _with_generator_operating_point(
        state: GridFMState,
        *,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
    ) -> GridFMState:
        gen_df = original_frames["gen"]
        gen_result = np.asarray(result_ppc["gen"])

        return _GeneratorOperatingPointState(
            scenario_id=state.scenario_id,
            load_scenario_idx=state.load_scenario_idx,
            bus_features=state.bus_features,
            branch_features=state.branch_features,
            edge_index=state.edge_index,
            branch_ids=state.branch_ids,
            branch_status=state.branch_status,
            metrics=state.metrics,
            outaged_branch_ids=state.outaged_branch_ids,
            bus_ids=state.bus_ids,
            generator_ids=gen_df["idx"].to_numpy(dtype=np.int64, copy=True),
            generator_p_mw=np.asarray(
                gen_result[:, PG], dtype=np.float64
            ).copy(),
            generator_q_mvar=np.asarray(
                gen_result[:, QG], dtype=np.float64
            ).copy(),
            generator_status=np.asarray(
                gen_result[:, GEN_STATUS], dtype=np.float64
            ).copy(),
        )

    def _scenario_problem_resources(
        self,
        scenario_id: int,
    ) -> tuple[ScenarioPowerFlowTemplate, dict[str, pd.DataFrame]]:
        scenario_id = int(scenario_id)
        template = self._active_problem_template
        frames = self._active_problem_frames
        if (
            template is not None
            and frames is not None
            and int(template.scenario_id) == scenario_id
        ):
            return template, frames

        bus_df = self.adapter.bus_df[
            self.adapter.bus_df["scenario"] == scenario_id
        ].sort_values("bus").reset_index(drop=True)
        branch_df = self.adapter.branch_df[
            self.adapter.branch_df["scenario"] == scenario_id
        ].sort_values("idx").reset_index(drop=True)
        gen_df = self.adapter.gen_df[
            self.adapter.gen_df["scenario"] == scenario_id
        ].sort_values("idx").reset_index(drop=True)

        if bus_df.empty:
            raise InvalidPhysicalState(
                f"Scenario {scenario_id} not found in bus_data."
            )
        if branch_df.empty:
            raise InvalidPhysicalState(
                f"Scenario {scenario_id} not found in branch_data."
            )
        if gen_df.empty:
            raise InvalidPhysicalState(
                f"Scenario {scenario_id} not found in gen_data."
            )

        template = build_scenario_power_flow_template(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
            base_mva=self.base_mva,
        )
        frames = {
            "bus": bus_df,
            "branch": branch_df,
            "gen": gen_df,
        }
        self._active_problem_template = template
        self._active_problem_frames = frames
        return template, frames

    def _build_trusted_power_flow_problem(
        self,
        *,
        template: ScenarioPowerFlowTemplate,
        state: _GeneratorOperatingPointState,
        action: GridFMAction | None,
        switched_off_branch_id: int | None,
    ) -> CanonicalPowerFlowProblem:
        """Rebuild a backend-produced state without revalidating its invariants."""

        bus = template.bus.copy()
        branch = template.branch.copy()
        gen = template.gen.copy()
        bus_features = np.asarray(state.bus_features)
        branch_features = np.asarray(state.branch_features)

        bus[:, BUS_TYPE] = BUS_TYPE_PQ
        pv = np.asarray(bus_features[:, _BUS_COL["PV"]], dtype=np.float64) > 0.5
        ref = np.asarray(bus_features[:, _BUS_COL["REF"]], dtype=np.float64) > 0.5
        bus[pv, BUS_TYPE] = BUS_TYPE_PV
        bus[ref, BUS_TYPE] = BUS_TYPE_REF
        bus[:, PD] = bus_features[:, _BUS_COL["Pd"]]
        bus[:, QD] = bus_features[:, _BUS_COL["Qd"]]
        bus[:, GS] = bus_features[:, _BUS_COL["GS"]] * template.base_mva
        bus[:, BS] = bus_features[:, _BUS_COL["BS"]] * template.base_mva
        bus[:, VM] = bus_features[:, _BUS_COL["Vm"]]
        bus[:, VA] = bus_features[:, _BUS_COL["Va"]]
        bus[:, BASE_KV] = bus_features[:, _BUS_COL["vn_kv"]]
        bus[:, VMAX] = bus_features[:, _BUS_COL["max_vm_pu"]]
        bus[:, VMIN] = bus_features[:, _BUS_COL["min_vm_pu"]]

        branch[:, BR_R] = branch_features[:, _BRANCH_COL["r"]]
        branch[:, BR_X] = branch_features[:, _BRANCH_COL["x"]]
        branch[:, BR_B] = branch_features[:, _BRANCH_COL["b"]]
        branch[:, TAP] = branch_features[:, _BRANCH_COL["tap"]]
        branch[:, SHIFT] = branch_features[:, _BRANCH_COL["shift"]]
        rate_a = branch_features[:, _BRANCH_COL["rate_a"]]
        branch[:, RATE_A] = rate_a
        branch[:, RATE_B] = rate_a
        branch[:, RATE_C] = rate_a
        branch[:, BR_STATUS] = branch_features[:, _BRANCH_COL["br_status"]]

        if action is not None:
            assert action.branch_pos is not None
            assert action.target_status is not None
            branch[int(action.branch_pos), BR_STATUS] = float(action.target_status)
        elif switched_off_branch_id is not None:
            positions = np.flatnonzero(
                template.branch_ids == int(switched_off_branch_id)
            )
            if positions.size != 1:
                raise ValueError(
                    f"Expected exactly one branch id {switched_off_branch_id}, "
                    f"found {positions.size}."
                )
            branch[int(positions[0]), BR_STATUS] = 0.0

        assert state.generator_p_mw is not None
        assert state.generator_q_mvar is not None
        assert state.generator_status is not None
        gen[:, PG] = state.generator_p_mw
        gen[:, QG] = state.generator_q_mvar
        gen[:, GEN_STATUS] = state.generator_status

        return CanonicalPowerFlowProblem(
            base_mva=float(template.base_mva),
            bus=bus,
            branch=branch,
            gen=gen,
        )

    def _build_ppc_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
        validated_action: bool = False,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        """Build the repeated AC problem without pandas reconstruction."""

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )
        template, frames = self._scenario_problem_resources(int(state.scenario_id))

        trusted_state = self._is_trusted_repeated_state(state)
        if trusted_state and (action is None or validated_action):
            problem = self._build_trusted_power_flow_problem(
                template=template,
                state=state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            )
        else:
            problem = build_power_flow_problem_from_state(
                template=template,
                state=state,
                branch_id=branch_id,
                target_status=target_status,
                generator_operating_point=GeneratorOperatingPoint.from_state(state),
            )
        return problem.to_ppc(), frames

    @staticmethod
    def _problem_from_ppc(ppc: dict[str, Any]) -> CanonicalPowerFlowProblem:
        return CanonicalPowerFlowProblem(
            base_mva=float(ppc["baseMVA"]),
            bus=np.asarray(ppc["bus"], dtype=np.float64),
            branch=np.asarray(ppc["branch"], dtype=np.float64),
            gen=np.asarray(ppc["gen"], dtype=np.float64),
        )

    def _state_from_solved_ppc(
        self,
        *,
        state: GridFMState,
        result_ppc: dict[str, Any],
        frames: dict[str, pd.DataFrame],
        metrics: dict[str, object],
    ) -> GridFMState:
        next_state = self._build_state_from_pypower_result_fast(
            scenario_id=int(state.scenario_id),
            result_ppc=result_ppc,
            previous_state=state,
            original_frames=frames,
            physical_metrics=metrics,
        )
        self._require_usable_next_state(next_state)
        return next_state

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
        validated_action: bool = False,
    ) -> GridFMPowerFlowResult:
        """Run one transition, reusing only an exactly identical AC problem."""

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )
        effective_switched_off = branch_id if target_status == 0 else None
        context = f"scenario={state.scenario_id} from_state"
        trusted_state = self._is_trusted_repeated_state(state)

        try:
            ppc, frames = self._build_ppc_from_state(
                state=state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
                validated_action=validated_action,
            )
            problem = self._problem_from_ppc(ppc)

            cache_key: PowerFlowCacheKey | None = None
            cached = None
            if self.enable_cache:
                cache_key, cached = self._exact_power_flow_cache.lookup(
                    problem,
                    physics_fingerprint=self.physics_config.fingerprint(),
                )

            if isinstance(cached, CachedPowerFlowFailure):
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=effective_switched_off,
                    next_state=None,
                    raw_result=None,
                    message=cached.message,
                    failure_kind=cached.failure_kind,
                    switched_branch_id=branch_id,
                    target_status=target_status,
                )

            if isinstance(cached, CachedPowerFlowSuccess):
                assert cache_key is not None
                cached_ppc = cached.to_ppc(
                    base_mva=problem.base_mva,
                    copy_arrays=bool(self.store_raw_result),
                )
                try:
                    metrics = self._cached_physical_metrics(cache_key)
                    if metrics is None:
                        metrics = calculate_physical_metrics_from_result(
                            cached_ppc,
                            power_flow_converged=True,
                            physics_config=self.physics_config,
                        )
                        self._store_physical_metrics(cache_key, metrics)
                    next_state = self._state_from_solved_ppc(
                        state=state,
                        result_ppc=cached_ppc,
                        frames=frames,
                        metrics=metrics,
                    )
                except InvalidPhysicalState:
                    self._exact_power_flow_cache.discard(cache_key)
                    self._discard_physical_metrics(cache_key)
                else:
                    return GridFMPowerFlowResult(
                        success=True,
                        scenario_id=int(state.scenario_id),
                        switched_off_branch_id=effective_switched_off,
                        next_state=next_state,
                        raw_result=(
                            cached_ppc if self.store_raw_result else None
                        ),
                        message="Power flow converged.",
                        switched_branch_id=branch_id,
                        target_status=target_status,
                    )

            try:
                result_ppc, metrics = self._solve_ppc(
                    ppc,
                    context=context,
                    validate_input=not trusted_state,
                )
            except PowerFlowNotConverged as exc:
                if self.enable_cache and cache_key is not None:
                    self._exact_power_flow_cache.store_not_converged(
                        cache_key,
                        str(exc),
                    )
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=effective_switched_off,
                    next_state=None,
                    raw_result=None,
                    message=str(exc),
                    failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
                    switched_branch_id=branch_id,
                    target_status=target_status,
                )

            next_state = self._state_from_solved_ppc(
                state=state,
                result_ppc=result_ppc,
                frames=frames,
                metrics=metrics,
            )

            if self.enable_cache and cache_key is not None:
                stored = self._exact_power_flow_cache.store_success(
                    cache_key,
                    result_ppc,
                )
                if stored:
                    self._store_physical_metrics(cache_key, metrics)

            return GridFMPowerFlowResult(
                success=True,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=effective_switched_off,
                next_state=next_state,
                raw_result=result_ppc if self.store_raw_result else None,
                message="Power flow converged.",
                switched_branch_id=branch_id,
                target_status=target_status,
            )

        except InvalidPhysicalState as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=effective_switched_off,
                next_state=None,
                raw_result=None,
                message=str(exc),
                failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
                switched_branch_id=branch_id,
                target_status=target_status,
            )

    def _solve_ppc(
        self,
        ppc: dict[str, Any],
        *,
        context: str,
        validate_input: bool = True,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        """Solve through this module so monkeypatched runpf stays observable."""

        if validate_input:
            validate_ppc_input(ppc, self.physics_config, context=context)
        before = get_power_flow_workload_counters()

        try:
            result_ppc, success = runpf(ppc, self._build_pp_options())
        finally:
            after = get_power_flow_workload_counters()
            self.stock_runpf_calls += max(
                int(after["stock_runpf_calls"])
                - int(before["stock_runpf_calls"]),
                0,
            )
            self.q_limit_resolves += max(
                int(after["q_limit_resolves"])
                - int(before["q_limit_resolves"]),
                0,
            )

        if not bool(success):
            raise PowerFlowNotConverged(
                f"PYPOWER power flow did not converge ({context})."
            )

        metrics = calculate_physical_metrics_from_result(
            result_ppc,
            power_flow_converged=True,
            physics_config=self.physics_config,
        )
        return result_ppc, metrics

    def _build_state_from_pypower_result(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        state = self._build_canonical_state(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )
        return self._with_generator_operating_point(
            state,
            result_ppc=result_ppc,
            original_frames=original_frames,
        )

    def _build_state_from_pypower_result_fast(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        """Build a repeated transition through the NumPy-oriented path."""

        bus_res = result_ppc["bus"]
        branch_res = result_ppc["branch"]
        if physical_metrics is None:
            physical_metrics = calculate_physical_metrics_from_result(
                result_ppc,
                power_flow_converged=True,
                physics_config=self.physics_config,
            )

        bus_features = previous_state.bus_features.copy()
        branch_features = previous_state.branch_features.copy()

        vm = bus_res[:, VM].astype(np.float32)
        va = bus_res[:, VA].astype(np.float32)
        bus_features[:, _BUS_COL["Vm"]] = vm
        bus_features[:, _BUS_COL["Va"]] = va

        pf64 = np.asarray(branch_res[:, PF], dtype=np.float64)
        qf64 = np.asarray(branch_res[:, QF], dtype=np.float64)
        pt64 = np.asarray(branch_res[:, PT], dtype=np.float64)
        qt64 = np.asarray(branch_res[:, QT], dtype=np.float64)
        if not all(
            np.isfinite(values).all()
            for values in (pf64, qf64, pt64, qt64)
        ):
            raise InvalidPhysicalState(
                "Branch flow result contains non-finite values."
            )

        float32_max = np.finfo(np.float32).max
        if any(
            np.any(np.abs(values) > float32_max)
            for values in (pf64, qf64, pt64, qt64)
        ):
            raise InvalidPhysicalState(
                "Branch flow cannot be represented in feature precision."
            )

        pf = pf64.astype(np.float32)
        qf = qf64.astype(np.float32)
        pt = pt64.astype(np.float32)
        qt = qt64.astype(np.float32)
        s_from64 = np.hypot(pf64, qf64)
        s_to64 = np.hypot(pt64, qt64)
        s_max64 = np.maximum(s_from64, s_to64)
        rate_a64 = np.asarray(branch_res[:, RATE_A], dtype=np.float64)
        status64 = np.asarray(branch_res[:, BR_STATUS], dtype=np.float64)

        if not np.isfinite(rate_a64).all():
            raise InvalidPhysicalState(
                "Branch RATE_A contains non-finite values."
            )
        if not np.isfinite(status64).all():
            raise InvalidPhysicalState(
                "Branch status contains non-finite values."
            )
        if np.any((status64 != 0.0) & (status64 != 1.0)):
            raise InvalidPhysicalState(
                "Branch status must contain only 0 or 1."
            )

        br_status = status64.astype(np.float32)
        active = br_status > 0.0
        rated = active & (rate_a64 > 0.0)
        unlimited = active & (rate_a64 == 0.0)
        if np.any(active & (rate_a64 < 0.0)):
            raise InvalidPhysicalState(
                "Active branch RATE_A must be non-negative."
            )

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rate_a = rate_a64.astype(np.float32)
        if not np.isfinite(rate_a).all():
            raise InvalidPhysicalState(
                "Branch RATE_A cannot be represented in feature precision."
            )
        if np.any((rate_a64 > 0.0) & (rate_a == 0.0)):
            raise InvalidPhysicalState(
                "Positive RATE_A underflows to zero in feature precision."
            )
        if (
            not np.isfinite(s_from64[active]).all()
            or not np.isfinite(s_to64[active]).all()
        ):
            raise InvalidPhysicalState(
                "Active branch flow magnitude is non-finite."
            )
        if (
            self.physics_config.zero_rate_a_policy.value == "error"
            and unlimited.any()
        ):
            raise InvalidPhysicalState(
                "Active branch RATE_A=0 is forbidden by policy."
            )

        loading64 = np.zeros_like(s_max64)
        loading64[rated] = s_max64[rated] / rate_a64[rated] * 100.0
        if not np.isfinite(loading64[rated]).all():
            raise InvalidPhysicalState(
                "Active rated branch loading is non-finite."
            )

        s_from = s_from64.astype(np.float32)
        s_to = s_to64.astype(np.float32)
        s_max = s_max64.astype(np.float32)
        loading = loading64.astype(np.float32)
        if (
            not np.isfinite(s_from[active]).all()
            or not np.isfinite(s_to[active]).all()
            or not np.isfinite(s_max[active]).all()
            or not np.isfinite(loading[rated]).all()
        ):
            raise InvalidPhysicalState(
                "Branch features cannot be represented finitely."
            )

        branch_features[:, _BRANCH_COL["pf"]] = pf
        branch_features[:, _BRANCH_COL["qf"]] = qf
        branch_features[:, _BRANCH_COL["pt"]] = pt
        branch_features[:, _BRANCH_COL["qt"]] = qt
        branch_features[:, _BRANCH_COL["rate_a"]] = rate_a
        branch_features[:, _BRANCH_COL["br_status"]] = br_status
        branch_features[:, _BRANCH_COL["s_from_mva"]] = s_from
        branch_features[:, _BRANCH_COL["s_to_mva"]] = s_to
        branch_features[:, _BRANCH_COL["s_max_mva"]] = s_max
        branch_features[:, _BRANCH_COL["loading_percent"]] = loading

        active_loading = loading[active]
        mean_loading = (
            float(np.mean(active_loading))
            if active_loading.size > 0
            else 0.0
        )
        outaged_mask = br_status <= 0.0
        metrics = {
            "num_buses": int(bus_features.shape[0]),
            "num_branches": int(branch_features.shape[0]),
            "mean_loading_percent": mean_loading,
            "min_vm_pu": float(np.min(vm)),
            "max_vm_pu": float(np.max(vm)),
            "num_outaged_branches": int(np.sum(outaged_mask)),
            **physical_metrics,
        }
        outaged_branch_ids = [
            int(branch_id)
            for branch_id in previous_state.branch_ids[outaged_mask]
        ]

        state = GridFMState(
            scenario_id=int(scenario_id),
            load_scenario_idx=float(previous_state.load_scenario_idx),
            bus_features=bus_features.astype(np.float32),
            branch_features=branch_features.astype(np.float32),
            edge_index=previous_state.edge_index,
            branch_ids=previous_state.branch_ids,
            branch_status=br_status.astype(np.float32),
            metrics=metrics,
            outaged_branch_ids=outaged_branch_ids,
        )

        bus_features = state.bus_features.copy()
        self._update_fast_generator_features(
            bus_features=bus_features,
            result_ppc=result_ppc,
            previous_state=previous_state,
            original_frames=original_frames,
        )

        branch_features = state.branch_features.copy()
        branch_result = np.asarray(result_ppc["branch"])
        rate_a = np.asarray(branch_result[:, RATE_A], dtype=np.float64)
        status = np.asarray(branch_result[:, BR_STATUS], dtype=np.float64)

        if np.any(rate_a < 0.0):
            raise InvalidPhysicalState("Branch RATE_A must be non-negative.")

        branch_features[:, _BRANCH_COL["unlimited_rating"]] = (
            rate_a == 0.0
        ).astype(np.float32)

        if not np.isfinite(bus_features).all():
            raise InvalidPhysicalState(
                "Bus features cannot be represented in float32."
            )
        if not np.isfinite(branch_features).all():
            raise InvalidPhysicalState(
                "Branch features cannot be represented in float32."
            )

        rated = (status > 0.0) & (rate_a > 0.0)
        loading = branch_features[:, _BRANCH_COL["loading_percent"]]
        vm = np.asarray(result_ppc["bus"][:, VM], dtype=np.float64)

        metrics = dict(state.metrics)
        metrics["num_generators"] = int(len(original_frames["gen"]))
        metrics["mean_loading_percent"] = (
            float(np.mean(loading[rated]))
            if np.any(rated)
            else 0.0
        )
        metrics["min_vm_pu"] = float(np.min(vm))
        metrics["max_vm_pu"] = float(np.max(vm))

        state = replace(
            state,
            bus_features=bus_features,
            branch_features=branch_features,
            metrics=metrics,
            bus_ids=previous_state.bus_ids,
        )
        return self._with_generator_operating_point(
            state,
            result_ppc=result_ppc,
            original_frames=original_frames,
        )

    @staticmethod
    def _update_fast_generator_features(
        *,
        bus_features: np.ndarray,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
    ) -> None:
        gen_df = original_frames["gen"]
        missing = set(_GENERATOR_COLUMNS) - set(gen_df.columns)
        if missing:
            raise InvalidPhysicalState(
                f"Generator data is missing schema columns: {sorted(missing)}."
            )

        gen_result = np.asarray(result_ppc["gen"])
        if gen_result.ndim != 2 or gen_result.shape[0] != len(gen_df):
            raise InvalidPhysicalState(
                "PYPOWER gen result does not match the source frame."
            )

        if previous_state.bus_ids is not None:
            bus_ids = np.asarray(previous_state.bus_ids, dtype=np.int64)
        else:
            bus_ids = original_frames["bus"]["bus"].to_numpy(dtype=np.int64)

        if len(bus_ids) != bus_features.shape[0]:
            raise InvalidPhysicalState(
                "Bus IDs do not match the source state."
            )

        bus_pos = {
            int(bus_id): position
            for position, bus_id in enumerate(bus_ids)
        }
        try:
            gen_bus_pos = np.asarray(
                [
                    bus_pos[int(bus_id)]
                    for bus_id in gen_df["bus"].to_numpy(dtype=np.int64)
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise InvalidPhysicalState(
                f"Generator references unknown bus {exc.args[0]}."
            ) from exc

        status = np.asarray(gen_result[:, GEN_STATUS], dtype=np.float64)
        if not np.isfinite(status).all():
            raise InvalidPhysicalState(
                "Generator in_service contains NaN or infinity."
            )
        if np.any((status != 0.0) & (status != 1.0)):
            raise InvalidPhysicalState(
                "Generator in_service must contain only 0 or 1."
            )

        pg = np.asarray(gen_result[:, PG], dtype=np.float64)
        qg = np.asarray(gen_result[:, QG], dtype=np.float64)
        p_min = gen_df["min_p_mw"].to_numpy(dtype=np.float64)
        p_max = gen_df["max_p_mw"].to_numpy(dtype=np.float64)
        q_min = gen_df["min_q_mvar"].to_numpy(dtype=np.float64)
        q_max = gen_df["max_q_mvar"].to_numpy(dtype=np.float64)

        active = status > 0.0
        numeric = np.column_stack((pg, qg, p_min, p_max, q_min, q_max))
        if active.any() and not np.isfinite(numeric[active]).all():
            raise InvalidPhysicalState(
                "Active generator limits and outputs must be finite."
            )

        n_bus = bus_features.shape[0]
        active_pos = gen_bus_pos[active]
        sums = {
            "Pg": np.zeros(n_bus, dtype=np.float64),
            "Qg": np.zeros(n_bus, dtype=np.float64),
            "gen_online_count": np.zeros(n_bus, dtype=np.float64),
            "gen_p_min_mw": np.zeros(n_bus, dtype=np.float64),
            "gen_p_max_mw": np.zeros(n_bus, dtype=np.float64),
            "gen_q_min_mvar": np.zeros(n_bus, dtype=np.float64),
            "gen_q_max_mvar": np.zeros(n_bus, dtype=np.float64),
            "gen_p_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
            "gen_q_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
        }
        minima = {
            "gen_min_p_down_margin_mw": np.full(n_bus, np.inf),
            "gen_min_p_up_margin_mw": np.full(n_bus, np.inf),
            "gen_min_q_down_margin_mvar": np.full(n_bus, np.inf),
            "gen_min_q_up_margin_mvar": np.full(n_bus, np.inf),
        }

        if active.any():
            pg_a = pg[active]
            qg_a = qg[active]
            p_min_a = p_min[active]
            p_max_a = p_max[active]
            q_min_a = q_min[active]
            q_max_a = q_max[active]

            p_down = pg_a - p_min_a
            p_up = p_max_a - pg_a
            q_down = qg_a - q_min_a
            q_up = q_max_a - qg_a

            for name, values in (
                ("Pg", pg_a),
                ("Qg", qg_a),
                ("gen_p_min_mw", p_min_a),
                ("gen_p_max_mw", p_max_a),
                ("gen_q_min_mvar", q_min_a),
                ("gen_q_max_mvar", q_max_a),
            ):
                np.add.at(sums[name], active_pos, values)

            np.add.at(sums["gen_online_count"], active_pos, 1.0)
            np.add.at(
                sums["gen_p_limit_violation_count"],
                active_pos,
                ((p_down < 0.0) | (p_up < 0.0)).astype(np.float64),
            )
            np.add.at(
                sums["gen_q_limit_violation_count"],
                active_pos,
                ((q_down < 0.0) | (q_up < 0.0)).astype(np.float64),
            )
            for name, values in (
                ("gen_min_p_down_margin_mw", p_down),
                ("gen_min_p_up_margin_mw", p_up),
                ("gen_min_q_down_margin_mvar", q_down),
                ("gen_min_q_up_margin_mvar", q_up),
            ):
                np.minimum.at(minima[name], active_pos, values)

        available = sums["gen_online_count"] > 0.0
        for values in minima.values():
            values[~available] = 0.0

        sums.update(minima)
        sums["gen_available"] = available.astype(np.float64)
        sums["gen_p_down_margin_mw"] = (
            sums["Pg"] - sums["gen_p_min_mw"]
        )
        sums["gen_p_up_margin_mw"] = (
            sums["gen_p_max_mw"] - sums["Pg"]
        )
        sums["gen_q_down_margin_mvar"] = (
            sums["Qg"] - sums["gen_q_min_mvar"]
        )
        sums["gen_q_up_margin_mvar"] = (
            sums["gen_q_max_mvar"] - sums["Qg"]
        )

        for name, values in sums.items():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                feature = values.astype(np.float32)
            if not np.isfinite(feature).all():
                raise InvalidPhysicalState(
                    f"Bus feature {name} cannot be represented in float32."
                )
            bus_features[:, _BUS_COL[name]] = feature

    def _build_canonical_state(
        self,
        *,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None,
    ) -> GridFMState:
        return self._state_builder.build_from_pypower_result(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )
