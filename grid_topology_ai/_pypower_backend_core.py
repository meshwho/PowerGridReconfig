from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pypower.api import ppoption, runpf
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

from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    QLimitPolicy,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.power_flow_errors import (
    InvalidPhysicalState,
    PowerFlowFailureKind,
    PowerFlowNotConverged,
)
from grid_topology_ai.topology_actions import GridFMAction


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


class GridFMPowerFlowBackend:
    """Uncached AC power-flow implementation shared by public backends.

    This module contains only physical problem construction, PYPOWER execution,
    and state conversion. Cache identity, eviction, persistence, and memory
    policy belong to ``grid_topology_ai.cache`` and the public backend wrapper.
    """

    def __init__(
        self,
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        enable_cache: bool = True,
        store_raw_result: bool = False,
    ) -> None:
        self.adapter = adapter
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")
        self.physics_config = physics_config
        # The physical core does not implement caching. The public wrapper owns
        # the exact cache and consumes this runtime toggle.
        self.enable_cache = bool(enable_cache)
        self.store_raw_result = bool(store_raw_result)

    @property
    def base_mva(self) -> float:
        return self.physics_config.base_mva

    @property
    def max_iter(self) -> int:
        return self.physics_config.max_iterations

    @property
    def pf_alg(self) -> int:
        return self.physics_config.pf_alg

    def _build_pp_options(self) -> dict[str, object]:
        config = self.physics_config
        return ppoption(
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

    def _solve_ppc(
        self,
        ppc: dict[str, Any],
        *,
        context: str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        validate_ppc_input(ppc, self.physics_config, context=context)
        result_ppc, success = runpf(ppc, self._build_pp_options())
        if not bool(success):
            raise PowerFlowNotConverged(
                f"PYPOWER power flow did not converge ({context})."
            )
        validate_pypower_result(
            result_ppc,
            self.physics_config,
            input_ppc=ppc,
            context=context,
        )
        metrics = calculate_physical_metrics_from_result(
            result_ppc,
            power_flow_converged=True,
            physics_config=self.physics_config,
        )
        return result_ppc, metrics

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

    def _build_ppc_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        bus_df = self._state_to_bus_df(state)
        branch_df = self._state_to_branch_df(state)

        scenario_id = int(state.scenario_id)
        source_bus_df = self.adapter.bus_df[
            self.adapter.bus_df["scenario"] == scenario_id
        ].copy()
        gen_df = self.adapter.gen_df[
            self.adapter.gen_df["scenario"] == scenario_id
        ].copy()
        if source_bus_df.empty:
            raise ValueError(
                f"Scenario {state.scenario_id} not found in bus_data."
            )
        if gen_df.empty:
            raise ValueError(
                f"Scenario {state.scenario_id} not found in gen_data."
            )

        source_bus_df = source_bus_df.sort_values("bus").reset_index(drop=True)
        gen_df = gen_df.sort_values("idx").reset_index(drop=True)

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )
        if branch_id is not None:
            assert target_status is not None
            self._apply_branch_status(
                branch_df,
                branch_id=branch_id,
                target_status=target_status,
                context=f"current state for scenario {state.scenario_id}",
            )

        return (
            {
                "version": "2",
                "baseMVA": self.base_mva,
                "bus": self._build_bus_matrix(bus_df),
                "branch": self._build_branch_matrix(branch_df),
                # Generator voltage control remains a source-scenario setpoint.
                "gen": self._build_gen_matrix(gen_df, source_bus_df),
            },
            {"bus": bus_df, "branch": branch_df, "gen": gen_df},
        )

    def _state_to_bus_df(self, state: GridFMState) -> pd.DataFrame:
        bus_df = self.adapter.bus_df[
            self.adapter.bus_df["scenario"] == int(state.scenario_id)
        ].copy()
        if bus_df.empty:
            raise ValueError(
                f"Scenario {state.scenario_id} not found in bus_data."
            )
        bus_df = bus_df.sort_values("bus").reset_index(drop=True)
        if len(bus_df) != state.bus_features.shape[0]:
            raise ValueError(
                "Bus count mismatch between adapter bus_df and GridFMState."
            )
        for feature_idx, column_name in enumerate(BUS_FEATURE_COLUMNS):
            bus_df[column_name] = state.bus_features[:, feature_idx]
        return bus_df

    def _state_to_branch_df(self, state: GridFMState) -> pd.DataFrame:
        branch_df = self.adapter.branch_df[
            self.adapter.branch_df["scenario"] == int(state.scenario_id)
        ].copy()
        if branch_df.empty:
            raise ValueError(
                f"Scenario {state.scenario_id} not found in branch_data."
            )
        branch_df = branch_df.sort_values("idx").reset_index(drop=True)
        if len(branch_df) != state.branch_features.shape[0]:
            raise ValueError(
                "Branch count mismatch between adapter branch_df and GridFMState."
            )
        for feature_idx, column_name in enumerate(BRANCH_FEATURE_COLUMNS):
            branch_df[column_name] = state.branch_features[:, feature_idx]
        return branch_df

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> GridFMPowerFlowResult:
        """Run an uncached AC transition from an already solved state."""

        switched_branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )
        effective_switched_off = (
            switched_branch_id if target_status == 0 else None
        )

        try:
            ppc, frames = self._build_ppc_from_state(
                state=state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            )
            result_ppc, metrics = self._solve_ppc(
                ppc,
                context=f"scenario={state.scenario_id} from_state",
            )
            next_state = self._build_state_from_pypower_result_fast(
                scenario_id=int(state.scenario_id),
                result_ppc=result_ppc,
                previous_state=state,
                original_frames=frames,
                physical_metrics=metrics,
            )
            try:
                self._require_usable_next_state(next_state)
            except InvalidPhysicalState as exc:
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=effective_switched_off,
                    next_state=None,
                    raw_result=(
                        result_ppc if self.store_raw_result else None
                    ),
                    message=f"Power flow returned an unusable state: {exc}",
                    failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
                    switched_branch_id=switched_branch_id,
                    target_status=target_status,
                )

            return GridFMPowerFlowResult(
                success=True,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=effective_switched_off,
                next_state=next_state,
                raw_result=result_ppc if self.store_raw_result else None,
                message="Power flow converged.",
                switched_branch_id=switched_branch_id,
                target_status=target_status,
            )
        except PowerFlowNotConverged as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=effective_switched_off,
                next_state=None,
                raw_result=None,
                message=str(exc),
                failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
                switched_branch_id=switched_branch_id,
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
                switched_branch_id=switched_branch_id,
                target_status=target_status,
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

    def _build_state_from_pypower_result_fast(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        bus_res = result_ppc["bus"]
        branch_res = result_ppc["branch"]
        gen_res = result_ppc["gen"]
        if physical_metrics is None:
            physical_metrics = calculate_physical_metrics_from_result(
                result_ppc,
                power_flow_converged=True,
                physics_config=self.physics_config,
            )

        bus_features = previous_state.bus_features.copy()
        branch_features = previous_state.branch_features.copy()
        bus_col = {
            name: idx for idx, name in enumerate(BUS_FEATURE_COLUMNS)
        }
        branch_col = {
            name: idx for idx, name in enumerate(BRANCH_FEATURE_COLUMNS)
        }

        vm = bus_res[:, VM].astype(np.float32)
        va = bus_res[:, VA].astype(np.float32)
        bus_features[:, bus_col["Vm"]] = vm
        bus_features[:, bus_col["Va"]] = va

        pg_by_bus = np.zeros(bus_features.shape[0], dtype=np.float32)
        qg_by_bus = np.zeros(bus_features.shape[0], dtype=np.float32)
        bus_df = original_frames["bus"]
        gen_df = original_frames["gen"]
        bus_id_to_pos = {
            int(bus_id): pos
            for pos, bus_id in enumerate(
                bus_df["bus"].to_numpy(dtype=int)
            )
        }
        for gen_pos, bus_id in enumerate(
            gen_df["bus"].to_numpy(dtype=int)
        ):
            bus_pos = bus_id_to_pos.get(int(bus_id))
            if bus_pos is None:
                continue
            pg_by_bus[bus_pos] += float(gen_res[gen_pos, PG])
            qg_by_bus[bus_pos] += float(gen_res[gen_pos, QG])
        bus_features[:, bus_col["Pg"]] = pg_by_bus
        bus_features[:, bus_col["Qg"]] = qg_by_bus

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

        branch_features[:, branch_col["pf"]] = pf
        branch_features[:, branch_col["qf"]] = qf
        branch_features[:, branch_col["pt"]] = pt
        branch_features[:, branch_col["qt"]] = qt
        branch_features[:, branch_col["rate_a"]] = rate_a
        branch_features[:, branch_col["br_status"]] = br_status
        branch_features[:, branch_col["s_from_mva"]] = s_from
        branch_features[:, branch_col["s_to_mva"]] = s_to
        branch_features[:, branch_col["s_max_mva"]] = s_max
        branch_features[:, branch_col["loading_percent"]] = loading

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

        return GridFMState(
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

    def _build_state_from_pypower_result(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        bus_df = original_frames["bus"].copy()
        branch_df = original_frames["branch"].copy()
        gen_df = original_frames["gen"].copy()
        bus_res = result_ppc["bus"]
        branch_res = result_ppc["branch"]
        gen_res = result_ppc["gen"]
        if physical_metrics is None:
            physical_metrics = calculate_physical_metrics_from_result(
                result_ppc,
                power_flow_converged=True,
                physics_config=self.physics_config,
            )

        bus_df["Vm"] = bus_res[:, VM]
        bus_df["Va"] = bus_res[:, VA]
        gen_df["p_mw"] = gen_res[:, PG]
        gen_df["q_mvar"] = gen_res[:, QG]
        gen_df["in_service"] = gen_res[:, GEN_STATUS]
        bus_df["Pg"] = 0.0
        bus_df["Qg"] = 0.0

        gen_by_bus = gen_df.groupby("bus")[["p_mw", "q_mvar"]].sum()
        for bus_id, row in gen_by_bus.iterrows():
            mask = bus_df["bus"].astype(int) == int(bus_id)
            bus_df.loc[mask, "Pg"] = float(row["p_mw"])
            bus_df.loc[mask, "Qg"] = float(row["q_mvar"])

        branch_df["br_status"] = branch_res[:, BR_STATUS]
        branch_df["rate_a"] = branch_res[:, RATE_A]
        branch_df["pf"] = branch_res[:, PF]
        branch_df["qf"] = branch_res[:, QF]
        branch_df["pt"] = branch_res[:, PT]
        branch_df["qt"] = branch_res[:, QT]
        branch_df = GridFMAdapter._add_branch_loading(
            branch_df,
            physics_config=self.physics_config,
        )
        return self._build_state_from_frames(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            physical_metrics=physical_metrics,
        )

    @staticmethod
    def _build_state_from_frames(
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        physical_metrics: dict[str, object],
    ) -> GridFMState:
        bus_df = bus_df.sort_values("bus").reset_index(drop=True)
        branch_df = branch_df.sort_values("idx").reset_index(drop=True)
        bus_features = bus_df[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        branch_features = branch_df[
            BRANCH_FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)
        edge_index = branch_df[
            ["from_bus", "to_bus"]
        ].to_numpy(dtype=np.int64).T
        branch_ids = branch_df["idx"].to_numpy(dtype=np.int64)
        branch_status = branch_df["br_status"].to_numpy(dtype=np.float32)
        in_service = branch_df[branch_df["br_status"] > 0]
        outaged = branch_df[branch_df["br_status"] <= 0]
        mean_loading = (
            float(in_service["loading_percent"].mean())
            if len(in_service) > 0
            else 0.0
        )
        metrics = {
            "num_buses": int(len(bus_df)),
            "num_branches": int(len(branch_df)),
            "mean_loading_percent": mean_loading,
            "min_vm_pu": float(bus_df["Vm"].min()),
            "max_vm_pu": float(bus_df["Vm"].max()),
            "num_outaged_branches": int(len(outaged)),
            **physical_metrics,
        }
        return GridFMState(
            scenario_id=int(scenario_id),
            load_scenario_idx=float(bus_df["load_scenario_idx"].iloc[0]),
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            branch_ids=branch_ids,
            branch_status=branch_status,
            metrics=metrics,
            outaged_branch_ids=[int(value) for value in outaged["idx"].values],
        )
