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

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
    compute_voltage_violation_metrics,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
)
from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)
import copy

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
    """
    Result of applying one topology action and running AC power flow.
    """

    success: bool
    scenario_id: int
    switched_off_branch_id: int | None
    next_state: GridFMState | None
    raw_result: dict[str, Any] | None
    message: str


class GridFMPowerFlowBackend:
    """
    AC power flow backend for GridFM scenarios using PYPOWER.

    Main purpose:
        GridFMState + topology action
        -> MATPOWER/PYPOWER case
        -> AC power flow
        -> next GridFMState

    Why we need this:
        gridfm-datakit gives us emergency states.
        AlphaZero/RL needs transitions:
            state, action, next_state, reward, done
    """

    def __init__(
            self,
            adapter: GridFMAdapter,
            base_mva: float = 100.0,
            max_iter: int = 30,
            pf_alg: int = 1,
            enable_cache: bool = True,
            store_raw_result: bool = False,
    ):
        self.adapter = adapter
        self.base_mva = float(base_mva)
        self.max_iter = int(max_iter)
        self.pf_alg = int(pf_alg)

        self.enable_cache = bool(enable_cache)
        self.store_raw_result = bool(store_raw_result)

        # Cache stores only next_state, not full PYPOWER raw_result.
        self._cache: dict[tuple, GridFMState] = {}

        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self) -> None:
        """
        Clear cached power flow results.
        """

        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict:
        """
        Return cache statistics.
        """

        total = self.cache_hits + self.cache_misses

        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "enabled": self.enable_cache,
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    @staticmethod
    def _require_usable_next_state(state: GridFMState) -> None:
        """
        Reject a PF result that cannot safely enter search, replay, or a model.

        Physical metrics are calculated from the raw PYPOWER result. Feature
        tensors are checked separately because MCTS and neural evaluators consume
        them directly.
        """

        assessment = assess_physical_state(state.metrics)

        if not assessment.all_values_finite:
            raise ValueError(
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
            array = np.asarray(values)

            if not np.isfinite(array).all():
                raise ValueError(
                    f"Power-flow result contains NaN or infinity in {name}."
                )

    def _make_cache_key_from_state(
            self,
            state: GridFMState,
            switched_off_branch_id: int | None,
    ) -> tuple:
        """
        Build cache key for static topology-switching power flow.

        For the current MVP, loads and generation are fixed inside a scenario.
        Therefore the solved state is determined by:
            scenario_id
            pf algorithm
            sorted outaged branch IDs after applying the action

        Later, when redispatch or time-varying load is added, this key must also
        include load/generation fingerprints.
        """

        outaged = set(int(x) for x in state.outaged_branch_ids)

        if switched_off_branch_id is not None:
            outaged.add(int(switched_off_branch_id))

        return (
            int(state.scenario_id),
            int(self.pf_alg),
            round(float(self.base_mva), 6),
            tuple(sorted(outaged)),
        )

    def run_power_flow(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None = None,
    ) -> GridFMPowerFlowResult:
        """
        Run AC power flow for a scenario after optionally switching off one branch.

        Parameters
        ----------
        scenario_id:
            GridFM scenario ID.

        switched_off_branch_id:
            Branch idx to switch off.
            If None, the scenario is solved as-is.

        Returns
        -------
        GridFMPowerFlowResult
        """

        try:
            ppc, frames = self._build_ppc(
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
            )

            ppopt = ppoption(
                VERBOSE=0,
                OUT_ALL=0,
                PF_ALG=self.pf_alg,
                PF_MAX_IT=self.max_iter,
            )

            result_ppc, success = runpf(ppc, ppopt)

            success = bool(success)

            if not success:
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=scenario_id,
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=None,
                    raw_result=result_ppc,
                    message="PYPOWER power flow did not converge.",
                )

            next_state = self._build_state_from_pypower_result(
                scenario_id=scenario_id,
                result_ppc=result_ppc,
                original_frames=frames,
            )

            try:
                self._require_usable_next_state(next_state)
            except ValueError as exc:
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=scenario_id,
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=None,
                    raw_result=result_ppc,
                    message=f"Power flow returned an unusable state: {exc}",
                )

            return GridFMPowerFlowResult(
                success=True,
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
                next_state=next_state,
                raw_result=result_ppc,
                message="Power flow converged.",
            )

        except Exception as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=scenario_id,
                switched_off_branch_id=switched_off_branch_id,
                next_state=None,
                raw_result=None,
                message=f"Power flow backend failed: {exc}",
            )

    def _build_ppc(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        """
        Convert GridFM scenario dataframes into PYPOWER ppc format.
        """

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

        ppc = {
            "version": "2",
            "baseMVA": self.base_mva,
            "bus": self._build_bus_matrix(bus_df),
            "branch": self._build_branch_matrix(branch_df),
            "gen": self._build_gen_matrix(gen_df, bus_df),
        }

        frames = {
            "bus": bus_df,
            "branch": branch_df,
            "gen": gen_df,
        }

        return ppc, frames

    def _build_ppc_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        """
        Convert an already modified GridFMState into PYPOWER ppc format.

        This is the key method for multi-step control.

        It reconstructs bus_df and branch_df from state tensors and takes
        generator data from the original scenario stored in the adapter.
        """

        bus_df = self._state_to_bus_df(state)
        branch_df = self._state_to_branch_df(state)

        gen_df = self.adapter.gen_df[
            self.adapter.gen_df["scenario"] == int(state.scenario_id)
        ].copy()

        if gen_df.empty:
            raise ValueError(
                f"Scenario {state.scenario_id} not found in gen_data."
            )

        gen_df = gen_df.sort_values("idx").reset_index(drop=True)

        if switched_off_branch_id is not None:
            mask = branch_df["idx"].astype(int) == int(switched_off_branch_id)

            if not mask.any():
                raise ValueError(
                    f"Branch id {switched_off_branch_id} not found "
                    f"in current state for scenario {state.scenario_id}."
                )

            current_status = float(branch_df.loc[mask, "br_status"].iloc[0])

            if current_status <= 0:
                raise ValueError(
                    f"Branch id {switched_off_branch_id} is already out of service."
                )

            branch_df.loc[mask, "br_status"] = 0.0

        ppc = {
            "version": "2",
            "baseMVA": self.base_mva,
            "bus": self._build_bus_matrix(bus_df),
            "branch": self._build_branch_matrix(branch_df),
            "gen": self._build_gen_matrix(gen_df, bus_df),
        }

        frames = {
            "bus": bus_df,
            "branch": branch_df,
            "gen": gen_df,
        }

        return ppc, frames

    def _state_to_bus_df(self, state: GridFMState) -> pd.DataFrame:
        """
        Reconstruct bus dataframe from GridFMState.

        We take static columns such as min/max voltage limits from the original
        adapter data and update dynamic feature columns from state.bus_features.
        """

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
        """
        Reconstruct branch dataframe from GridFMState.

        We take static branch columns from the original adapter data and update
        dynamic branch feature columns from state.branch_features.
        """

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
    ) -> GridFMPowerFlowResult:
        """
        Run AC power flow from an already modified GridFMState.

        This method is required for multi-step topology switching.

        Difference from run_power_flow():
            run_power_flow() starts from the original GridFM scenario.
            run_power_flow_from_state() starts from the current state.

        Example:
            step 1:
                scenario 7 + switch off branch 122 -> state_1

            step 2:
                state_1 + switch off branch 154 -> state_2

        Without this method, every new action would incorrectly start again
        from the original scenario.
        """

        cache_key = self._make_cache_key_from_state(
            state=state,
            switched_off_branch_id=switched_off_branch_id,
        )

        if self.enable_cache and cache_key in self._cache:
            cached_next_state = self._cache[cache_key]

            try:
                self._require_usable_next_state(cached_next_state)
            except ValueError:
                # Invalid cached states must never re-enter MCTS.
                del self._cache[cache_key]
            else:
                self.cache_hits += 1

                return GridFMPowerFlowResult(
                    success=True,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=cached_next_state,
                    raw_result=None,
                    message="Power flow converged. [cache hit]",
                )

        if self.enable_cache:
            self.cache_misses += 1

        try:
            ppc, frames = self._build_ppc_from_state(
                state=state,
                switched_off_branch_id=switched_off_branch_id,
            )

            ppopt = ppoption(
                VERBOSE=0,
                OUT_ALL=0,
                PF_ALG=self.pf_alg,
                PF_MAX_IT=self.max_iter,
            )

            result_ppc, success = runpf(ppc, ppopt)
            success = bool(success)

            if not success:
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=None,
                    raw_result=result_ppc,
                    message="PYPOWER power flow did not converge.",
                )

            next_state = self._build_state_from_pypower_result_fast(
                scenario_id=int(state.scenario_id),
                result_ppc=result_ppc,
                previous_state=state,
                original_frames=frames,
            )

            try:
                self._require_usable_next_state(next_state)
            except ValueError as exc:
                return GridFMPowerFlowResult(
                    success=False,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=None,
                    raw_result=(
                        result_ppc if self.store_raw_result else None
                    ),
                    message=f"Power flow returned an unusable state: {exc}",
                )

            result = GridFMPowerFlowResult(
                success=True,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=switched_off_branch_id,
                next_state=next_state,
                raw_result=result_ppc if self.store_raw_result else None,
                message="Power flow converged.",
            )

            if self.enable_cache:
                self._cache[cache_key] = next_state

            return result

        except Exception as exc:
            return GridFMPowerFlowResult(
                success=False,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=switched_off_branch_id,
                next_state=None,
                raw_result=None,
                message=f"Power flow backend failed: {exc}",
            )

    def _build_bus_matrix(self, bus_df: pd.DataFrame) -> np.ndarray:
        """
        Build PYPOWER bus matrix.

        PYPOWER bus columns:
            BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA,
            BASE_KV, ZONE, VMAX, VMIN
        """

        bus = np.zeros((len(bus_df), 13), dtype=float)

        bus[:, BUS_I] = bus_df["bus"].to_numpy(dtype=float)
        bus[:, BUS_TYPE] = self._infer_bus_types(bus_df)
        bus[:, PD] = bus_df["Pd"].to_numpy(dtype=float)
        bus[:, QD] = bus_df["Qd"].to_numpy(dtype=float)
        bus[:, GS] = bus_df["GS"].to_numpy(dtype=float)
        bus[:, BS] = bus_df["BS"].to_numpy(dtype=float)
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
        """
        Infer PYPOWER bus types from one-hot GridFM columns PQ, PV, REF.
        """

        bus_types = np.full(len(bus_df), BUS_TYPE_PQ, dtype=float)

        if "PV" in bus_df.columns:
            bus_types[bus_df["PV"].to_numpy(dtype=float) > 0.5] = BUS_TYPE_PV

        if "REF" in bus_df.columns:
            bus_types[bus_df["REF"].to_numpy(dtype=float) > 0.5] = BUS_TYPE_REF

        return bus_types

    def _build_branch_matrix(self, branch_df: pd.DataFrame) -> np.ndarray:
        """
        Build PYPOWER branch matrix.

        Input branch matrix has 13 columns.
        PYPOWER will append PF/QF/PT/QT result columns after solving.
        """

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
        """
        Build PYPOWER generator matrix.

        We create 21 columns to be compatible with PYPOWER constants.
        """

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
    ) -> GridFMState:
        """
        Fast conversion from PYPOWER result to GridFMState.

        This avoids expensive pandas operations in _build_state_from_pypower_result:
            - DataFrame.copy()
            - DataFrame column assignments
            - groupby()
            - loc-based Pg/Qg updates
            - GridFMAdapter._add_branch_loading()

        Instead, it updates numpy feature arrays directly.
        """

        bus_res = result_ppc["bus"]
        branch_res = result_ppc["branch"]
        gen_res = result_ppc["gen"]
        physical_metrics = calculate_physical_metrics_from_result(
            result_ppc,
            power_flow_converged=True,
        )

        bus_features = previous_state.bus_features.copy()
        branch_features = previous_state.branch_features.copy()

        bus_col = {name: idx for idx, name in enumerate(BUS_FEATURE_COLUMNS)}
        branch_col = {name: idx for idx, name in enumerate(BRANCH_FEATURE_COLUMNS)}

        # ------------------------------------------------------------------
        # Bus dynamic features
        # ------------------------------------------------------------------

        vm = bus_res[:, VM].astype(np.float32)
        va = bus_res[:, VA].astype(np.float32)

        bus_features[:, bus_col["Vm"]] = vm
        bus_features[:, bus_col["Va"]] = va

        # Recompute bus-level Pg/Qg from generator results.
        pg_by_bus = np.zeros(bus_features.shape[0], dtype=np.float32)
        qg_by_bus = np.zeros(bus_features.shape[0], dtype=np.float32)

        bus_df = original_frames["bus"]
        gen_df = original_frames["gen"]

        bus_id_to_pos = {
            int(bus_id): pos
            for pos, bus_id in enumerate(bus_df["bus"].to_numpy(dtype=int))
        }

        gen_bus_ids = gen_df["bus"].to_numpy(dtype=int)

        for gen_pos, bus_id in enumerate(gen_bus_ids):
            bus_pos = bus_id_to_pos.get(int(bus_id))

            if bus_pos is None:
                continue

            pg_by_bus[bus_pos] += float(gen_res[gen_pos, PG])
            qg_by_bus[bus_pos] += float(gen_res[gen_pos, QG])

        bus_features[:, bus_col["Pg"]] = pg_by_bus
        bus_features[:, bus_col["Qg"]] = qg_by_bus

        # ------------------------------------------------------------------
        # Branch dynamic features
        # ------------------------------------------------------------------

        pf = branch_res[:, PF].astype(np.float32)
        qf = branch_res[:, QF].astype(np.float32)
        pt = branch_res[:, PT].astype(np.float32)
        qt = branch_res[:, QT].astype(np.float32)
        br_status = branch_res[:, BR_STATUS].astype(np.float32)

        s_from = np.sqrt(pf * pf + qf * qf).astype(np.float32)
        s_to = np.sqrt(pt * pt + qt * qt).astype(np.float32)
        s_max = np.maximum(s_from, s_to).astype(np.float32)

        rate_a = branch_features[:, branch_col["rate_a"]].astype(np.float32)

        loading = np.divide(
            s_max,
            rate_a,
            out=np.zeros_like(s_max, dtype=np.float32),
            where=rate_a > 0.0,
        ).astype(np.float32) * 100.0

        loading[br_status <= 0.0] = 0.0
        loading = np.nan_to_num(
            loading,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        branch_features[:, branch_col["pf"]] = pf
        branch_features[:, branch_col["qf"]] = qf
        branch_features[:, branch_col["pt"]] = pt
        branch_features[:, branch_col["qt"]] = qt
        branch_features[:, branch_col["br_status"]] = br_status
        branch_features[:, branch_col["s_from_mva"]] = s_from
        branch_features[:, branch_col["s_to_mva"]] = s_to
        branch_features[:, branch_col["s_max_mva"]] = s_max
        branch_features[:, branch_col["loading_percent"]] = loading

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

        active = br_status > 0.0
        active_loading = loading[active]

        if active_loading.size > 0:
            max_loading = float(np.max(active_loading))
            mean_loading = float(np.mean(active_loading))
        else:
            max_loading = 0.0
            mean_loading = 0.0

        vmin = bus_df["min_vm_pu"].to_numpy(dtype=np.float32)
        vmax = bus_df["max_vm_pu"].to_numpy(dtype=np.float32)

        low_voltage_violation = np.maximum(vmin - vm, 0.0)
        high_voltage_violation = np.maximum(vm - vmax, 0.0)

        total_low_voltage_violation = float(np.sum(low_voltage_violation))
        total_high_voltage_violation = float(np.sum(high_voltage_violation))
        total_voltage_violation = (
                total_low_voltage_violation + total_high_voltage_violation
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
    ) -> GridFMState:
        """
        Convert PYPOWER result back to GridFMState.
        """

        bus_df = original_frames["bus"].copy()
        branch_df = original_frames["branch"].copy()
        gen_df = original_frames["gen"].copy()

        bus_res = result_ppc["bus"]
        branch_res = result_ppc["branch"]
        gen_res = result_ppc["gen"]
        physical_metrics = calculate_physical_metrics_from_result(
            result_ppc,
            power_flow_converged=True,
        )

        bus_df["Vm"] = bus_res[:, VM]
        bus_df["Va"] = bus_res[:, VA]

        gen_df["p_mw"] = gen_res[:, PG]
        gen_df["q_mvar"] = gen_res[:, QG]
        gen_df["in_service"] = gen_res[:, GEN_STATUS]

        # Recompute bus-level Pg/Qg from generator results.
        bus_df["Pg"] = 0.0
        bus_df["Qg"] = 0.0

        gen_by_bus = gen_df.groupby("bus")[["p_mw", "q_mvar"]].sum()

        for bus_id, row in gen_by_bus.iterrows():
            mask = bus_df["bus"].astype(int) == int(bus_id)
            bus_df.loc[mask, "Pg"] = float(row["p_mw"])
            bus_df.loc[mask, "Qg"] = float(row["q_mvar"])

        branch_df["br_status"] = branch_res[:, BR_STATUS]
        branch_df["pf"] = branch_res[:, PF]
        branch_df["qf"] = branch_res[:, QF]
        branch_df["pt"] = branch_res[:, PT]
        branch_df["qt"] = branch_res[:, QT]

        branch_df = GridFMAdapter._add_branch_loading(branch_df)

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
        """
        Build GridFMState from updated bus/branch dataframes.
        """

        bus_df = bus_df.sort_values("bus").reset_index(drop=True)
        branch_df = branch_df.sort_values("idx").reset_index(drop=True)

        bus_features = bus_df[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        branch_features = branch_df[BRANCH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        edge_index = branch_df[["from_bus", "to_bus"]].to_numpy(dtype=np.int64).T

        branch_ids = branch_df["idx"].to_numpy(dtype=np.int64)
        branch_status = branch_df["br_status"].to_numpy(dtype=np.float32)

        in_service = branch_df[branch_df["br_status"] > 0]
        outaged = branch_df[branch_df["br_status"] <= 0]

        if len(in_service) > 0:
            max_loading = float(in_service["loading_percent"].max())
            mean_loading = float(in_service["loading_percent"].mean())
        else:
            max_loading = 0.0
            mean_loading = 0.0

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
            outaged_branch_ids=[int(x) for x in outaged["idx"].values],
        )
