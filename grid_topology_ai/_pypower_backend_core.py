from __future__ import annotations

import hashlib
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
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig, QLimitPolicy
from grid_topology_ai.power_flow_errors import (
    InvalidPhysicalState, PowerFlowFailureKind, PowerFlowNotConverged,
)
from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    assess_physical_state,
)
from grid_topology_ai.state_fingerprint import physical_state_fingerprint
from grid_topology_ai.topology_actions import (
    GridFMAction,
)


_CACHE_BUS_INPUT_COLUMNS = (
    "Pd",
    "Qd",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "GS",
    "BS",
    "min_vm_pu",
    "max_vm_pu",
)
_CACHE_BRANCH_INPUT_COLUMNS = (
    "r",
    "x",
    "b",
    "tap",
    "shift",
    "rate_a",
)
_CACHE_BUS_INPUT_INDICES = tuple(
    BUS_FEATURE_COLUMNS.index(name)
    for name in _CACHE_BUS_INPUT_COLUMNS
)
_CACHE_BRANCH_INPUT_INDICES = tuple(
    BRANCH_FEATURE_COLUMNS.index(name)
    for name in _CACHE_BRANCH_INPUT_COLUMNS
)


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
    failure_kind: PowerFlowFailureKind | None = None
    switched_branch_id: int | None = None
    target_status: int | None = None


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
            physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
            enable_cache: bool = True,
            store_raw_result: bool = False,
    ):
        self.adapter = adapter
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")
        self.physics_config = physics_config

        self.enable_cache = bool(enable_cache)
        self.store_raw_result = bool(store_raw_result)

        # Cache stores only next_state, not full PYPOWER raw_result.
        self._cache: dict[tuple, GridFMState] = {}
        self._scenario_static_input_fingerprints: dict[int, str] = {}

        self.cache_hits = 0
        self.cache_misses = 0

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
        return ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=False, PF_ALG=config.pf_alg,
                        PF_TOL=config.pf_tolerance, PF_MAX_IT=config.max_iterations,
                        PF_MAX_IT_FD=config.max_iterations, PF_MAX_IT_GS=config.max_iterations,
                        ENFORCE_Q_LIMS=1 if config.q_limit_policy is QLimitPolicy.ENFORCE else 0)

    def _solve_ppc(self, ppc: dict[str, Any], *, context: str) -> tuple[dict[str, Any], dict[str, object]]:
        validate_ppc_input(ppc, self.physics_config, context=context)
        result_ppc, success = runpf(ppc, self._build_pp_options())
        if not bool(success):
            raise PowerFlowNotConverged(f"PYPOWER power flow did not converge ({context}).")
        validate_pypower_result(result_ppc, self.physics_config, input_ppc=ppc, context=context)
        metrics = calculate_physical_metrics_from_result(result_ppc, power_flow_converged=True, physics_config=self.physics_config)
        return result_ppc, metrics

    def clear_cache(self) -> None:
        """
        Clear cached power flow results.
        """

        self._cache.clear()
        self._scenario_static_input_fingerprints.clear()
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
            array = np.asarray(values)

            if not np.isfinite(array).all():
                raise InvalidPhysicalState(
                    f"Power-flow result contains NaN or infinity in {name}."
                )

    @staticmethod
    def _resolve_branch_status_action(
        *,
        action: GridFMAction | None,
        switched_off_branch_id: int | None,
    ) -> tuple[int | None, int | None]:
        if (
            action is not None
            and switched_off_branch_id is not None
        ):
            raise ValueError(
                "Pass either action or "
                "switched_off_branch_id, not both."
            )

        if action is None:
            if switched_off_branch_id is None:
                return None, None

            return (
                int(switched_off_branch_id),
                0,
            )

        if action.kind != "set_branch_status":
            raise ValueError(
                "Power-flow backend accepts only "
                "branch-status topology actions."
            )

        if (
            action.branch_id is None
            or action.target_status is None
        ):
            raise ValueError(
                "Branch-status action is missing its "
                "branch target or target_status."
            )

        return (
            int(action.branch_id),
            int(action.target_status),
        )

    @staticmethod
    def _apply_branch_status(
        branch_df: pd.DataFrame,
        *,
        branch_id: int,
        target_status: int,
        context: str,
    ) -> None:
        if target_status not in (0, 1):
            raise ValueError(
                "target_status must be either 0 or 1."
            )

        mask = (
            branch_df["idx"].astype(int)
            == int(branch_id)
        )

        match_count = int(mask.sum())

        if match_count != 1:
            raise ValueError(
                f"Expected exactly one branch id "
                f"{branch_id} in {context}, found "
                f"{match_count}."
            )

        current_status = int(
            float(
                branch_df.loc[
                    mask,
                    "br_status",
                ].iloc[0]
            )
            > 0.5
        )

        if current_status == target_status:
            raise ValueError(
                f"Branch id {branch_id} already has "
                f"status {target_status} in {context}."
            )

        branch_df.loc[
            mask,
            "br_status",
        ] = float(target_status)

    def _make_cache_key_from_state(
            self,
            state: GridFMState,
            *,
            action: GridFMAction | None = None,
            switched_off_branch_id: int | None = None,
    ) -> tuple:
        """Return the strict source-state cache identity kept for compatibility."""

        branch_id, target_status = (
            self._resolve_branch_status_action(
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            )
        )

        outaged = {
            int(branch_id)
            for branch_id
            in state.outaged_branch_ids
        }

        if branch_id is not None:
            if target_status == 0:
                outaged.add(branch_id)
            else:
                outaged.discard(branch_id)

        return (
            self.physics_config.fingerprint(),
            physical_state_fingerprint(state),
            branch_id,
            target_status,
            tuple(sorted(outaged)),
        )

    @staticmethod
    def _hash_array(digest: Any, values: Any) -> None:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    def _scenario_static_input_fingerprint(self, scenario_id: int) -> str:
        scenario_id = int(scenario_id)
        cache = getattr(self, "_scenario_static_input_fingerprints", None)
        if cache is None:
            cache = {}
            self._scenario_static_input_fingerprints = cache
        if scenario_id in cache:
            return cache[scenario_id]

        digest = hashlib.sha256()
        digest.update(f"scenario:{scenario_id}".encode("ascii"))

        bus_df = getattr(self.adapter, "bus_df", None)
        if isinstance(bus_df, pd.DataFrame) and "scenario" in bus_df.columns:
            buses = bus_df[
                bus_df["scenario"] == scenario_id
            ].sort_values("bus")
            if not buses.empty and {"bus", "Vm"}.issubset(buses.columns):
                self._hash_array(
                    digest,
                    buses[["bus", "Vm"]].to_numpy(dtype=np.float64),
                )

        gen_df = getattr(self.adapter, "gen_df", None)
        generator_columns = [
            "idx",
            "bus",
            "p_mw",
            "q_mvar",
            "max_q_mvar",
            "min_q_mvar",
            "in_service",
            "max_p_mw",
            "min_p_mw",
        ]
        if isinstance(gen_df, pd.DataFrame) and "scenario" in gen_df.columns:
            generators = gen_df[
                gen_df["scenario"] == scenario_id
            ].sort_values("idx")
            if not generators.empty:
                missing = set(generator_columns) - set(generators.columns)
                if missing:
                    raise ValueError(
                        "Generator data is missing power-flow cache columns: "
                        f"{sorted(missing)}."
                    )
                self._hash_array(
                    digest,
                    generators[generator_columns].to_numpy(dtype=np.float64),
                )

        fingerprint = digest.hexdigest()
        cache[scenario_id] = fingerprint
        return fingerprint

    def _power_flow_input_fingerprint(self, state: GridFMState) -> str:
        digest = hashlib.sha256()
        digest.update(
            self._scenario_static_input_fingerprint(
                int(state.scenario_id)
            ).encode("ascii")
        )

        bus_ids = getattr(state, "bus_ids", None)
        if bus_ids is not None:
            self._hash_array(
                digest,
                np.asarray(bus_ids, dtype=np.int64),
            )

        self._hash_array(
            digest,
            np.asarray(state.edge_index, dtype=np.int64),
        )
        self._hash_array(
            digest,
            np.asarray(state.bus_features)[
                :, _CACHE_BUS_INPUT_INDICES
            ],
        )
        self._hash_array(
            digest,
            np.asarray(state.branch_features)[
                :, _CACHE_BRANCH_INPUT_INDICES
            ],
        )
        return digest.hexdigest()

    def _resulting_topology_signature(
        self,
        state: GridFMState,
        *,
        action: GridFMAction | None = None,
        switched_off_branch_id: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        branch_ids = np.asarray(state.branch_ids, dtype=np.int64)
        statuses = (
            np.asarray(state.branch_status, dtype=np.float64) > 0.5
        ).astype(np.int8)

        if branch_ids.ndim != 1 or statuses.ndim != 1:
            raise ValueError("Branch ids and status must be one-dimensional.")
        if len(branch_ids) != len(statuses):
            raise ValueError("Branch ids and status length mismatch.")

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )

        if branch_id is not None:
            assert target_status is not None
            matches = np.flatnonzero(branch_ids == int(branch_id))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one branch id {branch_id} in state, "
                    f"found {len(matches)}."
                )
            branch_pos = int(matches[0])
            current_status = int(statuses[branch_pos])
            if current_status == target_status:
                raise ValueError(
                    f"Branch id {branch_id} already has status "
                    f"{target_status} in current state."
                )
            statuses = statuses.copy()
            statuses[branch_pos] = int(target_status)

        order = np.argsort(branch_ids, kind="stable")
        return tuple(
            (int(branch_ids[pos]), int(statuses[pos]))
            for pos in order
        )

    def _make_topology_cache_key_from_state(
        self,
        state: GridFMState,
        *,
        action: GridFMAction | None = None,
        switched_off_branch_id: int | None = None,
    ) -> tuple:
        return (
            self.physics_config.fingerprint(),
            int(state.scenario_id),
            self._power_flow_input_fingerprint(state),
            self._resulting_topology_signature(
                state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            ),
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

            result_ppc, metrics = self._solve_ppc(ppc, context=f"scenario={scenario_id}")

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
                message=str(exc), failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
            )
        except InvalidPhysicalState as exc:
            return GridFMPowerFlowResult(
                success=False, scenario_id=scenario_id, switched_off_branch_id=switched_off_branch_id,
                next_state=None, raw_result=None, message=str(exc),
                failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
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
            switched_off_branch_id: int | None = None,
            *,
            action: GridFMAction | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, pd.DataFrame],
    ]:
        """
        Convert an already modified GridFMState into PYPOWER ppc format.

        This is the key method for multi-step control.

        It reconstructs bus_df and branch_df from state tensors and takes
        generator data from the original scenario stored in the adapter.
        """

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

        branch_id, target_status = (
            self._resolve_branch_status_action(
                action=action,
                switched_off_branch_id=(
                    switched_off_branch_id
                ),
            )
        )

        if branch_id is not None:
            assert target_status is not None

            self._apply_branch_status(
                branch_df,
                branch_id=branch_id,
                target_status=target_status,
                context=(
                    "current state for scenario "
                    f"{state.scenario_id}"
                ),
            )

        ppc = {
            "version": "2",
            "baseMVA": self.base_mva,
            "bus": self._build_bus_matrix(bus_df),
            "branch": self._build_branch_matrix(branch_df),
            # Generator voltage control is a scenario input.  The solved Vm in
            # the parent state is only a warm-start value and must not become a
            # new setpoint after every topology action.
            "gen": self._build_gen_matrix(gen_df, source_bus_df),
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
            *,
            action: GridFMAction | None = None,
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

        switched_branch_id, target_status = (
            self._resolve_branch_status_action(
                action=action,
                switched_off_branch_id=(
                    switched_off_branch_id
                ),
            )
        )

        cache_key = self._make_topology_cache_key_from_state(
            state=state,
            action=action,
            switched_off_branch_id=(
                switched_off_branch_id
            ),
        )

        if self.enable_cache and cache_key in self._cache:
            cached_next_state = self._cache[cache_key]

            try:
                self._require_usable_next_state(cached_next_state)
            except InvalidPhysicalState:
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
                action=action,
                switched_off_branch_id=(
                    switched_branch_id
                    if target_status == 0
                    else None
                ),
            )

            result_ppc, metrics = self._solve_ppc(
                ppc, context=f"scenario={state.scenario_id} from_state"
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
                    switched_off_branch_id=switched_off_branch_id,
                    next_state=None,
                    raw_result=(
                        result_ppc if self.store_raw_result else None
                    ),
                    message=f"Power flow returned an unusable state: {exc}",
                    failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
                    switched_branch_id=switched_branch_id,
                    target_status=target_status,
                )

            result = GridFMPowerFlowResult(
                success=True,
                scenario_id=int(state.scenario_id),
                switched_off_branch_id=switched_off_branch_id,
                next_state=next_state,
                raw_result=result_ppc if self.store_raw_result else None,
                message="Power flow converged.",
                switched_branch_id=switched_branch_id,
                target_status=target_status,
            )

            if self.enable_cache:
                self._cache[cache_key] = next_state

            return result

        except PowerFlowNotConverged as exc:
            return GridFMPowerFlowResult(
                success=False, scenario_id=int(state.scenario_id),
                switched_off_branch_id=switched_off_branch_id, next_state=None,
                raw_result=None, message=str(exc),
                failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
                switched_branch_id=switched_branch_id,
                target_status=target_status,
            )
        except InvalidPhysicalState as exc:
            return GridFMPowerFlowResult(
                success=False, scenario_id=int(state.scenario_id),
                switched_off_branch_id=switched_off_branch_id, next_state=None,
                raw_result=None, message=str(exc),
                failure_kind=PowerFlowFailureKind.INVALID_PHYSICAL_STATE,
                switched_branch_id=switched_branch_id,
                target_status=target_status,
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
            physical_metrics: dict[str, object] | None = None,
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
        if physical_metrics is None:
            physical_metrics = calculate_physical_metrics_from_result(
                result_ppc, power_flow_converged=True,
                physics_config=self.physics_config,
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

        # Calculate electrical magnitudes in float64.  float32 squaring can
        # overflow finite PYPOWER flows and must never be sanitised to zero.
        pf64 = np.asarray(branch_res[:, PF], dtype=np.float64)
        qf64 = np.asarray(branch_res[:, QF], dtype=np.float64)
        pt64 = np.asarray(branch_res[:, PT], dtype=np.float64)
        qt64 = np.asarray(branch_res[:, QT], dtype=np.float64)
        if not all(
            np.isfinite(values).all()
            for values in (pf64, qf64, pt64, qt64)
        ):
            raise InvalidPhysicalState("Branch flow result contains non-finite values.")
        float32_max = np.finfo(np.float32).max
        if any(
            np.any(np.abs(values) > float32_max)
            for values in (pf64, qf64, pt64, qt64)
        ):
            raise InvalidPhysicalState("Branch flow cannot be represented in feature precision.")
        pf = pf64.astype(np.float32)
        qf = qf64.astype(np.float32)
        pt = pt64.astype(np.float32)
        qt = qt64.astype(np.float32)

        s_from64 = np.hypot(pf64, qf64)
        s_to64 = np.hypot(pt64, qt64)
        s_max64 = np.maximum(s_from64, s_to64)

        rate_a64 = np.asarray(
            branch_res[:, RATE_A],
            dtype=np.float64,
        )
        status64 = np.asarray(
            branch_res[:, BR_STATUS],
            dtype=np.float64,
        )

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

        if not np.isfinite(s_from64[active]).all() or not np.isfinite(s_to64[active]).all():
            raise InvalidPhysicalState("Active branch flow magnitude is non-finite.")
        if self.physics_config.zero_rate_a_policy.value == "error" and unlimited.any():
            raise InvalidPhysicalState("Active branch RATE_A=0 is forbidden by policy.")
        loading64 = np.zeros_like(s_max64)
        loading64[rated] = s_max64[rated] / rate_a64[rated] * 100.0
        if not np.isfinite(loading64[rated]).all():
            raise InvalidPhysicalState("Active rated branch loading is non-finite.")
        s_from = s_from64.astype(np.float32)
        s_to = s_to64.astype(np.float32)
        s_max = s_max64.astype(np.float32)
        loading = loading64.astype(np.float32)
        if not np.isfinite(s_from[active]).all() or not np.isfinite(s_to[active]).all() or not np.isfinite(s_max[active]).all() or not np.isfinite(loading[rated]).all():
            raise InvalidPhysicalState("Branch features cannot be represented finitely.")

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

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

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
            physical_metrics: dict[str, object] | None = None,
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
        if physical_metrics is None:
            physical_metrics = calculate_physical_metrics_from_result(
                result_ppc, power_flow_converged=True,
                physics_config=self.physics_config,
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
