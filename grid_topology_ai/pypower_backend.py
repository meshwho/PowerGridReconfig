from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, RATE_A
from pypower.idx_bus import VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai._pypower_backend_core import *  # noqa: F401,F403
from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as _CoreGridFMPowerFlowBackend,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
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
from grid_topology_ai.power_flow_errors import (
    InvalidPhysicalState,
    PowerFlowNotConverged,
)
from grid_topology_ai.power_flow_state_builder import PowerFlowStateBuilder
from grid_topology_ai.pypower_compat import (
    get_power_flow_workload_counters,
    runpf,
)
from grid_topology_ai.topology_actions import GridFMAction


_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_COL = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}
_GENERATOR_COLUMNS = (
    "bus",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
)


# Preserve the public module path used by pickled results and type displays.
GridFMPowerFlowResult.__module__ = __name__


class GridFMPowerFlowBackend(_CoreGridFMPowerFlowBackend):
    """PYPOWER backend with canonical initial and fast transition builders."""

    def __init__(
        self,
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        enable_cache: bool = True,
        store_raw_result: bool = False,
    ) -> None:
        self._require_matching_physics_contract(adapter, physics_config)
        super().__init__(
            adapter=adapter,
            physics_config=physics_config,
            enable_cache=enable_cache,
            store_raw_result=store_raw_result,
        )
        self._state_builder = PowerFlowStateBuilder(self.physics_config)
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0

    def performance_info(self) -> dict[str, object]:
        """Return backend-local cache and PYPOWER workload counters."""

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
        """Reset counters without discarding cached power-flow states."""

        self.cache_hits = 0
        self.cache_misses = 0
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0

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
        options = super()._build_pp_options()
        options["OPF_VIOLATION"] = self.physics_config.generator_q_tolerance_mvar
        return options

    def _make_cache_key_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> tuple:
        """Preserve the public legacy argument order."""

        return super()._make_cache_key_from_state(
            state=state,
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )

    def _build_ppc_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        """Build a case from either the legacy ID or a topology action."""

        if action is not None:
            switched_off_branch_id = None

        return super()._build_ppc_from_state(
            state=state,
            switched_off_branch_id=switched_off_branch_id,
            action=action,
        )

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> GridFMPowerFlowResult:
        """Run from a solved state while preserving both action APIs."""

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )

        result = super().run_power_flow_from_state(
            state=state,
            switched_off_branch_id=switched_off_branch_id,
            action=action,
        )

        return replace(
            result,
            switched_off_branch_id=(
                branch_id if target_status == 0 else None
            ),
            switched_branch_id=branch_id,
            target_status=target_status,
        )

    def _solve_ppc(
        self,
        ppc: dict[str, Any],
        *,
        context: str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        """
        Solve through this public module so monkeypatched ``runpf`` remains
        observable in tests and diagnostics after the implementation split.
        """

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

    def _build_state_from_pypower_result(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        return self._build_canonical_state(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )

    def _build_state_from_pypower_result_fast(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        """Build a repeated transition through the NumPy-oriented core path."""

        state = super()._build_state_from_pypower_result_fast(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            previous_state=previous_state,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
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

        return replace(
            state,
            bus_features=bus_features,
            branch_features=branch_features,
            metrics=metrics,
            bus_ids=previous_state.bus_ids,
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
        return self._state_builder.build(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )
