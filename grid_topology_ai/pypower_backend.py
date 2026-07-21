from __future__ import annotations

from typing import Any

import pandas as pd
from pypower.api import runpf

from grid_topology_ai._pypower_backend_core import *  # noqa: F401,F403
from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as _CoreGridFMPowerFlowBackend,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
)
from grid_topology_ai.data_adapter import GridFMAdapter, GridFMState
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.power_flow_errors import PowerFlowNotConverged
from grid_topology_ai.power_flow_state_builder import PowerFlowStateBuilder


# Preserve the public module path used by pickled results and type displays.
GridFMPowerFlowResult.__module__ = __name__


class GridFMPowerFlowBackend(_CoreGridFMPowerFlowBackend):
    """PYPOWER backend with one canonical builder for every solved state."""

    def __init__(
        self,
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        enable_cache: bool = True,
        store_raw_result: bool = False,
    ) -> None:
        super().__init__(
            adapter=adapter,
            physics_config=physics_config,
            enable_cache=enable_cache,
            store_raw_result=store_raw_result,
        )
        self._state_builder = PowerFlowStateBuilder(self.physics_config)

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
        # Kept as a compatibility entry point; representation is intentionally
        # identical to the initial-state path.
        del previous_state
        return self._build_canonical_state(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )

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
