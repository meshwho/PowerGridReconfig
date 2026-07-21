from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from pypower.api import runpf

from grid_topology_ai._pypower_backend_core import *  # noqa: F401,F403
from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as _CoreGridFMPowerFlowBackend,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMState,
    UNRATED_LOADING_PERCENT,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.power_flow_errors import PowerFlowNotConverged


# Preserve the public module path used by pickled results and type displays.
GridFMPowerFlowResult.__module__ = __name__


class GridFMPowerFlowBackend(_CoreGridFMPowerFlowBackend):
    """PYPOWER backend with explicit active-unrated branch features."""

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

    def _build_state_from_pypower_result_fast(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        """
        Preserve the optimized builder while correcting active-unrated features.

        The core builder validates all flows and policies. This finalization
        step only replaces its ambiguous zero-percent encoding for active
        ``RATE_A == 0`` branches and recomputes the rated-only mean.
        """

        state = super()._build_state_from_pypower_result_fast(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            previous_state=previous_state,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )

        branch_col = {
            name: index
            for index, name in enumerate(BRANCH_FEATURE_COLUMNS)
        }
        features = state.branch_features.copy()
        rate_a = features[:, branch_col["rate_a"]]
        status = features[:, branch_col["br_status"]]
        active = status > 0.0
        rated = active & (rate_a > 0.0)
        unrated = active & (rate_a == 0.0)

        features[unrated, branch_col["loading_percent"]] = (
            UNRATED_LOADING_PERCENT
        )

        rated_loading = features[rated, branch_col["loading_percent"]]
        mean_loading = (
            float(np.mean(rated_loading))
            if rated_loading.size
            else 0.0
        )

        metrics = dict(state.metrics)
        metrics["mean_loading_percent"] = mean_loading
        return replace(
            state,
            branch_features=features.astype(np.float32),
            metrics=metrics,
        )

    @staticmethod
    def _build_state_from_frames(
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        physical_metrics: dict[str, object],
    ) -> GridFMState:
        """Build a slow-path state with a rated-only loading aggregate."""

        state = _CoreGridFMPowerFlowBackend._build_state_from_frames(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            physical_metrics=physical_metrics,
        )
        rated = branch_df[
            (branch_df["br_status"] > 0.0)
            & (branch_df["rate_a"] > 0.0)
        ]
        mean_loading = (
            float(rated["loading_percent"].mean())
            if len(rated)
            else 0.0
        )
        metrics = dict(state.metrics)
        metrics["mean_loading_percent"] = mean_loading
        return replace(state, metrics=metrics)
