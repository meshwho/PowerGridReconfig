from __future__ import annotations

import pandas as pd

from grid_topology_ai._data_adapter_core import *  # noqa: F401,F403
from grid_topology_ai._data_adapter_core import (
    GridFMAdapter as _CoreGridFMAdapter,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_frames,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.state_builder import GridFMState, GridFMStateBuilder
from grid_topology_ai.state_schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    STATE_FEATURE_SCHEMA_VERSION,
    finite_feature_matrix,
    state_feature_schema_fingerprint,
    state_feature_schema_payload,
    state_feature_schema_provenance,
    with_branch_rating_features,
    with_bus_generator_features,
)


# Preserve the public path used by pickled states and type displays.
GridFMState.__module__ = __name__


class GridFMAdapter(_CoreGridFMAdapter):
    """GridFM adapter that emits the versioned state representation."""

    def _validate_required_columns(self) -> None:
        try:
            super()._validate_required_columns()
        except ValueError as exc:
            raise InvalidPhysicalState(str(exc)) from exc

        required_bus = {"min_vm_pu", "max_vm_pu"}
        missing_bus = required_bus - set(self.bus_df.columns)
        if missing_bus:
            raise InvalidPhysicalState(
                f"Missing bus columns: {sorted(missing_bus)}"
            )

    @staticmethod
    def _add_branch_loading(
        branch_df: pd.DataFrame,
        physics_config: PhysicsConfig | None = None,
    ) -> pd.DataFrame:
        builder = GridFMStateBuilder(
            physics_config=physics_config or DEFAULT_PHYSICS_CONFIG,
        )
        return builder.add_branch_loading(branch_df)

    def build_summary(self) -> pd.DataFrame:
        """Build summaries using only active thermally rated branches."""

        summary = super().build_summary()

        for scenario_id in self.scenario_ids():
            branch = self.branch_df[
                self.branch_df["scenario"] == scenario_id
            ]
            active = branch[branch["br_status"] > 0.0]
            rated = active[active["rate_a"] > 0.0]
            unrated = active[active["rate_a"] == 0.0]
            row_mask = (
                summary["scenario"].astype(int) == int(scenario_id)
            )

            if len(rated):
                max_loading = float(rated["loading_percent"].max())
                mean_loading = float(rated["loading_percent"].mean())
            else:
                max_loading = 0.0
                mean_loading = 0.0

            summary.loc[row_mask, "max_loading_percent"] = max_loading
            summary.loc[row_mask, "mean_loading_percent"] = mean_loading
            summary.loc[
                row_mask,
                "num_unrated_active_branches",
            ] = int(len(unrated))

        if "num_unrated_active_branches" in summary.columns:
            summary["num_unrated_active_branches"] = (
                summary["num_unrated_active_branches"].astype(int)
            )

        return summary

    def build_state(self, scenario_id: int) -> GridFMState:
        """Build an initial state through the canonical state builder."""

        builder = GridFMStateBuilder(
            physics_config=self.physics_config,
            frame_metrics_calculator=(
                calculate_physical_metrics_from_frames
            ),
        )
        return builder.build_from_frames(
            scenario_id=scenario_id,
            bus_df=self.bus_df[
                self.bus_df["scenario"] == scenario_id
            ],
            branch_df=self.branch_df[
                self.branch_df["scenario"] == scenario_id
            ],
            gen_df=self.gen_df[
                self.gen_df["scenario"] == scenario_id
            ],
            power_flow_converged=False,
        )
