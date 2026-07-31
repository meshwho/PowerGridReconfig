from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from grid_topology_ai._data_adapter_core import *  # noqa: F401,F403
from grid_topology_ai._data_adapter_core import (
    GridFMAdapter as _CoreGridFMAdapter,
)
from grid_topology_ai._data_adapter_core import (
    GridFMState as _CoreGridFMState,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    ZeroRateAPolicy,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_frames,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
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
from grid_topology_ai.state_topology import validate_state_topology


# Retained as a public compatibility constant. Schema v2 no longer stores this
# sentinel in branch features; consumers must use ``unlimited_rating``.
UNRATED_LOADING_PERCENT = -1.0


@dataclass(frozen=True)
class GridFMState(_CoreGridFMState):
    """A graph state with original bus IDs stored beside contiguous edges."""

    bus_ids: np.ndarray | None = None


# Preserve the public module path used by pickled states and type displays.
GridFMState.__module__ = __name__


class GridFMAdapter(_CoreGridFMAdapter):
    """GridFM adapter that emits the versioned state feature schema."""

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
        """Add validated apparent-power, loading, and rating features."""

        config = physics_config or DEFAULT_PHYSICS_CONFIG
        df = branch_df.copy()

        try:
            pf = df["pf"].to_numpy(dtype=np.float64)
            qf = df["qf"].to_numpy(dtype=np.float64)
            pt = df["pt"].to_numpy(dtype=np.float64)
            qt = df["qt"].to_numpy(dtype=np.float64)
            rate_a = df["rate_a"].to_numpy(dtype=np.float64)
            status = df["br_status"].to_numpy(dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPhysicalState(
                "Branch flow data must contain numeric "
                "pf/qf/pt/qt/rate_a/br_status columns."
            ) from exc

        mandatory_arrays = {
            "pf": pf,
            "qf": qf,
            "pt": pt,
            "qt": qt,
            "rate_a": rate_a,
            "br_status": status,
        }

        for name, values in mandatory_arrays.items():
            if not np.isfinite(values).all():
                raise InvalidPhysicalState(
                    f"Branch column {name} contains NaN or infinity."
                )

        if not np.isin(status, (0.0, 1.0)).all():
            raise InvalidPhysicalState(
                "Branch status must contain only 0 or 1."
            )
        if np.any(rate_a < 0.0):
            raise InvalidPhysicalState(
                "Branch RATE_A must be non-negative."
            )

        active = status > 0.0
        rated = active & (rate_a > 0.0)
        unlimited = active & (rate_a == 0.0)

        if (
            config.zero_rate_a_policy is ZeroRateAPolicy.ERROR
            and unlimited.any()
        ):
            raise InvalidPhysicalState(
                "Active branch RATE_A=0 is forbidden by PhysicsConfig."
            )

        s_from = np.hypot(pf, qf)
        s_to = np.hypot(pt, qt)
        s_max = np.maximum(s_from, s_to)

        if not all(
            np.isfinite(values).all()
            for values in (s_from, s_to, s_max)
        ):
            raise InvalidPhysicalState(
                "Branch apparent-power magnitude is non-finite."
            )

        loading = np.zeros_like(s_max, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            loading[rated] = s_max[rated] / rate_a[rated] * 100.0

        if not np.isfinite(loading[rated]).all():
            raise InvalidPhysicalState(
                "Active rated branch loading is non-finite."
            )

        float32_features = {
            **mandatory_arrays,
            "s_from_mva": s_from,
            "s_to_mva": s_to,
            "s_max_mva": s_max,
            "loading_percent": loading,
        }

        converted: dict[str, np.ndarray] = {}
        for name, values in float32_features.items():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                feature = values.astype(np.float32)

            if not np.isfinite(feature).all():
                raise InvalidPhysicalState(
                    f"Branch feature {name} cannot be represented in float32."
                )
            if name == "rate_a" and np.any(
                (values > 0.0) & (feature == 0.0)
            ):
                raise InvalidPhysicalState(
                    "Positive RATE_A underflows to zero in feature precision."
                )

            converted[name] = feature

        df["s_from_mva"] = converted["s_from_mva"]
        df["s_to_mva"] = converted["s_to_mva"]
        df["s_max_mva"] = converted["s_max_mva"]
        df["loading_percent"] = converted["loading_percent"]
        return with_branch_rating_features(df)

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
            row_mask = summary["scenario"].astype(int) == int(scenario_id)

            if len(rated):
                max_loading = float(rated["loading_percent"].max())
                mean_loading = float(rated["loading_percent"].mean())
            else:
                max_loading = 0.0
                mean_loading = 0.0

            summary.loc[row_mask, "max_loading_percent"] = max_loading
            summary.loc[row_mask, "mean_loading_percent"] = mean_loading
            summary.loc[row_mask, "num_unrated_active_branches"] = int(
                len(unrated)
            )

        if "num_unrated_active_branches" in summary.columns:
            summary["num_unrated_active_branches"] = (
                summary["num_unrated_active_branches"].astype(int)
            )

        return summary

    def build_state(self, scenario_id: int) -> GridFMState:
        """Build a schema-v2 state with validated contiguous graph indices."""

        bus = self.bus_df[
            self.bus_df["scenario"] == scenario_id
        ].copy()
        branch = self.branch_df[
            self.branch_df["scenario"] == scenario_id
        ].copy()
        gen = self.gen_df[
            self.gen_df["scenario"] == scenario_id
        ].copy()

        if bus.empty:
            raise ValueError(
                f"Scenario {scenario_id} not found in bus_data."
            )
        if branch.empty:
            raise ValueError(
                f"Scenario {scenario_id} not found in branch_data."
            )
        if gen.empty:
            raise ValueError(
                f"Scenario {scenario_id} not found in gen_data."
            )

        topology = validate_state_topology(
            scenario_id=scenario_id,
            bus_df=bus,
            branch_df=branch,
            gen_df=gen,
        )
        bus = with_bus_generator_features(
            topology.bus_df,
            topology.gen_df,
        )
        branch = with_branch_rating_features(
            topology.branch_df,
        )

        bus_features = finite_feature_matrix(
            bus,
            BUS_FEATURE_COLUMNS,
            label="bus",
        )
        branch_features = finite_feature_matrix(
            branch,
            BRANCH_FEATURE_COLUMNS,
            label="branch",
        )

        rated = branch[
            (branch["br_status"] > 0.0)
            & (branch["rate_a"] > 0.0)
        ]
        outaged = branch[branch["br_status"] <= 0.0]

        physical_metrics = calculate_physical_metrics_from_frames(
            bus_df=bus,
            branch_df=branch,
            gen_df=topology.gen_df,
            power_flow_converged=False,
            physics_config=self.physics_config,
        )
        metrics = {
            "num_buses": int(len(bus)),
            "num_branches": int(len(branch)),
            "num_generators": int(len(topology.gen_df)),
            "mean_loading_percent": (
                float(rated["loading_percent"].mean())
                if len(rated)
                else 0.0
            ),
            "min_vm_pu": float(bus["Vm"].min()),
            "max_vm_pu": float(bus["Vm"].max()),
            "num_outaged_branches": int(len(outaged)),
            **physical_metrics,
        }

        return GridFMState(
            scenario_id=int(scenario_id),
            load_scenario_idx=float(bus["load_scenario_idx"].iloc[0]),
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=topology.edge_index,
            branch_ids=topology.branch_ids,
            branch_status=topology.branch_status,
            metrics=metrics,
            outaged_branch_ids=[
                int(value)
                for value in outaged["idx"].to_numpy(dtype=np.int64)
            ],
            bus_ids=topology.bus_ids,
        )
