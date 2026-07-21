from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from grid_topology_ai._data_adapter_core import *  # noqa: F401,F403
from grid_topology_ai._data_adapter_core import (
    GridFMAdapter as _CoreGridFMAdapter,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    ZeroRateAPolicy,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState


# Active branches with RATE_A == 0 have no thermal denominator. Store a
# finite sentinel instead of a false zero-percent loading so feature consumers
# can distinguish "unrated" from "rated and unloaded".
UNRATED_LOADING_PERCENT = -1.0

# Preserve the public module path used by pickled states and type displays.
GridFMState.__module__ = __name__


class GridFMAdapter(_CoreGridFMAdapter):
    """GridFM adapter with explicit semantics for active unrated branches."""

    @staticmethod
    def _add_branch_loading(
        branch_df: pd.DataFrame,
        physics_config: PhysicsConfig | None = None,
    ) -> pd.DataFrame:
        """
        Add validated MVA-flow and loading columns.

        Encoding:
        - active rated branch: actual loading percentage;
        - active unrated branch: ``UNRATED_LOADING_PERCENT``;
        - inactive branch: ``0.0``.
        """

        config = physics_config or DEFAULT_PHYSICS_CONFIG
        df = branch_df

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

        active = status > 0.0
        rated = active & (rate_a > 0.0)
        unlimited = active & (rate_a == 0.0)

        if np.any(active & (rate_a < 0.0)):
            raise InvalidPhysicalState(
                "Active branch RATE_A must be non-negative."
            )

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
        loading[unlimited] = UNRATED_LOADING_PERCENT

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            loading[rated] = (
                s_max[rated] / rate_a[rated] * 100.0
            )

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

        return df

    def build_summary(self) -> pd.DataFrame:
        """Build scenario summaries using only active rated loading values."""

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
        """Build a state whose loading aggregate excludes unrated branches."""

        state = super().build_state(scenario_id)
        branch = self.branch_df[
            self.branch_df["scenario"] == scenario_id
        ]
        rated = branch[
            (branch["br_status"] > 0.0) & (branch["rate_a"] > 0.0)
        ]
        mean_loading = (
            float(rated["loading_percent"].mean())
            if len(rated)
            else 0.0
        )
        metrics = dict(state.metrics)
        metrics["mean_loading_percent"] = mean_loading
        return replace(state, metrics=metrics)
