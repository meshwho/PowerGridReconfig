from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    ZeroRateAPolicy,
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


_RAW_BUS_FEATURE_COLUMNS = [
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "GS",
    "BS",
]

_RAW_BRANCH_FEATURE_COLUMNS = [
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "tap",
    "shift",
    "rate_a",
    "br_status",
    "s_from_mva",
    "s_to_mva",
    "s_max_mva",
    "loading_percent",
]


# Preserve the public path used by pickled states and type displays.
GridFMState.__module__ = __name__


def compute_voltage_violation_metrics(
    bus_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Compute voltage-violation counts and magnitudes."""

    vm = bus_df["Vm"].to_numpy(dtype=float)
    vmin = bus_df["min_vm_pu"].to_numpy(dtype=float)
    vmax = bus_df["max_vm_pu"].to_numpy(dtype=float)

    low_voltage_violation = np.maximum(vmin - vm, 0.0)
    high_voltage_violation = np.maximum(vm - vmax, 0.0)

    total_low_voltage_violation = float(np.sum(low_voltage_violation))
    total_high_voltage_violation = float(np.sum(high_voltage_violation))

    return {
        "num_low_voltage_buses": int(np.sum(low_voltage_violation > 0.0)),
        "num_high_voltage_buses": int(np.sum(high_voltage_violation > 0.0)),
        "total_low_voltage_violation": total_low_voltage_violation,
        "total_high_voltage_violation": total_high_voltage_violation,
        "total_voltage_violation": (
            total_low_voltage_violation + total_high_voltage_violation
        ),
    }


class GridFMAdapter:
    """Adapter from GridFM parquet output to versioned grid states."""

    def __init__(
        self,
        raw_dir: str | Path,
        scenario_ids: Sequence[int] | None = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG

        if scenario_ids is None:
            self._scenario_filter: tuple[int, ...] | None = None
        else:
            normalized_ids = tuple(
                sorted({int(value) for value in scenario_ids})
            )
            if not normalized_ids:
                raise ValueError(
                    "scenario_ids was provided, but it is empty."
                )
            self._scenario_filter = normalized_ids

        self.bus_df = self._read_required_parquet("bus_data.parquet")
        self.branch_df = self._read_required_parquet("branch_data.parquet")
        self.gen_df = self._read_required_parquet("gen_data.parquet")
        self.branch_df = self._add_branch_loading(
            self.branch_df,
            physics_config=self.physics_config,
        )
        self._validate_required_columns()

    def _read_required_parquet(
        self,
        file_name: str,
    ) -> pd.DataFrame:
        path = self.raw_dir / file_name

        if not path.exists():
            raise FileNotFoundError(
                f"Required GridFM file not found: {path}"
            )

        if self._scenario_filter is None:
            frame = pd.read_parquet(path)
        else:
            scenario_ids = list(self._scenario_filter)
            try:
                frame = pd.read_parquet(
                    path,
                    filters=[("scenario", "in", scenario_ids)],
                )
            except (TypeError, ValueError, NotImplementedError):
                frame = pd.read_parquet(path)
                if "scenario" not in frame.columns:
                    raise ValueError(
                        f"Parquet file has no scenario column: {path}"
                    )
                frame = frame.loc[
                    frame["scenario"].astype(int).isin(scenario_ids)
                ]

        if frame.empty:
            raise ValueError(
                f"No rows were loaded from {path}. "
                f"Scenario filter: {self._scenario_filter}"
            )

        return frame

    def _validate_required_columns(self) -> None:
        required_bus = {
            "scenario",
            "load_scenario_idx",
            "bus",
            *_RAW_BUS_FEATURE_COLUMNS,
        }
        required_branch = {
            "scenario",
            "load_scenario_idx",
            "idx",
            "from_bus",
            "to_bus",
            "br_status",
            *_RAW_BRANCH_FEATURE_COLUMNS,
        }
        required_gen = {
            "scenario",
            "idx",
            "bus",
            "p_mw",
            "q_mvar",
            "min_p_mw",
            "max_p_mw",
            "min_q_mvar",
            "max_q_mvar",
            "in_service",
        }

        missing_bus = required_bus - set(self.bus_df.columns)
        missing_branch = required_branch - set(self.branch_df.columns)
        missing_gen = required_gen - set(self.gen_df.columns)

        try:
            if missing_bus:
                raise ValueError(
                    f"Missing bus columns: {sorted(missing_bus)}"
                )
            if missing_branch:
                raise ValueError(
                    f"Missing branch columns: {sorted(missing_branch)}"
                )
            if missing_gen:
                raise ValueError(
                    f"Missing generator columns: {sorted(missing_gen)}"
                )
        except ValueError as exc:
            raise InvalidPhysicalState(str(exc)) from exc

        missing_bus = {"min_vm_pu", "max_vm_pu"} - set(
            self.bus_df.columns
        )
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

    def scenario_ids(self) -> list[int]:
        return sorted(
            int(value)
            for value in self.bus_df["scenario"].unique()
        )

    def build_summary(self) -> pd.DataFrame:
        """Build summaries using only active thermally rated branches."""

        overload_threshold = (
            self.physics_config.overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        hard_overload_threshold = (
            self.physics_config.hard_overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        rows = []

        for scenario_id in self.scenario_ids():
            bus = self.bus_df[
                self.bus_df["scenario"] == scenario_id
            ]
            branch = self.branch_df[
                self.branch_df["scenario"] == scenario_id
            ]
            gen = self.gen_df[
                self.gen_df["scenario"] == scenario_id
            ]

            in_service = branch[branch["br_status"] > 0]
            outaged = branch[branch["br_status"] <= 0]
            overloaded = in_service[
                in_service["loading_percent"] > overload_threshold
            ]
            hard_overloaded = in_service[
                in_service["loading_percent"] > hard_overload_threshold
            ]
            voltage_metrics = compute_voltage_violation_metrics(bus)

            rows.append(
                {
                    "scenario": scenario_id,
                    "load_scenario_idx": float(
                        bus["load_scenario_idx"].iloc[0]
                    ),
                    "num_buses": int(len(bus)),
                    "num_branches": int(len(branch)),
                    "num_generators": int(len(gen)),
                    "total_load_p_mw": float(bus["Pd"].sum()),
                    "total_load_q_mvar": float(bus["Qd"].sum()),
                    "total_gen_p_mw": float(
                        gen[gen["in_service"] > 0]["p_mw"].sum()
                    ),
                    "max_loading_percent": float(
                        in_service["loading_percent"].max()
                    ),
                    "mean_loading_percent": float(
                        in_service["loading_percent"].mean()
                    ),
                    "num_overloaded_branches": int(len(overloaded)),
                    "num_hard_overloaded_branches": int(
                        len(hard_overloaded)
                    ),
                    "min_vm_pu": float(bus["Vm"].min()),
                    "max_vm_pu": float(bus["Vm"].max()),
                    **voltage_metrics,
                    "num_outaged_branches": int(len(outaged)),
                    "outaged_branch_ids": list(
                        outaged["idx"].astype(int).values
                    ),
                }
            )

        summary = pd.DataFrame(rows)

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

    def useful_scenario_ids(
        self,
        min_loading_percent: float = 100.0,
        max_loading_percent: float = 250.0,
        require_outage: bool = True,
    ) -> list[int]:
        summary = self.build_summary()
        mask = (
            (summary["num_overloaded_branches"] > 0)
            & (
                summary["max_loading_percent"]
                >= min_loading_percent
            )
            & (
                summary["max_loading_percent"]
                <= max_loading_percent
            )
        )
        if require_outage:
            mask = mask & (summary["num_outaged_branches"] > 0)
        return [
            int(value)
            for value in summary.loc[mask, "scenario"].values
        ]

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
