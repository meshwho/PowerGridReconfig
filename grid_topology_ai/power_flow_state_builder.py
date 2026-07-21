from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState


@dataclass(frozen=True, slots=True)
class PowerFlowStateBuilder:
    """Build every solved GridFM state through one canonical representation path."""

    physics_config: PhysicsConfig

    def __post_init__(self) -> None:
        if not isinstance(self.physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")

    def build(
        self,
        *,
        scenario_id: int,
        result_ppc: Mapping[str, Any],
        original_frames: Mapping[str, pd.DataFrame],
        physical_metrics: Mapping[str, object] | None = None,
    ) -> GridFMState:
        bus_df, branch_df, gen_df = self._copy_frames(original_frames)
        bus_res, branch_res, gen_res = self._result_arrays(result_ppc)

        self._require_row_count("bus", bus_df, bus_res)
        self._require_row_count("branch", branch_df, branch_res)
        self._require_row_count("gen", gen_df, gen_res)

        bus_df["Vm"] = bus_res[:, VM]
        bus_df["Va"] = bus_res[:, VA]

        gen_df["p_mw"] = gen_res[:, PG]
        gen_df["q_mvar"] = gen_res[:, QG]
        gen_df["in_service"] = gen_res[:, GEN_STATUS]
        self._update_bus_generation(bus_df, gen_df)

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

        metrics = dict(
            physical_metrics
            if physical_metrics is not None
            else calculate_physical_metrics_from_result(
                dict(result_ppc),
                power_flow_converged=True,
                physics_config=self.physics_config,
            )
        )
        return self._to_state(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            physical_metrics=metrics,
        )

    @staticmethod
    def _copy_frames(
        original_frames: Mapping[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            return (
                original_frames["bus"].copy(),
                original_frames["branch"].copy(),
                original_frames["gen"].copy(),
            )
        except KeyError as exc:
            raise InvalidPhysicalState(
                f"Missing power-flow source frame: {exc.args[0]}."
            ) from exc

    @staticmethod
    def _result_arrays(
        result_ppc: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            return (
                np.asarray(result_ppc["bus"]),
                np.asarray(result_ppc["branch"]),
                np.asarray(result_ppc["gen"]),
            )
        except KeyError as exc:
            raise InvalidPhysicalState(
                f"Missing PYPOWER result matrix: {exc.args[0]}."
            ) from exc

    @staticmethod
    def _require_row_count(
        name: str,
        frame: pd.DataFrame,
        matrix: np.ndarray,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != len(frame):
            raise InvalidPhysicalState(
                f"PYPOWER {name} result does not match the source frame."
            )

    @staticmethod
    def _update_bus_generation(
        bus_df: pd.DataFrame,
        gen_df: pd.DataFrame,
    ) -> None:
        bus_df["Pg"] = 0.0
        bus_df["Qg"] = 0.0
        generation = gen_df.groupby("bus", sort=False)[["p_mw", "q_mvar"]].sum()

        for bus_id, values in generation.iterrows():
            mask = bus_df["bus"].astype(int) == int(bus_id)
            bus_df.loc[mask, "Pg"] = float(values["p_mw"])
            bus_df.loc[mask, "Qg"] = float(values["q_mvar"])

    @staticmethod
    def _to_state(
        *,
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        physical_metrics: Mapping[str, object],
    ) -> GridFMState:
        bus_df = bus_df.sort_values("bus").reset_index(drop=True)
        branch_df = branch_df.sort_values("idx").reset_index(drop=True)

        bus_features = PowerFlowStateBuilder._finite_float32_features(
            bus_df,
            BUS_FEATURE_COLUMNS,
            label="bus",
        )
        branch_features = PowerFlowStateBuilder._finite_float32_features(
            branch_df,
            BRANCH_FEATURE_COLUMNS,
            label="branch",
        )

        branch_status = branch_df["br_status"].to_numpy(dtype=np.float32)
        rated = branch_df[
            (branch_df["br_status"] > 0.0)
            & (branch_df["rate_a"] > 0.0)
        ]
        outaged = branch_df[branch_df["br_status"] <= 0.0]

        metrics = {
            "num_buses": int(len(bus_df)),
            "num_branches": int(len(branch_df)),
            "mean_loading_percent": (
                float(rated["loading_percent"].mean())
                if len(rated)
                else 0.0
            ),
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
            edge_index=branch_df[["from_bus", "to_bus"]]
            .to_numpy(dtype=np.int64)
            .T,
            branch_ids=branch_df["idx"].to_numpy(dtype=np.int64),
            branch_status=branch_status,
            metrics=metrics,
            outaged_branch_ids=[int(value) for value in outaged["idx"]],
        )

    @staticmethod
    def _finite_float32_features(
        frame: pd.DataFrame,
        columns: list[str],
        *,
        label: str,
    ) -> np.ndarray:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            features = frame[columns].to_numpy(dtype=np.float32)

        if not np.isfinite(features).all():
            raise InvalidPhysicalState(
                f"{label.capitalize()} features cannot be represented in float32."
            )
        return features
