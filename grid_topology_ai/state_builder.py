from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai._data_adapter_core import GridFMState as _CoreGridFMState
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    ZeroRateAPolicy,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_frames,
    calculate_physical_metrics_from_result,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.state_schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    finite_feature_matrix,
    with_branch_rating_features,
    with_bus_generator_features,
)
from grid_topology_ai.state_topology import validate_state_topology


MetricsCalculator = Callable[..., Mapping[str, object]]


@dataclass(frozen=True)
class GridFMState(_CoreGridFMState):
    """A graph state with original bus IDs beside contiguous graph indices."""

    bus_ids: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class GridFMStateBuilder:
    """Build initial and solved states through one representation path."""

    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG
    frame_metrics_calculator: MetricsCalculator = (
        calculate_physical_metrics_from_frames
    )
    result_metrics_calculator: MetricsCalculator = (
        calculate_physical_metrics_from_result
    )

    def __post_init__(self) -> None:
        if not isinstance(self.physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")

    def build_from_frames(
        self,
        *,
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        gen_df: pd.DataFrame,
        power_flow_converged: bool,
        physical_metrics: Mapping[str, object] | None = None,
    ) -> GridFMState:
        """Build a state from GridFM-style bus, branch, and generator frames."""

        self._require_non_empty_frames(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )
        topology = validate_state_topology(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        )
        bus = with_bus_generator_features(
            topology.bus_df,
            topology.gen_df,
        )
        branch = self.add_branch_loading(topology.branch_df)

        metrics = dict(
            physical_metrics
            if physical_metrics is not None
            else self.frame_metrics_calculator(
                bus_df=bus,
                branch_df=branch,
                gen_df=topology.gen_df,
                power_flow_converged=bool(power_flow_converged),
                physics_config=self.physics_config,
            )
        )
        return self._assemble_state(
            scenario_id=scenario_id,
            bus_df=bus,
            branch_df=branch,
            bus_ids=topology.bus_ids,
            edge_index=topology.edge_index,
            branch_ids=topology.branch_ids,
            branch_status=topology.branch_status,
            num_generators=len(topology.gen_df),
            physical_metrics=metrics,
        )

    def build_from_pypower_result(
        self,
        *,
        scenario_id: int,
        result_ppc: Mapping[str, Any],
        original_frames: Mapping[str, pd.DataFrame],
        physical_metrics: Mapping[str, object] | None = None,
    ) -> GridFMState:
        """Update source frames from a PYPOWER result and build the state."""

        bus_df, branch_df, gen_df = self._copy_frames(original_frames)
        bus_result, branch_result, gen_result = self._result_arrays(result_ppc)

        self._require_row_count("bus", bus_df, bus_result)
        self._require_row_count("branch", branch_df, branch_result)
        self._require_row_count("gen", gen_df, gen_result)

        bus_df["Vm"] = bus_result[:, VM]
        bus_df["Va"] = bus_result[:, VA]

        gen_df["p_mw"] = gen_result[:, PG]
        gen_df["q_mvar"] = gen_result[:, QG]
        gen_df["in_service"] = gen_result[:, GEN_STATUS]

        branch_df["br_status"] = branch_result[:, BR_STATUS]
        branch_df["rate_a"] = branch_result[:, RATE_A]
        branch_df["pf"] = branch_result[:, PF]
        branch_df["qf"] = branch_result[:, QF]
        branch_df["pt"] = branch_result[:, PT]
        branch_df["qt"] = branch_result[:, QT]

        metrics = dict(
            physical_metrics
            if physical_metrics is not None
            else self.result_metrics_calculator(
                dict(result_ppc),
                power_flow_converged=True,
                physics_config=self.physics_config,
            )
        )
        return self.build_from_frames(
            scenario_id=scenario_id,
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
            power_flow_converged=True,
            physical_metrics=metrics,
        )

    def add_branch_loading(
        self,
        branch_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add validated apparent-power, loading, and rating features."""

        frame = branch_df.copy()
        try:
            pf = frame["pf"].to_numpy(dtype=np.float64)
            qf = frame["qf"].to_numpy(dtype=np.float64)
            pt = frame["pt"].to_numpy(dtype=np.float64)
            qt = frame["qt"].to_numpy(dtype=np.float64)
            rate_a = frame["rate_a"].to_numpy(dtype=np.float64)
            status = frame["br_status"].to_numpy(dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPhysicalState(
                "Branch flow data must contain numeric "
                "pf/qf/pt/qt/rate_a/br_status columns."
            ) from exc

        required_values = {
            "pf": pf,
            "qf": qf,
            "pt": pt,
            "qt": qt,
            "rate_a": rate_a,
            "br_status": status,
        }
        for name, values in required_values.items():
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
            self.physics_config.zero_rate_a_policy
            is ZeroRateAPolicy.ERROR
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

        feature_values = {
            **required_values,
            "s_from_mva": s_from,
            "s_to_mva": s_to,
            "s_max_mva": s_max,
            "loading_percent": loading,
        }
        converted: dict[str, np.ndarray] = {}
        for name, values in feature_values.items():
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

        frame["s_from_mva"] = converted["s_from_mva"]
        frame["s_to_mva"] = converted["s_to_mva"]
        frame["s_max_mva"] = converted["s_max_mva"]
        frame["loading_percent"] = converted["loading_percent"]
        return with_branch_rating_features(frame)

    @staticmethod
    def _require_non_empty_frames(
        *,
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        gen_df: pd.DataFrame,
    ) -> None:
        frames = {
            "bus_data": bus_df,
            "branch_data": branch_df,
            "gen_data": gen_df,
        }
        for name, frame in frames.items():
            if frame.empty:
                raise ValueError(
                    f"Scenario {scenario_id} not found in {name}."
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
    def _assemble_state(
        *,
        scenario_id: int,
        bus_df: pd.DataFrame,
        branch_df: pd.DataFrame,
        bus_ids: np.ndarray,
        edge_index: np.ndarray,
        branch_ids: np.ndarray,
        branch_status: np.ndarray,
        num_generators: int,
        physical_metrics: Mapping[str, object],
    ) -> GridFMState:
        bus_features = finite_feature_matrix(
            bus_df,
            BUS_FEATURE_COLUMNS,
            label="bus",
        )
        branch_features = finite_feature_matrix(
            branch_df,
            BRANCH_FEATURE_COLUMNS,
            label="branch",
        )

        rated = branch_df[
            (branch_df["br_status"] > 0.0)
            & (branch_df["rate_a"] > 0.0)
        ]
        outaged = branch_df[branch_df["br_status"] <= 0.0]
        metrics = {
            "num_buses": int(len(bus_df)),
            "num_branches": int(len(branch_df)),
            "num_generators": int(num_generators),
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
            load_scenario_idx=float(
                bus_df["load_scenario_idx"].iloc[0]
            ),
            bus_features=bus_features,
            branch_features=branch_features,
            edge_index=edge_index,
            branch_ids=branch_ids,
            branch_status=branch_status,
            metrics=metrics,
            outaged_branch_ids=[
                int(value)
                for value in outaged["idx"].to_numpy(dtype=np.int64)
            ],
            bus_ids=bus_ids,
        )
