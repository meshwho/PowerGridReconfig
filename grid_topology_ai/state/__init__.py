from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.power_flow.errors import InvalidPhysicalState
from pypower.idx_brch import BR_STATUS, PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig, ZeroRateAPolicy
from grid_topology_ai.physics.constraints import (
    calculate_physical_metrics_from_frames,
    calculate_physical_metrics_from_result,
)
from .topology import validate_state_topology


BUS_FEATURE_COLUMNS = [
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
    "min_vm_pu",
    "max_vm_pu",
    "gen_online_count",
    "gen_available",
    "gen_p_min_mw",
    "gen_p_max_mw",
    "gen_q_min_mvar",
    "gen_q_max_mvar",
    "gen_p_down_margin_mw",
    "gen_p_up_margin_mw",
    "gen_q_down_margin_mvar",
    "gen_q_up_margin_mvar",
    "gen_min_p_down_margin_mw",
    "gen_min_p_up_margin_mw",
    "gen_min_q_down_margin_mvar",
    "gen_min_q_up_margin_mvar",
    "gen_p_limit_violation_count",
    "gen_q_limit_violation_count",
]

BRANCH_FEATURE_COLUMNS = [
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
    "unlimited_rating",
]

_GENERATOR_FEATURE_COLUMNS = [
    "gen_online_count",
    "gen_available",
    "gen_p_min_mw",
    "gen_p_max_mw",
    "gen_q_min_mvar",
    "gen_q_max_mvar",
    "gen_p_down_margin_mw",
    "gen_p_up_margin_mw",
    "gen_q_down_margin_mvar",
    "gen_q_up_margin_mvar",
    "gen_min_p_down_margin_mw",
    "gen_min_p_up_margin_mw",
    "gen_min_q_down_margin_mvar",
    "gen_min_q_up_margin_mvar",
    "gen_p_limit_violation_count",
    "gen_q_limit_violation_count",
]


def with_bus_generator_features(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return bus rows with aggregate and worst individual generator margins."""

    required_bus_columns = {"bus", "min_vm_pu", "max_vm_pu"}
    required_gen_columns = {
        "bus",
        "p_mw",
        "q_mvar",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    }

    missing_bus = required_bus_columns - set(bus_df.columns)
    missing_gen = required_gen_columns - set(gen_df.columns)

    if missing_bus:
        raise InvalidPhysicalState(
            f"Bus data is missing schema columns: {sorted(missing_bus)}."
        )
    if missing_gen:
        raise InvalidPhysicalState(
            f"Generator data is missing schema columns: {sorted(missing_gen)}."
        )

    result = bus_df.copy()
    result["Pg"] = 0.0
    result["Qg"] = 0.0

    for column in _GENERATOR_FEATURE_COLUMNS:
        result[column] = 0.0

    status = gen_df["in_service"].to_numpy(dtype=np.float64)
    if not np.isfinite(status).all():
        raise InvalidPhysicalState(
            "Generator in_service contains NaN or infinity."
        )
    if not np.isin(status, (0.0, 1.0)).all():
        raise InvalidPhysicalState(
            "Generator in_service must contain only 0 or 1."
        )

    active = gen_df.loc[status > 0.0].copy()
    if active.empty:
        return result

    numeric_columns = [
        "p_mw",
        "q_mvar",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
    ]
    numeric = active[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise InvalidPhysicalState(
            "Active generator limits and outputs must be finite."
        )

    active["p_down_margin_mw"] = active["p_mw"] - active["min_p_mw"]
    active["p_up_margin_mw"] = active["max_p_mw"] - active["p_mw"]
    active["q_down_margin_mvar"] = (
        active["q_mvar"] - active["min_q_mvar"]
    )
    active["q_up_margin_mvar"] = (
        active["max_q_mvar"] - active["q_mvar"]
    )
    active["p_limit_violated"] = (
        (active["p_down_margin_mw"] < 0.0)
        | (active["p_up_margin_mw"] < 0.0)
    ).astype(np.float64)
    active["q_limit_violated"] = (
        (active["q_down_margin_mvar"] < 0.0)
        | (active["q_up_margin_mvar"] < 0.0)
    ).astype(np.float64)

    grouped = active.groupby("bus", sort=False).agg(
        Pg=("p_mw", "sum"),
        Qg=("q_mvar", "sum"),
        gen_online_count=("bus", "size"),
        gen_p_min_mw=("min_p_mw", "sum"),
        gen_p_max_mw=("max_p_mw", "sum"),
        gen_q_min_mvar=("min_q_mvar", "sum"),
        gen_q_max_mvar=("max_q_mvar", "sum"),
        gen_min_p_down_margin_mw=("p_down_margin_mw", "min"),
        gen_min_p_up_margin_mw=("p_up_margin_mw", "min"),
        gen_min_q_down_margin_mvar=("q_down_margin_mvar", "min"),
        gen_min_q_up_margin_mvar=("q_up_margin_mvar", "min"),
        gen_p_limit_violation_count=("p_limit_violated", "sum"),
        gen_q_limit_violation_count=("q_limit_violated", "sum"),
    )

    mapped_columns = [
        "Pg",
        "Qg",
        "gen_online_count",
        "gen_p_min_mw",
        "gen_p_max_mw",
        "gen_q_min_mvar",
        "gen_q_max_mvar",
        "gen_min_p_down_margin_mw",
        "gen_min_p_up_margin_mw",
        "gen_min_q_down_margin_mvar",
        "gen_min_q_up_margin_mvar",
        "gen_p_limit_violation_count",
        "gen_q_limit_violation_count",
    ]
    bus_ids = result["bus"]

    for column in mapped_columns:
        result[column] = bus_ids.map(grouped[column]).fillna(0.0)

    result["gen_available"] = (
        result["gen_online_count"] > 0.0
    ).astype(np.float64)
    result["gen_p_down_margin_mw"] = (
        result["Pg"] - result["gen_p_min_mw"]
    )
    result["gen_p_up_margin_mw"] = (
        result["gen_p_max_mw"] - result["Pg"]
    )
    result["gen_q_down_margin_mvar"] = (
        result["Qg"] - result["gen_q_min_mvar"]
    )
    result["gen_q_up_margin_mvar"] = (
        result["gen_q_max_mvar"] - result["Qg"]
    )

    return result


def with_branch_rating_features(
    branch_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add the explicit unlimited-rating flag used by the current schema."""

    if "rate_a" not in branch_df.columns:
        raise InvalidPhysicalState(
            "Branch data is missing required column: rate_a."
        )

    result = branch_df.copy()
    rate_a = result["rate_a"].to_numpy(dtype=np.float64)

    if not np.isfinite(rate_a).all():
        raise InvalidPhysicalState(
            "Branch rate_a contains NaN or infinity."
        )
    if np.any(rate_a < 0.0):
        raise InvalidPhysicalState(
            "Branch rate_a must be non-negative."
        )

    result["unlimited_rating"] = (rate_a == 0.0).astype(np.float32)
    return result


def finite_feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise InvalidPhysicalState(
            f"{label.capitalize()} data is missing feature columns: "
            f"{sorted(missing)}."
        )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        features = frame[list(columns)].to_numpy(dtype=np.float32)

    if not np.isfinite(features).all():
        raise InvalidPhysicalState(
            f"{label.capitalize()} features cannot be represented in float32."
        )

    return features

MetricsCalculator = Callable[..., Mapping[str, object]]


@dataclass(frozen=True)
class GridFMState:
    """A graph state with original bus IDs beside contiguous graph indices."""

    scenario_id: int
    load_scenario_idx: float
    bus_features: np.ndarray
    branch_features: np.ndarray
    edge_index: np.ndarray
    branch_ids: np.ndarray
    branch_status: np.ndarray
    metrics: dict[str, Any]
    outaged_branch_ids: list[int]
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

# State artifact IO remains separate from in-memory construction, but its small
# public surface is owned by this canonical package.
from grid_topology_ai.state.io import (  # noqa: E402
    GridFMStateStore,
    validate_state_npz_schema_arrays,
)
