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
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.physical_objective import (
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_frames,
)


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
]


def compute_voltage_violation_metrics(bus_df: pd.DataFrame) -> dict[str, float | int]:
    """
    Compute voltage violation metrics.

    We use violation magnitude, not only the number of violated buses.

    Example:
        Vm = 1.0601, max = 1.0600 -> tiny violation
        Vm = 1.1000, max = 1.0600 -> large violation

    This is important for reward stability.
    """

    vm = bus_df["Vm"].to_numpy(dtype=float)
    vmin = bus_df["min_vm_pu"].to_numpy(dtype=float)
    vmax = bus_df["max_vm_pu"].to_numpy(dtype=float)

    low_voltage_violation = np.maximum(vmin - vm, 0.0)
    high_voltage_violation = np.maximum(vm - vmax, 0.0)

    total_low_voltage_violation = float(np.sum(low_voltage_violation))
    total_high_voltage_violation = float(np.sum(high_voltage_violation))
    total_voltage_violation = (
        total_low_voltage_violation + total_high_voltage_violation
    )

    num_low_voltage_buses = int(np.sum(low_voltage_violation > 0.0))
    num_high_voltage_buses = int(np.sum(high_voltage_violation > 0.0))

    return {
        "num_low_voltage_buses": num_low_voltage_buses,
        "num_high_voltage_buses": num_high_voltage_buses,
        "total_low_voltage_violation": total_low_voltage_violation,
        "total_high_voltage_violation": total_high_voltage_violation,
        "total_voltage_violation": total_voltage_violation,
    }


@dataclass(frozen=True)
class GridFMState:
    """
    One power-grid state loaded from gridfm-datakit output.

    This is not yet a transition.

    It represents one emergency grid state:
        buses + branches + topology + metrics.

    Later we will use it as:
        state -> action -> next_state -> reward
    """

    scenario_id: int
    load_scenario_idx: float

    # Node features for GNN.
    # Shape: [num_buses, num_bus_features]
    bus_features: np.ndarray

    # Branch / edge features for GNN.
    # Shape: [num_branches, num_branch_features]
    branch_features: np.ndarray

    # Edge index in PyTorch Geometric style.
    # Shape: [2, num_branches]
    edge_index: np.ndarray

    # Original branch IDs from gridfm-datakit.
    # Shape: [num_branches]
    branch_ids: np.ndarray

    # Branch status:
    # 1 = in service
    # 0 = out of service
    branch_status: np.ndarray

    # Useful scalar information.
    metrics: dict[str, Any]

    # Which branches are outaged in this scenario.
    outaged_branch_ids: list[int]


class GridFMAdapter:
    """
    Adapter for gridfm-datakit parquet output.

    Responsibilities:
    1. Load bus_data.parquet, branch_data.parquet, gen_data.parquet.
    2. Compute branch loading from pf/qf/pt/qt/rate_a.
    3. Build GNN-ready state objects.
    4. Filter useful emergency scenarios.

    Important:
    gridfm-datakit generates scenarios, but it does not generate our RL transitions.
    This adapter is the bridge between GridFM data and our AlphaZero/RL pipeline.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        scenario_ids: Sequence[int] | None = None,
    ):
        self.raw_dir = Path(raw_dir)

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

        self.bus_df = self._read_required_parquet(
            "bus_data.parquet"
        )
        self.branch_df = self._read_required_parquet(
            "branch_data.parquet"
        )
        self.gen_df = self._read_required_parquet(
            "gen_data.parquet"
        )

        self.branch_df = self._add_branch_loading(
            self.branch_df
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
                    filters=[
                        (
                            "scenario",
                            "in",
                            scenario_ids,
                        )
                    ],
                )
            except (
                TypeError,
                ValueError,
                NotImplementedError,
            ):
                # Fallback for parquet engines/files that do not
                # support the "in" predicate efficiently.
                frame = pd.read_parquet(path)

                if "scenario" not in frame.columns:
                    raise ValueError(
                        f"Parquet file has no scenario column: "
                        f"{path}"
                    )

                frame = frame.loc[
                    frame["scenario"].astype(int).isin(
                        scenario_ids
                    )
                ]

        if frame.empty:
            raise ValueError(
                f"No rows were loaded from {path}. "
                f"Scenario filter: {self._scenario_filter}"
            )

        return frame.reset_index(drop=True)

    def _validate_required_columns(self) -> None:
        required_bus = {"scenario", "load_scenario_idx", "bus", *BUS_FEATURE_COLUMNS}

        required_branch = {
            "scenario",
            "load_scenario_idx",
            "idx",
            "from_bus",
            "to_bus",
            "br_status",
            *BRANCH_FEATURE_COLUMNS,
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

        if missing_bus:
            raise ValueError(f"Missing bus columns: {sorted(missing_bus)}")

        if missing_branch:
            raise ValueError(f"Missing branch columns: {sorted(missing_branch)}")

        if missing_gen:
            raise ValueError(f"Missing generator columns: {sorted(missing_gen)}")

    @staticmethod
    def _add_branch_loading(
        branch_df: pd.DataFrame,
        physics_config: PhysicsConfig | None = None,
    ) -> pd.DataFrame:
        """
        Add strictly validated MVA flow and loading columns.

        Non-finite and non-representable mandatory physical values are rejected.
        They must never be converted to zero.

        RATE_A == 0 follows PhysicsConfig.zero_rate_a_policy.
        Inactive branches have loading_percent == 0.
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

    def scenario_ids(self) -> list[int]:
        """
        Return all scenario IDs available in the dataset.
        """

        return sorted(int(x) for x in self.bus_df["scenario"].unique())

    def build_summary(self) -> pd.DataFrame:
        """
        Build one summary row per scenario.

        This is useful for:
        - filtering emergency states;
        - debugging;
        - choosing training scenarios.
        """

        rows = []

        for scenario_id in self.scenario_ids():
            bus = self.bus_df[self.bus_df["scenario"] == scenario_id]
            branch = self.branch_df[self.branch_df["scenario"] == scenario_id]
            gen = self.gen_df[self.gen_df["scenario"] == scenario_id]

            in_service = branch[branch["br_status"] > 0]
            outaged = branch[branch["br_status"] <= 0]

            overloaded = in_service[in_service["loading_percent"] > OVERLOAD_LIMIT_PERCENT]
            hard_overloaded = in_service[in_service["loading_percent"] > HARD_OVERLOAD_LIMIT_PERCENT]

            voltage_metrics = compute_voltage_violation_metrics(bus)

            low_voltage_violation = np.maximum(
                bus["min_vm_pu"].to_numpy(dtype=float) - bus["Vm"].to_numpy(dtype=float),
                0.0,
            )

            high_voltage_violation = np.maximum(
                bus["Vm"].to_numpy(dtype=float) - bus["max_vm_pu"].to_numpy(dtype=float),
                0.0,
            )

            total_low_voltage_violation = float(np.sum(low_voltage_violation))
            total_high_voltage_violation = float(np.sum(high_voltage_violation))
            total_voltage_violation = (
                    total_low_voltage_violation + total_high_voltage_violation
            )

            rows.append(
                {
                    "scenario": scenario_id,
                    "load_scenario_idx": float(bus["load_scenario_idx"].iloc[0]),
                    "num_buses": int(len(bus)),
                    "num_branches": int(len(branch)),
                    "num_generators": int(len(gen)),
                    "total_load_p_mw": float(bus["Pd"].sum()),
                    "total_load_q_mvar": float(bus["Qd"].sum()),
                    "total_gen_p_mw": float(gen[gen["in_service"] > 0]["p_mw"].sum()),
                    "max_loading_percent": float(in_service["loading_percent"].max()),
                    "mean_loading_percent": float(in_service["loading_percent"].mean()),
                    "num_overloaded_branches": int(len(overloaded)),
                    "num_hard_overloaded_branches": int(len(hard_overloaded)),
                    "min_vm_pu": float(bus["Vm"].min()),
                    "max_vm_pu": float(bus["Vm"].max()),
                    **voltage_metrics,
                    "num_outaged_branches": int(len(outaged)),
                    "outaged_branch_ids": list(outaged["idx"].astype(int).values),
                }
            )

        return pd.DataFrame(rows)

    def useful_scenario_ids(
        self,
        min_loading_percent: float = 100.0,
        max_loading_percent: float = 250.0,
        require_outage: bool = True,
    ) -> list[int]:
        """
        Select useful emergency scenarios for the first MVP.

        A useful scenario:
        - has at least one overloaded branch;
        - has max loading in a reasonable range;
        - optionally has at least one branch outage.
        """

        summary = self.build_summary()

        mask = (
            (summary["num_overloaded_branches"] > 0)
            & (summary["max_loading_percent"] >= min_loading_percent)
            & (summary["max_loading_percent"] <= max_loading_percent)
        )

        if require_outage:
            mask = mask & (summary["num_outaged_branches"] > 0)

        return [int(x) for x in summary.loc[mask, "scenario"].values]

    def build_state(self, scenario_id: int) -> GridFMState:
        """
        Build one GNN-ready state for a given scenario.
        """

        bus = self.bus_df[self.bus_df["scenario"] == scenario_id].copy()
        branch = self.branch_df[self.branch_df["scenario"] == scenario_id].copy()
        gen = self.gen_df[self.gen_df["scenario"] == scenario_id].copy()

        if bus.empty:
            raise ValueError(f"Scenario {scenario_id} not found in bus_data.")

        if branch.empty:
            raise ValueError(f"Scenario {scenario_id} not found in branch_data.")

        if gen.empty:
            raise ValueError(f"Scenario {scenario_id} not found in gen_data.")

        bus = bus.sort_values("bus")
        branch = branch.sort_values("idx")
        gen = gen.sort_values("idx")

        bus_features = bus[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        branch_features = branch[BRANCH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        edge_index = branch[["from_bus", "to_bus"]].to_numpy(dtype=np.int64).T

        branch_ids = branch["idx"].to_numpy(dtype=np.int64)
        branch_status = branch["br_status"].to_numpy(dtype=np.float32)

        in_service = branch[branch["br_status"] > 0]
        outaged = branch[branch["br_status"] <= 0]

        physical_metrics = calculate_physical_metrics_from_frames(
            bus_df=bus,
            branch_df=branch,
            gen_df=gen,
            # GridFM parquet rows do not carry reliable AC-PF convergence
            # provenance. The environment obtains it with an explicit no-op PF.
            power_flow_converged=False,
        )

        metrics = {
            "num_buses": int(len(bus)),
            "num_branches": int(len(branch)),
            "num_generators": int(len(gen)),
            "mean_loading_percent": (
                float(in_service["loading_percent"].mean())
                if len(in_service)
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
            edge_index=edge_index,
            branch_ids=branch_ids,
            branch_status=branch_status,
            metrics=metrics,
            outaged_branch_ids=[int(x) for x in outaged["idx"].values],
        )
