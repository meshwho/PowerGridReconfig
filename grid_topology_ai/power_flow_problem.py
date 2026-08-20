from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_bus import PQ as BUS_TYPE_PQ
from pypower.idx_bus import PV as BUS_TYPE_PV
from pypower.idx_bus import REF as BUS_TYPE_REF
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)


_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_COL = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}


@dataclass(frozen=True, slots=True)
class GeneratorOperatingPoint:
    """Exact per-generator state carried between AC power-flow transitions."""

    generator_ids: np.ndarray
    p_mw: np.ndarray
    q_mvar: np.ndarray
    status: np.ndarray

    @classmethod
    def from_state(cls, state: GridFMState) -> GeneratorOperatingPoint | None:
        values = (
            getattr(state, "generator_ids", None),
            getattr(state, "generator_p_mw", None),
            getattr(state, "generator_q_mvar", None),
            getattr(state, "generator_status", None),
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise InvalidPhysicalState(
                "Generator operating point is incomplete in GridFMState."
            )

        ids = np.asarray(values[0], dtype=np.int64)
        p_mw = np.asarray(values[1], dtype=np.float64)
        q_mvar = np.asarray(values[2], dtype=np.float64)
        status = np.asarray(values[3], dtype=np.float64)

        if ids.ndim != 1:
            raise InvalidPhysicalState("Generator IDs must be one-dimensional.")
        if any(array.shape != ids.shape for array in (p_mw, q_mvar, status)):
            raise InvalidPhysicalState(
                "Generator operating-point arrays must have matching shapes."
            )
        if np.unique(ids).size != ids.size:
            raise InvalidPhysicalState("Generator IDs must be unique.")
        if not np.isfinite(p_mw).all() or not np.isfinite(q_mvar).all():
            raise InvalidPhysicalState(
                "Generator operating point contains NaN or infinity."
            )
        if not np.isfinite(status).all() or np.any(
            (status != 0.0) & (status != 1.0)
        ):
            raise InvalidPhysicalState(
                "Generator status must contain only 0 or 1."
            )

        return cls(
            generator_ids=ids,
            p_mw=p_mw,
            q_mvar=q_mvar,
            status=status,
        )


@dataclass(frozen=True, slots=True)
class ScenarioPowerFlowTemplate:
    """Immutable NumPy template for one source scenario.

    The template contains only canonical PYPOWER inputs and stable row identities.
    It has no cache policy and no mutable solver state.
    """

    scenario_id: int
    base_mva: float
    bus_ids: np.ndarray
    branch_ids: np.ndarray
    generator_ids: np.ndarray
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray


@dataclass(frozen=True, slots=True)
class CanonicalPowerFlowProblem:
    """One complete AC power-flow problem presented to PYPOWER."""

    base_mva: float
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray

    def to_ppc(self, *, copy: bool = False) -> dict[str, Any]:
        if copy:
            bus = self.bus.copy()
            branch = self.branch.copy()
            gen = self.gen.copy()
        else:
            bus = self.bus
            branch = self.branch
            gen = self.gen

        return {
            "version": "2",
            "baseMVA": float(self.base_mva),
            "bus": bus,
            "branch": branch,
            "gen": gen,
        }


def build_scenario_power_flow_template(
    *,
    scenario_id: int,
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    base_mva: float,
) -> ScenarioPowerFlowTemplate:
    """Compile GridFM frames once into canonical float64 solver matrices."""

    scenario_id = int(scenario_id)
    base_mva = float(base_mva)
    if not np.isfinite(base_mva) or base_mva <= 0.0:
        raise ValueError(f"base_mva must be finite and positive, got {base_mva}.")

    bus = _scenario_rows(bus_df, scenario_id, "bus", "bus_data")
    branch = _scenario_rows(branch_df, scenario_id, "idx", "branch_data")
    gen = _scenario_rows(gen_df, scenario_id, "idx", "gen_data")

    bus_matrix = _build_bus_matrix(bus, base_mva)
    branch_matrix = _build_branch_matrix(branch)
    gen_matrix = _build_gen_matrix(gen, bus, base_mva)

    return ScenarioPowerFlowTemplate(
        scenario_id=scenario_id,
        base_mva=base_mva,
        bus_ids=bus["bus"].to_numpy(dtype=np.int64, copy=True),
        branch_ids=branch["idx"].to_numpy(dtype=np.int64, copy=True),
        generator_ids=gen["idx"].to_numpy(dtype=np.int64, copy=True),
        bus=bus_matrix,
        branch=branch_matrix,
        gen=gen_matrix,
    )


def build_power_flow_problem_from_state(
    *,
    template: ScenarioPowerFlowTemplate,
    state: GridFMState,
    branch_id: int | None = None,
    target_status: int | None = None,
    generator_operating_point: GeneratorOperatingPoint | None = None,
) -> CanonicalPowerFlowProblem:
    """Build the repeated-transition PYPOWER input using NumPy only.

    DataFrame work is deliberately confined to template construction. Every beam
    search transition after that copies three compact float64 matrices and updates
    only solver-relevant dynamic columns.
    """

    if int(state.scenario_id) != int(template.scenario_id):
        raise InvalidPhysicalState(
            "Power-flow template scenario does not match GridFMState."
        )

    bus_features = np.asarray(state.bus_features)
    branch_features = np.asarray(state.branch_features)
    if bus_features.ndim != 2:
        raise InvalidPhysicalState("bus_features must be a 2D matrix.")
    if branch_features.ndim != 2:
        raise InvalidPhysicalState("branch_features must be a 2D matrix.")
    if bus_features.shape[0] != template.bus.shape[0]:
        raise InvalidPhysicalState(
            "Bus count does not match the power-flow scenario template."
        )
    if branch_features.shape[0] != template.branch.shape[0]:
        raise InvalidPhysicalState(
            "Branch count does not match the power-flow scenario template."
        )

    state_bus_ids = getattr(state, "bus_ids", None)
    if state_bus_ids is not None and not np.array_equal(
        np.asarray(state_bus_ids, dtype=np.int64),
        template.bus_ids,
    ):
        raise InvalidPhysicalState(
            "Bus IDs do not match the power-flow scenario template."
        )
    if not np.array_equal(
        np.asarray(state.branch_ids, dtype=np.int64),
        template.branch_ids,
    ):
        raise InvalidPhysicalState(
            "Branch IDs do not match the power-flow scenario template."
        )

    bus = template.bus.copy()
    branch = template.branch.copy()
    gen = template.gen.copy()

    _update_bus_from_state(bus, bus_features, template.base_mva)
    _update_branch_from_state(branch, branch_features)
    _apply_branch_action(
        branch=branch,
        branch_ids=template.branch_ids,
        branch_id=branch_id,
        target_status=target_status,
    )
    if generator_operating_point is not None:
        _apply_generator_operating_point(
            gen=gen,
            template=template,
            operating_point=generator_operating_point,
        )

    return CanonicalPowerFlowProblem(
        base_mva=float(template.base_mva),
        bus=bus,
        branch=branch,
        gen=gen,
    )


def _scenario_rows(
    frame: pd.DataFrame,
    scenario_id: int,
    order_column: str,
    source_name: str,
) -> pd.DataFrame:
    if "scenario" not in frame.columns:
        raise ValueError(f"{source_name} is missing scenario column.")
    rows = frame[frame["scenario"] == int(scenario_id)]
    if rows.empty:
        raise ValueError(f"Scenario {scenario_id} not found in {source_name}.")
    if order_column not in rows.columns:
        raise ValueError(f"{source_name} is missing {order_column} column.")
    return rows.sort_values(order_column).reset_index(drop=True)


def _build_bus_matrix(bus_df: pd.DataFrame, base_mva: float) -> np.ndarray:
    required = {
        "bus",
        "Pd",
        "Qd",
        "GS",
        "BS",
        "Vm",
        "Va",
        "vn_kv",
        "min_vm_pu",
        "max_vm_pu",
    }
    _require_columns(bus_df, required, "bus_data")

    bus = np.zeros((len(bus_df), 13), dtype=np.float64)
    bus[:, BUS_I] = bus_df["bus"].to_numpy(dtype=np.float64)
    bus[:, BUS_TYPE] = _bus_types_from_frame(bus_df)
    bus[:, PD] = bus_df["Pd"].to_numpy(dtype=np.float64)
    bus[:, QD] = bus_df["Qd"].to_numpy(dtype=np.float64)
    bus[:, GS] = bus_df["GS"].to_numpy(dtype=np.float64) * base_mva
    bus[:, BS] = bus_df["BS"].to_numpy(dtype=np.float64) * base_mva
    bus[:, BUS_AREA] = 1.0
    bus[:, VM] = bus_df["Vm"].to_numpy(dtype=np.float64)
    bus[:, VA] = bus_df["Va"].to_numpy(dtype=np.float64)
    bus[:, BASE_KV] = bus_df["vn_kv"].to_numpy(dtype=np.float64)
    bus[:, ZONE] = 1.0
    bus[:, VMAX] = bus_df["max_vm_pu"].to_numpy(dtype=np.float64)
    bus[:, VMIN] = bus_df["min_vm_pu"].to_numpy(dtype=np.float64)
    return bus


def _build_branch_matrix(branch_df: pd.DataFrame) -> np.ndarray:
    required = {
        "from_bus",
        "to_bus",
        "r",
        "x",
        "b",
        "tap",
        "shift",
        "rate_a",
        "br_status",
        "ang_min",
        "ang_max",
    }
    _require_columns(branch_df, required, "branch_data")

    branch = np.zeros((len(branch_df), 13), dtype=np.float64)
    branch[:, F_BUS] = branch_df["from_bus"].to_numpy(dtype=np.float64)
    branch[:, T_BUS] = branch_df["to_bus"].to_numpy(dtype=np.float64)
    branch[:, BR_R] = branch_df["r"].to_numpy(dtype=np.float64)
    branch[:, BR_X] = branch_df["x"].to_numpy(dtype=np.float64)
    branch[:, BR_B] = branch_df["b"].to_numpy(dtype=np.float64)
    rate_a = branch_df["rate_a"].to_numpy(dtype=np.float64)
    branch[:, RATE_A] = rate_a
    branch[:, RATE_B] = rate_a
    branch[:, RATE_C] = rate_a
    branch[:, TAP] = branch_df["tap"].to_numpy(dtype=np.float64)
    branch[:, SHIFT] = branch_df["shift"].to_numpy(dtype=np.float64)
    branch[:, BR_STATUS] = branch_df["br_status"].to_numpy(dtype=np.float64)
    branch[:, ANGMIN] = branch_df["ang_min"].to_numpy(dtype=np.float64)
    branch[:, ANGMAX] = branch_df["ang_max"].to_numpy(dtype=np.float64)
    return branch


def _build_gen_matrix(
    gen_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    base_mva: float,
) -> np.ndarray:
    required = {
        "bus",
        "p_mw",
        "q_mvar",
        "max_q_mvar",
        "min_q_mvar",
        "in_service",
        "max_p_mw",
        "min_p_mw",
    }
    _require_columns(gen_df, required, "gen_data")

    gen = np.zeros((len(gen_df), 21), dtype=np.float64)
    gen[:, GEN_BUS] = gen_df["bus"].to_numpy(dtype=np.float64)
    gen[:, PG] = gen_df["p_mw"].to_numpy(dtype=np.float64)
    gen[:, QG] = gen_df["q_mvar"].to_numpy(dtype=np.float64)
    gen[:, QMAX] = gen_df["max_q_mvar"].to_numpy(dtype=np.float64)
    gen[:, QMIN] = gen_df["min_q_mvar"].to_numpy(dtype=np.float64)

    vm_by_bus = dict(
        zip(
            bus_df["bus"].to_numpy(dtype=np.int64),
            bus_df["Vm"].to_numpy(dtype=np.float64),
        )
    )
    gen[:, VG] = np.asarray(
        [vm_by_bus.get(int(bus_id), 1.0) for bus_id in gen_df["bus"]],
        dtype=np.float64,
    )
    gen[:, MBASE] = base_mva
    gen[:, GEN_STATUS] = gen_df["in_service"].to_numpy(dtype=np.float64)
    gen[:, PMAX] = gen_df["max_p_mw"].to_numpy(dtype=np.float64)
    gen[:, PMIN] = gen_df["min_p_mw"].to_numpy(dtype=np.float64)
    return gen


def _update_bus_from_state(
    bus: np.ndarray,
    features: np.ndarray,
    base_mva: float,
) -> None:
    bus[:, BUS_TYPE] = BUS_TYPE_PQ
    pv = np.asarray(features[:, _BUS_COL["PV"]], dtype=np.float64) > 0.5
    ref = np.asarray(features[:, _BUS_COL["REF"]], dtype=np.float64) > 0.5
    bus[pv, BUS_TYPE] = BUS_TYPE_PV
    bus[ref, BUS_TYPE] = BUS_TYPE_REF
    bus[:, PD] = features[:, _BUS_COL["Pd"]]
    bus[:, QD] = features[:, _BUS_COL["Qd"]]
    bus[:, GS] = features[:, _BUS_COL["GS"]] * base_mva
    bus[:, BS] = features[:, _BUS_COL["BS"]] * base_mva
    bus[:, VM] = features[:, _BUS_COL["Vm"]]
    bus[:, VA] = features[:, _BUS_COL["Va"]]
    bus[:, BASE_KV] = features[:, _BUS_COL["vn_kv"]]
    bus[:, VMAX] = features[:, _BUS_COL["max_vm_pu"]]
    bus[:, VMIN] = features[:, _BUS_COL["min_vm_pu"]]


def _update_branch_from_state(branch: np.ndarray, features: np.ndarray) -> None:
    branch[:, BR_R] = features[:, _BRANCH_COL["r"]]
    branch[:, BR_X] = features[:, _BRANCH_COL["x"]]
    branch[:, BR_B] = features[:, _BRANCH_COL["b"]]
    branch[:, TAP] = features[:, _BRANCH_COL["tap"]]
    branch[:, SHIFT] = features[:, _BRANCH_COL["shift"]]
    rate_a = features[:, _BRANCH_COL["rate_a"]]
    branch[:, RATE_A] = rate_a
    branch[:, RATE_B] = rate_a
    branch[:, RATE_C] = rate_a
    branch[:, BR_STATUS] = features[:, _BRANCH_COL["br_status"]]


def _apply_branch_action(
    *,
    branch: np.ndarray,
    branch_ids: np.ndarray,
    branch_id: int | None,
    target_status: int | None,
) -> None:
    if branch_id is None:
        if target_status is not None:
            raise ValueError("target_status requires branch_id.")
        return
    if target_status not in (0, 1):
        raise ValueError("target_status must be either 0 or 1.")

    matches = np.flatnonzero(branch_ids == int(branch_id))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one branch id {branch_id}, found {len(matches)}."
        )
    position = int(matches[0])
    current_status = int(float(branch[position, BR_STATUS]) > 0.5)
    if current_status == int(target_status):
        raise ValueError(
            f"Branch id {branch_id} already has status {target_status}."
        )
    branch[position, BR_STATUS] = float(target_status)


def _apply_generator_operating_point(
    *,
    gen: np.ndarray,
    template: ScenarioPowerFlowTemplate,
    operating_point: GeneratorOperatingPoint,
) -> None:
    if not np.array_equal(
        operating_point.generator_ids,
        template.generator_ids,
    ):
        raise InvalidPhysicalState(
            "Generator operating point does not match scenario generator IDs."
        )
    if operating_point.p_mw.shape != template.generator_ids.shape:
        raise InvalidPhysicalState(
            "Generator operating-point length does not match the template."
        )

    gen[:, PG] = operating_point.p_mw
    gen[:, QG] = operating_point.q_mvar
    gen[:, GEN_STATUS] = operating_point.status


def _bus_types_from_frame(bus_df: pd.DataFrame) -> np.ndarray:
    bus_types = np.full(len(bus_df), BUS_TYPE_PQ, dtype=np.float64)
    if "PV" in bus_df.columns:
        bus_types[bus_df["PV"].to_numpy(dtype=np.float64) > 0.5] = BUS_TYPE_PV
    if "REF" in bus_df.columns:
        bus_types[bus_df["REF"].to_numpy(dtype=np.float64) > 0.5] = BUS_TYPE_REF
    return bus_types


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}.")
