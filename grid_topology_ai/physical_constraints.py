from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import BUS_I, VA, VM, VMAX, VMIN
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
)

from grid_topology_ai.physical_objective import (
    ANGLE_LIMIT_TOLERANCE_DEGREES,
    GENERATOR_LIMIT_TOLERANCE_MVAR,
    GENERATOR_LIMIT_TOLERANCE_MW,
    HARD_OVERLOAD_LIMIT_PERCENT,
    OVERLOAD_LIMIT_PERCENT,
    THERMAL_LIMIT_TOLERANCE_PERCENT,
    VOLTAGE_LIMIT_TOLERANCE_PU,
)
from grid_topology_ai.config.physics import PhysicsConfig, ZeroRateAPolicy
from grid_topology_ai.power_flow_errors import InvalidPhysicalState


@dataclass(frozen=True, slots=True)
class PhysicalNetworkArrays:
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray


def arrays_from_pypower_result(result_ppc: dict[str, Any]) -> PhysicalNetworkArrays:
    return PhysicalNetworkArrays(
        bus=np.asarray(result_ppc["bus"], dtype=float),
        branch=np.asarray(result_ppc["branch"], dtype=float),
        gen=np.asarray(result_ppc["gen"], dtype=float),
    )


def arrays_from_gridfm_frames(
    *,
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> PhysicalNetworkArrays:
    bus = np.zeros((len(bus_df), 13), dtype=float)
    bus[:, BUS_I] = bus_df["bus"].to_numpy(dtype=float)
    bus[:, VM] = bus_df["Vm"].to_numpy(dtype=float)
    bus[:, VA] = bus_df["Va"].to_numpy(dtype=float)
    bus[:, VMIN] = bus_df["min_vm_pu"].to_numpy(dtype=float)
    bus[:, VMAX] = bus_df["max_vm_pu"].to_numpy(dtype=float)

    branch = np.zeros((len(branch_df), QT + 1), dtype=float)
    branch[:, F_BUS] = branch_df["from_bus"].to_numpy(dtype=float)
    branch[:, T_BUS] = branch_df["to_bus"].to_numpy(dtype=float)
    branch[:, RATE_A] = branch_df["rate_a"].to_numpy(dtype=float)
    branch[:, BR_STATUS] = branch_df["br_status"].to_numpy(dtype=float)
    branch[:, ANGMIN] = branch_df["ang_min"].to_numpy(dtype=float)
    branch[:, ANGMAX] = branch_df["ang_max"].to_numpy(dtype=float)
    branch[:, PF] = branch_df["pf"].to_numpy(dtype=float)
    branch[:, QF] = branch_df["qf"].to_numpy(dtype=float)
    branch[:, PT] = branch_df["pt"].to_numpy(dtype=float)
    branch[:, QT] = branch_df["qt"].to_numpy(dtype=float)

    gen = np.zeros((len(gen_df), 21), dtype=float)
    gen[:, GEN_BUS] = gen_df["bus"].to_numpy(dtype=float)
    gen[:, PG] = gen_df["p_mw"].to_numpy(dtype=float)
    gen[:, QG] = gen_df["q_mvar"].to_numpy(dtype=float)
    gen[:, QMIN] = gen_df["min_q_mvar"].to_numpy(dtype=float)
    gen[:, QMAX] = gen_df["max_q_mvar"].to_numpy(dtype=float)
    gen[:, GEN_STATUS] = gen_df["in_service"].to_numpy(dtype=float)
    gen[:, PMIN] = gen_df["min_p_mw"].to_numpy(dtype=float)
    gen[:, PMAX] = gen_df["max_p_mw"].to_numpy(dtype=float)
    return PhysicalNetworkArrays(bus=bus, branch=branch, gen=gen)


def _require_matrix(name: str, value: np.ndarray, min_columns: int) -> np.ndarray:
    if value.ndim != 2 or value.shape[1] < min_columns:
        raise ValueError(
            f"{name} must be a 2D PYPOWER matrix with at least "
            f"{min_columns} columns; got {value.shape}."
        )
    return value


def _matrix(ppc: dict[str, Any], name: str, columns: int, context: str) -> np.ndarray:
    if name not in ppc:
        raise InvalidPhysicalState(f"{context}: missing required {name} matrix.")
    try:
        array = np.asarray(ppc[name], dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidPhysicalState(f"{context}: {name} is not numeric.") from exc
    if array.ndim != 2 or array.shape[1] < columns:
        raise InvalidPhysicalState(f"{context}: {name} must be 2D with at least {columns} columns; got {array.shape}.")
    return array


def validate_ppc_input(ppc: dict[str, Any], physics_config: PhysicsConfig, *, context: str = "ppc") -> None:
    """Validate structural integrity before PYPOWER sees a case."""
    if not isinstance(ppc, dict):
        raise InvalidPhysicalState(f"{context}: ppc must be a mapping.")
    bus, branch, gen = (_matrix(ppc, "bus", VMIN + 1, context), _matrix(ppc, "branch", ANGMAX + 1, context), _matrix(ppc, "gen", PMIN + 1, context))
    if not np.isfinite(bus).all() or not np.isfinite(branch).all() or not np.isfinite(gen).all():
        raise InvalidPhysicalState(f"{context}: input matrices contain NaN or infinity.")
    ids = bus[:, BUS_I]
    if not np.equal(ids, np.rint(ids)).all() or len(set(ids.astype(int))) != len(ids):
        raise InvalidPhysicalState(f"{context}: bus.BUS_I must contain unique integral IDs.")
    known = set(ids.astype(int))
    active_branch = branch[:, BR_STATUS] > 0
    for row in np.flatnonzero(active_branch):
        if not float(branch[row, F_BUS]).is_integer() or not float(branch[row, T_BUS]).is_integer() or int(branch[row, F_BUS]) not in known or int(branch[row, T_BUS]) not in known:
            raise InvalidPhysicalState(f"{context}: branch row {row} references an unknown bus.")
    active_gen = gen[:, GEN_STATUS] > 0
    for row in np.flatnonzero(active_gen):
        if not float(gen[row, GEN_BUS]).is_integer() or int(gen[row, GEN_BUS]) not in known:
            raise InvalidPhysicalState(f"{context}: gen row {row} references an unknown bus.")
    if np.any(bus[:, VMIN] > bus[:, VMAX]) or np.any(gen[:, PMIN] > gen[:, PMAX]) or np.any(gen[:, QMIN] > gen[:, QMAX]):
        raise InvalidPhysicalState(f"{context}: min/max limits are inverted.")
    rate = branch[active_branch, RATE_A]
    if np.any(rate < 0) or (physics_config.zero_rate_a_policy is ZeroRateAPolicy.ERROR and np.any(rate == 0)):
        raise InvalidPhysicalState(f"{context}: active branch RATE_A is invalid for configured policy.")
    graph = nx.Graph(); graph.add_nodes_from(known); graph.add_edges_from((int(branch[r, F_BUS]), int(branch[r, T_BUS])) for r in np.flatnonzero(active_branch))
    if physics_config.island_policy.value == "reject" and (not known or not nx.is_connected(graph)):
        raise InvalidPhysicalState(f"{context}: active topology is disconnected.")
    # PYPOWER may select/normalise the reference bus during case preparation;
    # retain compatibility with GridFM inputs where REF is inferred downstream.


def validate_pypower_result(result_ppc: dict[str, Any], physics_config: PhysicsConfig, *, input_ppc: dict[str, Any], context: str = "result") -> None:
    for name in ("bus", "branch", "gen"):
        if name not in result_ppc or not np.isfinite(np.asarray(result_ppc[name], dtype=float)).all():
            raise InvalidPhysicalState(f"{context}: {name} result contains non-finite values.")
    validate_ppc_input(result_ppc, physics_config, context=context)
    for name in ("bus", "branch", "gen"):
        if np.asarray(result_ppc[name]).shape[0] != np.asarray(input_ppc[name]).shape[0]:
            raise InvalidPhysicalState(f"{context}: {name} row count differs from input.")
    if np.asarray(result_ppc["branch"]).shape[1] < QT + 1:
        raise InvalidPhysicalState(f"{context}: branch result lacks flow columns.")


def _finite_sum(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sum(finite)) if finite.size else 0.0


def calculate_physical_metrics(
    arrays: PhysicalNetworkArrays,
    *,
    power_flow_converged: bool, physics_config: PhysicsConfig | None = None,
) -> dict[str, object]:
    bus = _require_matrix("bus", arrays.bus, VMIN + 1)
    branch = _require_matrix("branch", arrays.branch, QT + 1)
    gen = _require_matrix("gen", arrays.gen, PMIN + 1)

    bus_ids = bus[:, BUS_I]
    vm = bus[:, VM]
    va = bus[:, VA]
    vmin = bus[:, VMIN]
    vmax = bus[:, VMAX]
    branch_status = branch[:, BR_STATUS]
    gen_status = gen[:, GEN_STATUS]
    active_branch = np.isfinite(branch_status) & (branch_status > 0.0)
    active_gen = np.isfinite(gen_status) & (gen_status > 0.0)

    required_finite_masks = [
        np.isfinite(bus_ids),
        np.isfinite(vm),
        np.isfinite(va),
        np.isfinite(vmin),
        np.isfinite(vmax),
        np.isfinite(branch_status),
        np.isfinite(gen_status),
        np.isfinite(branch[active_branch, F_BUS]),
        np.isfinite(branch[active_branch, T_BUS]),
        np.isfinite(branch[active_branch, RATE_A]),
        np.isfinite(branch[active_branch, ANGMIN]),
        np.isfinite(branch[active_branch, ANGMAX]),
        np.isfinite(branch[active_branch, PF]),
        np.isfinite(branch[active_branch, QF]),
        np.isfinite(branch[active_branch, PT]),
        np.isfinite(branch[active_branch, QT]),
        np.isfinite(gen[active_gen, GEN_BUS]),
        np.isfinite(gen[active_gen, PG]),
        np.isfinite(gen[active_gen, PMIN]),
        np.isfinite(gen[active_gen, PMAX]),
        np.isfinite(gen[active_gen, QG]),
        np.isfinite(gen[active_gen, QMIN]),
        np.isfinite(gen[active_gen, QMAX]),
    ]
    all_values_finite = all(bool(mask.all()) for mask in required_finite_masks)

    bus_ids_integral = np.isfinite(bus_ids) & np.equal(bus_ids, np.rint(bus_ids))
    finite_integral_ids = bus_ids[bus_ids_integral].astype(np.int64, copy=False)
    unique_bus_ids = set(int(value) for value in finite_integral_ids)
    bus_ids_valid = (
        bool(bus_ids_integral.all()) and len(unique_bus_ids) == len(bus_ids)
    )
    bus_id_to_position = {
        int(bus_id): position
        for position, bus_id in enumerate(finite_integral_ids)
    } if bus_ids_valid else {}

    graph = nx.Graph()
    graph.add_nodes_from(range(len(bus)))
    endpoints_valid = True
    for branch_row in branch[active_branch]:
        from_id = branch_row[F_BUS]
        to_id = branch_row[T_BUS]
        if (
            not np.isfinite(from_id)
            or not np.isfinite(to_id)
            or not float(from_id).is_integer()
            or not float(to_id).is_integer()
        ):
            endpoints_valid = False
            continue
        from_pos = bus_id_to_position.get(int(from_id))
        to_pos = bus_id_to_position.get(int(to_id))
        if from_pos is None or to_pos is None:
            endpoints_valid = False
            continue
        graph.add_edge(from_pos, to_pos)
    topology_connected = bool(
        bus_ids_valid
        and endpoints_valid
        and len(bus) > 0
        and nx.is_connected(graph)
    )

    pf = branch[:, PF]
    qf = branch[:, QF]
    pt = branch[:, PT]
    qt = branch[:, QT]
    s_from = np.sqrt(np.square(pf) + np.square(qf))
    s_to = np.sqrt(np.square(pt) + np.square(qt))
    s_max = np.maximum(s_from, s_to)
    rate_a = branch[:, RATE_A]
    constrained = active_branch & np.isfinite(rate_a) & (rate_a > 0.0)
    invalid_rate = active_branch & (~np.isfinite(rate_a) | (rate_a < 0.0))
    loading = np.zeros(len(branch), dtype=float)
    loading[constrained] = s_max[constrained] / rate_a[constrained] * 100.0
    invalid_flow = active_branch & ~np.isfinite(s_max)
    config = physics_config or PhysicsConfig()
    overload = constrained & (
        loading > config.overload_limit_percent + config.thermal_tolerance_percent
    )
    hard_overload = constrained & (
        loading > config.hard_overload_limit_percent + config.thermal_tolerance_percent
    )
    thermal_failure = overload | invalid_rate | invalid_flow
    hard_failure = hard_overload | invalid_rate | invalid_flow
    finite_active_loading = loading[active_branch & np.isfinite(loading)]
    max_loading = (
        float(np.max(finite_active_loading)) if finite_active_loading.size else 0.0
    )
    thermal_excess_mva = np.maximum(s_max - np.maximum(rate_a, 0.0), 0.0)

    low_voltage = np.maximum(vmin - vm - config.voltage_tolerance_pu, 0.0)
    high_voltage = np.maximum(vm - vmax - config.voltage_tolerance_pu, 0.0)
    invalid_voltage = ~(np.isfinite(vm) & np.isfinite(vmin) & np.isfinite(vmax))
    low_voltage_mask = (low_voltage > 0.0) | invalid_voltage
    high_voltage_mask = (high_voltage > 0.0) | invalid_voltage

    pg = gen[:, PG]
    pmin = gen[:, PMIN]
    pmax = gen[:, PMAX]
    qg = gen[:, QG]
    qmin = gen[:, QMIN]
    qmax = gen[:, QMAX]
    invalid_p = active_gen & ~(
        np.isfinite(pg) & np.isfinite(pmin) & np.isfinite(pmax)
    )
    invalid_q = active_gen & ~(
        np.isfinite(qg) & np.isfinite(qmin) & np.isfinite(qmax)
    )
    generator_bus_valid = np.zeros(len(gen), dtype=bool)
    for position in np.flatnonzero(active_gen):
        generator_bus = gen[position, GEN_BUS]
        generator_bus_valid[position] = bool(
            np.isfinite(generator_bus)
            and float(generator_bus).is_integer()
            and int(generator_bus) in bus_id_to_position
        )
    invalid_generator_bus = active_gen & ~generator_bus_valid
    invalid_p |= invalid_generator_bus
    invalid_q |= invalid_generator_bus
    low_p = np.maximum(pmin - pg - config.generator_p_tolerance_mw, 0.0)
    high_p = np.maximum(pg - pmax - config.generator_p_tolerance_mw, 0.0)
    low_q = np.maximum(qmin - qg - config.generator_q_tolerance_mvar, 0.0)
    high_q = np.maximum(qg - qmax - config.generator_q_tolerance_mvar, 0.0)
    p_violation = active_gen & ((low_p > 0.0) | (high_p > 0.0) | invalid_p)
    q_violation = active_gen & ((low_q > 0.0) | (high_q > 0.0) | invalid_q)

    angle_violation = np.zeros(len(branch), dtype=bool)
    angle_excess = np.zeros(len(branch), dtype=float)
    for position in np.flatnonzero(active_branch):
        from_id = branch[position, F_BUS]
        to_id = branch[position, T_BUS]
        if not (
            np.isfinite(from_id)
            and np.isfinite(to_id)
            and float(from_id).is_integer()
            and float(to_id).is_integer()
        ):
            angle_violation[position] = True
            continue
        from_pos = bus_id_to_position.get(int(from_id))
        to_pos = bus_id_to_position.get(int(to_id))
        if from_pos is None or to_pos is None:
            angle_violation[position] = True
            continue
        angle_min = branch[position, ANGMIN]
        angle_max = branch[position, ANGMAX]
        if not (
            np.isfinite(va[from_pos])
            and np.isfinite(va[to_pos])
            and np.isfinite(angle_min)
            and np.isfinite(angle_max)
        ):
            angle_violation[position] = True
            continue
        difference = va[from_pos] - va[to_pos]
        lower_excess = (
            max(angle_min - difference - config.angle_tolerance_degrees, 0.0)
            if angle_min > -360.0
            else 0.0
        )
        upper_excess = (
            max(difference - angle_max - config.angle_tolerance_degrees, 0.0)
            if angle_max < 360.0
            else 0.0
        )
        angle_excess[position] = lower_excess + upper_excess
        angle_violation[position] = angle_excess[position] > 0.0

    return {
        "power_flow_converged": bool(power_flow_converged),
        "all_values_finite": bool(all_values_finite),
        "topology_connected": topology_connected,
        "max_loading_percent": max_loading,
        "num_overloaded_branches": int(np.sum(thermal_failure)),
        "num_hard_overloaded_branches": int(np.sum(hard_failure)),
        "num_unrated_active_branches": int(np.sum(active_branch & (rate_a == 0.0))),
        "total_thermal_overload_mva": _finite_sum(
            thermal_excess_mva[constrained]
        ),
        "num_low_voltage_buses": int(np.sum(low_voltage_mask)),
        "num_high_voltage_buses": int(np.sum(high_voltage_mask)),
        "total_low_voltage_violation": _finite_sum(low_voltage),
        "total_high_voltage_violation": _finite_sum(high_voltage),
        "total_voltage_violation": _finite_sum(low_voltage + high_voltage),
        "num_generator_p_violations": int(np.sum(p_violation)),
        "total_generator_p_violation_mw": _finite_sum(
            (low_p + high_p)[active_gen]
        ),
        "num_generator_q_violations": int(np.sum(q_violation)),
        "total_generator_q_violation_mvar": _finite_sum(
            (low_q + high_q)[active_gen]
        ),
        "num_angle_difference_violations": int(np.sum(angle_violation)),
        "total_angle_difference_violation_degrees": _finite_sum(angle_excess),
    }


def calculate_physical_metrics_from_result(
    result_ppc: dict[str, Any],
    *,
    power_flow_converged: bool, physics_config: PhysicsConfig | None = None,
) -> dict[str, object]:
    return calculate_physical_metrics(
        arrays_from_pypower_result(result_ppc),
        power_flow_converged=power_flow_converged, physics_config=physics_config,
    )


def calculate_physical_metrics_from_frames(
    *,
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    power_flow_converged: bool,
    physics_config: PhysicsConfig | None = None,
) -> dict[str, object]:
    return calculate_physical_metrics(
        arrays_from_gridfm_frames(
            bus_df=bus_df,
            branch_df=branch_df,
            gen_df=gen_df,
        ),
        power_flow_converged=power_flow_converged,
        physics_config=physics_config,
    )
