from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

GEN_TOLERANCE = 1e-6
ANGLE_TOLERANCE_DEG = 1e-6
UNLIMITED_ANGLE_DEG = 359.999


def topology_connected(num_buses: int, edge_index: np.ndarray, branch_status: np.ndarray) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_buses)))
    for pos in range(len(branch_status)):
        if float(branch_status[pos]) <= 0.0:
            continue
        u = int(edge_index[0, pos])
        v = int(edge_index[1, pos])
        if u != v:
            graph.add_edge(u, v)
    if int(num_buses) == 0:
        return False
    return nx.is_connected(graph)


def finite_array(*arrays: np.ndarray) -> bool:
    return all(np.all(np.isfinite(np.asarray(arr, dtype=float))) for arr in arrays)


def gen_limit_metrics(gen: np.ndarray) -> dict[str, Any]:
    from pypower.idx_gen import GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN
    if gen.size == 0:
        return {"generator_p_feasible": True, "generator_q_feasible": True, "num_generator_p_violations": 0, "num_generator_q_violations": 0}
    active = gen[:, GEN_STATUS] > 0
    pg = gen[:, PG]; qg = gen[:, QG]
    p_bad = active & ((pg < gen[:, PMIN] - GEN_TOLERANCE) | (pg > gen[:, PMAX] + GEN_TOLERANCE))
    q_bad = active & ((qg < gen[:, QMIN] - GEN_TOLERANCE) | (qg > gen[:, QMAX] + GEN_TOLERANCE))
    return {"generator_p_feasible": int(np.sum(p_bad)) == 0, "generator_q_feasible": int(np.sum(q_bad)) == 0, "num_generator_p_violations": int(np.sum(p_bad)), "num_generator_q_violations": int(np.sum(q_bad))}


def angle_limit_metrics(bus: np.ndarray, branch: np.ndarray) -> dict[str, Any]:
    from pypower.idx_bus import BUS_I, VA
    from pypower.idx_brch import ANGMAX, ANGMIN, BR_STATUS, F_BUS, T_BUS
    if branch.size == 0:
        return {"angle_difference_feasible": True, "num_angle_difference_violations": 0}
    va_by_bus = {int(row[BUS_I]): float(row[VA]) for row in bus}
    bad = 0
    for row in branch:
        if float(row[BR_STATUS]) <= 0.0:
            continue
        amin = float(row[ANGMIN]); amax = float(row[ANGMAX])
        delta = va_by_bus[int(row[F_BUS])] - va_by_bus[int(row[T_BUS])]
        if amin <= -UNLIMITED_ANGLE_DEG and amax >= UNLIMITED_ANGLE_DEG:
            continue
        if amin > -UNLIMITED_ANGLE_DEG and delta < amin - ANGLE_TOLERANCE_DEG:
            bad += 1; continue
        if amax < UNLIMITED_ANGLE_DEG and delta > amax + ANGLE_TOLERANCE_DEG:
            bad += 1
    return {"angle_difference_feasible": bad == 0, "num_angle_difference_violations": bad}
