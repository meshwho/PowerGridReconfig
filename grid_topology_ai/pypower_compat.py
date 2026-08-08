from __future__ import annotations

from copy import deepcopy
from sys import stderr, stdout
from typing import Any

import numpy as np
from pypower.api import case9, runpf as _runpf
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VA, PV, PQ, REF
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PG,
    QG,
    QMAX,
    QMIN,
)
from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
from pypower.printpf import printpf
from pypower.savecase import savecase


_WORKLOAD_COUNTERS = {
    "stock_runpf_calls": 0,
    "q_limit_resolves": 0,
}


def get_power_flow_workload_counters() -> dict[str, int]:
    """Return process-local counters for stock PYPOWER solves."""

    return {
        "stock_runpf_calls": int(_WORKLOAD_COUNTERS["stock_runpf_calls"]),
        "q_limit_resolves": int(_WORKLOAD_COUNTERS["q_limit_resolves"]),
    }


def reset_power_flow_workload_counters() -> None:
    """Reset process-local PYPOWER workload counters."""

    _WORKLOAD_COUNTERS["stock_runpf_calls"] = 0
    _WORKLOAD_COUNTERS["q_limit_resolves"] = 0


def _record_stock_runpf(*, q_limit_resolve: bool = False) -> None:
    _WORKLOAD_COUNTERS["stock_runpf_calls"] += 1

    if q_limit_resolve:
        _WORKLOAD_COUNTERS["q_limit_resolves"] += 1


def _load_case(casedata: Any) -> dict[str, Any]:
    case = loadcase(case9() if casedata is None else casedata)
    if not isinstance(case, dict):
        raise ValueError("PYPOWER could not load the supplied case.")
    return case


def _bus_rows(bus: np.ndarray, gen: np.ndarray) -> np.ndarray:
    bus_ids = np.asarray(bus[:, BUS_I], dtype=float)
    gen_bus_ids = np.asarray(gen[:, GEN_BUS], dtype=float)

    if (
        not np.isfinite(bus_ids).all()
        or not np.equal(bus_ids, np.rint(bus_ids)).all()
        or not np.isfinite(gen_bus_ids).all()
        or not np.equal(gen_bus_ids, np.rint(gen_bus_ids)).all()
    ):
        raise ValueError("PYPOWER bus references must be finite integers.")

    integral_bus_ids = bus_ids.astype(np.int64)
    if len(np.unique(integral_bus_ids)) != len(integral_bus_ids):
        raise ValueError("PYPOWER case contains duplicate bus IDs.")

    row_by_id = {
        int(bus_id): row
        for row, bus_id in enumerate(integral_bus_ids)
    }

    try:
        return np.asarray(
            [row_by_id[int(bus_id)] for bus_id in gen_bus_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"Generator references unknown bus ID {int(exc.args[0])}."
        ) from exc


def _violations(
    gen: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = gen[:, GEN_STATUS] > 0
    upper = np.flatnonzero(
        active & (gen[:, QG] > gen[:, QMAX] + tolerance)
    )
    lower = np.flatnonzero(
        active & (gen[:, QG] < gen[:, QMIN] - tolerance)
    )
    return upper, lower


def _remaining_voltage_controlled(
    bus: np.ndarray,
    gen: np.ndarray,
) -> np.ndarray:
    rows = _bus_rows(bus, gen)
    types = bus[rows, BUS_TYPE]
    active = gen[:, GEN_STATUS] > 0
    return np.flatnonzero(
        active & ((types == PV) | (types == REF))
    )


def _largest_violation_only(
    gen: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper_excess = gen[upper, QG] - gen[upper, QMAX]
    lower_excess = gen[lower, QMIN] - gen[lower, QG]
    combined = np.r_[upper_excess, lower_excess]

    if not len(combined):
        return upper, lower

    index = int(np.argmax(combined))
    empty = np.empty(0, dtype=np.int64)

    if index < len(upper):
        return np.asarray([upper[index]], dtype=np.int64), empty

    return empty, np.asarray(
        [lower[index - len(upper)]],
        dtype=np.int64,
    )


def _next_case(
    result: dict[str, Any],
    upper: np.ndarray,
    lower: np.ndarray,
    fixed_pg: np.ndarray,
    fixed_qg: np.ndarray,
    limited: list[int],
) -> dict[str, Any]:
    case = {
        key: deepcopy(value)
        for key, value in result.items()
        if key not in {"et", "order", "success"}
    }
    bus = np.asarray(case["bus"], dtype=float)
    gen = np.asarray(case["gen"], dtype=float)
    rows = _bus_rows(bus, gen)

    selected = np.r_[upper, lower].astype(np.int64, copy=False)
    reference_rows = set(
        np.flatnonzero(bus[:, BUS_TYPE] == REF).tolist()
    )

    if len(reference_rows) > 1 and any(
        int(rows[int(gen_index)]) in reference_rows
        for gen_index in selected
    ):
        raise RuntimeError(
            "Q-limit enforcement cannot convert a slack generator "
            "when multiple reference buses are active."
        )

    q_limits = {
        int(gen_index): float(gen[int(gen_index), QMAX])
        for gen_index in upper
    }
    q_limits.update(
        {
            int(gen_index): float(gen[int(gen_index), QMIN])
            for gen_index in lower
        }
    )

    for raw_index in selected:
        gen_index = int(raw_index)
        bus_row = int(rows[gen_index])

        fixed_pg[gen_index] = float(gen[gen_index, PG])
        fixed_qg[gen_index] = q_limits[gen_index]
        limited.append(gen_index)

        gen[gen_index, QG] = fixed_qg[gen_index]
        gen[gen_index, GEN_STATUS] = 0.0
        bus[bus_row, PD] -= fixed_pg[gen_index]
        bus[bus_row, QD] -= fixed_qg[gen_index]
        bus[bus_row, BUS_TYPE] = PQ

    rows = _bus_rows(bus, gen)
    active_bus_rows = sorted(
        set(int(row) for row in rows[gen[:, GEN_STATUS] > 0])
    )
    active_refs = [
        row
        for row in active_bus_rows
        if int(bus[row, BUS_TYPE]) == REF
    ]

    if not active_refs:
        active_pvs = [
            row
            for row in active_bus_rows
            if int(bus[row, BUS_TYPE]) == PV
        ]
        if not active_pvs:
            raise RuntimeError(
                "No active PV generator remains for a replacement "
                "reference bus during Q-limit enforcement."
            )
        bus[active_pvs[0], BUS_TYPE] = REF

    case["bus"] = bus
    case["gen"] = gen
    return case


def _restore(
    result: dict[str, Any],
    limited: list[int],
    fixed_pg: np.ndarray,
    fixed_qg: np.ndarray,
    original_ref: tuple[int, float] | None,
    original_order: Any,
) -> dict[str, Any]:
    restored = deepcopy(result)
    bus = np.asarray(restored["bus"], dtype=float)
    gen = np.asarray(restored["gen"], dtype=float)

    if limited:
        rows = _bus_rows(bus, gen)

        for gen_index in limited:
            bus_row = int(rows[gen_index])
            gen[gen_index, PG] = fixed_pg[gen_index]
            gen[gen_index, QG] = fixed_qg[gen_index]
            gen[gen_index, GEN_STATUS] = 1.0
            bus[bus_row, PD] += fixed_pg[gen_index]
            bus[bus_row, QD] += fixed_qg[gen_index]

    if original_ref is not None:
        ref_bus_id, ref_angle = original_ref
        bus_ids = np.asarray(bus[:, BUS_I], dtype=float)
        matches = np.flatnonzero(
            np.rint(bus_ids).astype(np.int64) == ref_bus_id
        )
        if len(matches) == 1:
            row = int(matches[0])
            bus[:, VA] += ref_angle - float(bus[row, VA])

    restored["bus"] = bus
    restored["gen"] = gen

    if original_order is not None:
        restored["order"] = deepcopy(original_order)

    return restored


def _emit(
    result: dict[str, Any],
    options: dict[str, Any],
    fname: str,
    solvedcase: str,
) -> None:
    if fname:
        try:
            with open(fname, "a") as stream:
                printpf(result, stream, options)
        except OSError as exc:
            stderr.write(f"Error opening {fname}: {exc}.\n")
    else:
        printpf(result, stdout, options)

    if solvedcase:
        savecase(solvedcase, result)


def _run_with_q_limits(
    casedata: Any,
    options: dict[str, Any],
    fname: str,
    solvedcase: str,
) -> tuple[dict[str, Any], bool]:
    case = _load_case(casedata)
    initial_bus = np.asarray(case["bus"], dtype=float)
    ref_rows = np.flatnonzero(initial_bus[:, BUS_TYPE] == REF)
    original_ref = None

    if len(ref_rows) == 1:
        row = int(ref_rows[0])
        original_ref = (
            int(round(float(initial_bus[row, BUS_I]))),
            float(initial_bus[row, VA]),
        )

    solver_options = ppoption(
        options,
        ENFORCE_Q_LIMS=0,
        VERBOSE=0,
        OUT_ALL=0,
    )
    tolerance = float(options["OPF_VIOLATION"])
    qlim_mode = int(options["ENFORCE_Q_LIMS"])
    ng = int(np.asarray(case["gen"]).shape[0])

    fixed_pg = np.full(ng, np.nan, dtype=float)
    fixed_qg = np.full(ng, np.nan, dtype=float)
    limited: list[int] = []
    original_order = None
    elapsed = 0.0
    working = case

    for iteration in range(ng + 1):
        _record_stock_runpf(q_limit_resolve=iteration > 0)
        result, success = _runpf(
            working,
            solver_options,
            "",
            "",
        )
        elapsed += float(result.get("et", 0.0))

        if original_order is None:
            original_order = deepcopy(result.get("order"))

        if not bool(success):
            final = _restore(
                result,
                limited,
                fixed_pg,
                fixed_qg,
                original_ref,
                original_order,
            )
            final["success"] = False
            final["et"] = elapsed
            _emit(final, options, fname, solvedcase)
            return final, False

        bus = np.asarray(result["bus"], dtype=float)
        gen = np.asarray(result["gen"], dtype=float)
        upper, lower = _violations(gen, tolerance)

        if not len(upper) and not len(lower):
            final = _restore(
                result,
                limited,
                fixed_pg,
                fixed_qg,
                original_ref,
                original_order,
            )
            final["success"] = True
            final["et"] = elapsed
            _emit(final, options, fname, solvedcase)
            return final, True

        infeasible = np.union1d(upper, lower)
        remaining = _remaining_voltage_controlled(bus, gen)

        if (
            len(infeasible) == len(remaining)
            and np.array_equal(infeasible, remaining)
        ):
            final = _restore(
                result,
                limited,
                fixed_pg,
                fixed_qg,
                original_ref,
                original_order,
            )
            final["success"] = False
            final["et"] = elapsed
            _emit(final, options, fname, solvedcase)
            return final, False

        if qlim_mode == 2:
            upper, lower = _largest_violation_only(
                gen,
                upper,
                lower,
            )

        try:
            working = _next_case(
                result,
                upper,
                lower,
                fixed_pg,
                fixed_qg,
                limited,
            )
        except RuntimeError:
            final = _restore(
                result,
                limited,
                fixed_pg,
                fixed_qg,
                original_ref,
                original_order,
            )
            final["success"] = False
            final["et"] = elapsed
            _emit(final, options, fname, solvedcase)
            return final, False

    raise RuntimeError(
        "Q-limit enforcement exceeded the number of available generators."
    )


def runpf(
    casedata: Any = None,
    ppopt: Any = None,
    fname: str = "",
    solvedcase: str = "",
):
    """Run PYPOWER, replacing only its broken Q-limit enforcement path."""

    options = ppoption(ppopt)

    if (
        not bool(options["ENFORCE_Q_LIMS"])
        or bool(options["PF_DC"])
    ):
        _record_stock_runpf()
        return _runpf(
            casedata,
            options,
            fname,
            solvedcase,
        )

    return _run_with_q_limits(
        casedata,
        options,
        fname,
        solvedcase,
    )
