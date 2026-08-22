from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.idx_brch import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import BS, BUS_I, GS, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG
from pypower.int2ext import int2ext
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.pfsoln import pfsoln
from pypower.pplinsolve import pplinsolve


@dataclass(frozen=True)
class PowerDerivativeWorkspace:
    """Fixed CSR structure used to evaluate AC power derivatives."""

    ybus: csr_matrix
    rows: np.ndarray
    columns: np.ndarray
    diagonal_positions: np.ndarray

    @classmethod
    def from_ybus(cls, ybus: Any) -> "PowerDerivativeWorkspace":
        matrix = csr_matrix(ybus, dtype=np.complex128, copy=True)
        matrix.sort_indices()

        # Keep an explicit diagonal slot in every row. The value is unchanged;
        # the slot only lets the diagonal derivative term be added in place.
        diagonal = matrix.diagonal().copy()
        matrix.setdiag(diagonal)
        matrix.sort_indices()

        row_count = matrix.shape[0]
        rows = np.repeat(np.arange(row_count, dtype=np.int64), np.diff(matrix.indptr))
        columns = np.array(matrix.indices, dtype=np.int64, copy=True)
        diagonal_positions = np.empty(row_count, dtype=np.int64)

        for row in range(row_count):
            start = int(matrix.indptr[row])
            stop = int(matrix.indptr[row + 1])
            matches = np.flatnonzero(columns[start:stop] == row)
            if len(matches) != 1:
                raise ValueError("Ybus must contain one diagonal entry per bus.")
            diagonal_positions[row] = start + int(matches[0])

        return cls(
            ybus=matrix,
            rows=rows,
            columns=columns,
            diagonal_positions=diagonal_positions,
        )

    def derivative_data(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        voltage = np.asarray(voltage, dtype=np.complex128)
        if voltage.ndim != 1 or len(voltage) != self.ybus.shape[0]:
            raise ValueError("Voltage vector does not match Ybus dimensions.")

        magnitude = np.abs(voltage)
        if np.any(magnitude == 0.0):
            raise ValueError("Voltage magnitude must be non-zero.")

        current = self.ybus @ voltage
        unit_voltage = voltage / magnitude
        rows = self.rows
        columns = self.columns
        ydata = self.ybus.data

        d_vm_data = voltage[rows] * np.conj(ydata * unit_voltage[columns])
        d_va_data = -1j * voltage[rows] * np.conj(ydata * voltage[columns])

        diagonal = self.diagonal_positions
        d_vm_data[diagonal] += np.conj(current) * unit_voltage
        d_va_data[diagonal] += 1j * voltage * np.conj(current)
        return d_vm_data, d_va_data

    def derivatives(self, voltage: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
        d_vm_data, d_va_data = self.derivative_data(voltage)
        shape = self.ybus.shape
        d_vm = csr_matrix(
            (d_vm_data, self.ybus.indices, self.ybus.indptr),
            shape=shape,
            copy=False,
        )
        d_va = csr_matrix(
            (d_va_data, self.ybus.indices, self.ybus.indptr),
            shape=shape,
            copy=False,
        )
        return d_vm, d_va


@dataclass
class JacobianWorkspace:
    """Fixed CSR layout for one Newton PV/PQ partition."""

    derivatives: PowerDerivativeWorkspace
    matrix: csr_matrix
    va_real_output: np.ndarray
    va_real_source: np.ndarray
    vm_real_output: np.ndarray
    vm_real_source: np.ndarray
    va_imag_output: np.ndarray
    va_imag_source: np.ndarray
    vm_imag_output: np.ndarray
    vm_imag_source: np.ndarray

    @classmethod
    def from_derivatives(
        cls,
        derivatives: PowerDerivativeWorkspace,
        pv: np.ndarray,
        pq: np.ndarray,
    ) -> "JacobianWorkspace":
        bus_count = derivatives.ybus.shape[0]
        pv = np.asarray(pv, dtype=np.int64)
        pq = np.asarray(pq, dtype=np.int64)
        if pv.ndim != 1 or pq.ndim != 1:
            raise ValueError("PV and PQ bus indices must be one-dimensional.")
        if np.any(pv < 0) or np.any(pv >= bus_count):
            raise ValueError("PV bus index is outside Ybus dimensions.")
        if np.any(pq < 0) or np.any(pq >= bus_count):
            raise ValueError("PQ bus index is outside Ybus dimensions.")
        if len(np.unique(pv)) != len(pv) or len(np.unique(pq)) != len(pq):
            raise ValueError("PV and PQ bus indices must be unique.")
        if np.intersect1d(pv, pq).size:
            raise ValueError("PV and PQ bus sets must not overlap.")

        pvpq = np.r_[pv, pq]
        angle_count = len(pvpq)
        state_size = angle_count + len(pq)

        angle_column = np.full(bus_count, -1, dtype=np.int64)
        magnitude_column = np.full(bus_count, -1, dtype=np.int64)
        active_row = np.full(bus_count, -1, dtype=np.int64)
        reactive_row = np.full(bus_count, -1, dtype=np.int64)

        angle_column[pvpq] = np.arange(angle_count, dtype=np.int64)
        magnitude_column[pq] = angle_count + np.arange(len(pq), dtype=np.int64)
        active_row[pvpq] = np.arange(angle_count, dtype=np.int64)
        reactive_row[pq] = angle_count + np.arange(len(pq), dtype=np.int64)

        source_rows = derivatives.rows
        source_columns = derivatives.columns
        source_positions = np.arange(len(source_rows), dtype=np.int64)

        blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        specifications = (
            (active_row, angle_column, 0),
            (active_row, magnitude_column, 1),
            (reactive_row, angle_column, 2),
            (reactive_row, magnitude_column, 3),
        )
        for row_map, column_map, code in specifications:
            output_rows = row_map[source_rows]
            output_columns = column_map[source_columns]
            keep = (output_rows >= 0) & (output_columns >= 0)
            blocks.append(
                (
                    output_rows[keep],
                    output_columns[keep],
                    source_positions[keep],
                    code,
                )
            )

        if blocks:
            rows = np.concatenate([block[0] for block in blocks])
            columns = np.concatenate([block[1] for block in blocks])
            sources = np.concatenate([block[2] for block in blocks])
            codes = np.concatenate(
                [np.full(len(block[0]), block[3], dtype=np.int8) for block in blocks]
            )
        else:
            rows = np.empty(0, dtype=np.int64)
            columns = np.empty(0, dtype=np.int64)
            sources = np.empty(0, dtype=np.int64)
            codes = np.empty(0, dtype=np.int8)

        order = np.lexsort((columns, rows))
        rows = rows[order]
        columns = columns[order]
        sources = sources[order]
        codes = codes[order]

        counts = np.bincount(rows, minlength=state_size)
        indptr = np.empty(state_size + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        matrix = csr_matrix(
            (
                np.zeros(len(rows), dtype=float),
                columns.astype(np.int32, copy=False),
                indptr,
            ),
            shape=(state_size, state_size),
            copy=False,
        )

        def positions(code: int) -> tuple[np.ndarray, np.ndarray]:
            output = np.flatnonzero(codes == code)
            return output, sources[output]

        va_real_output, va_real_source = positions(0)
        vm_real_output, vm_real_source = positions(1)
        va_imag_output, va_imag_source = positions(2)
        vm_imag_output, vm_imag_source = positions(3)

        return cls(
            derivatives=derivatives,
            matrix=matrix,
            va_real_output=va_real_output,
            va_real_source=va_real_source,
            vm_real_output=vm_real_output,
            vm_real_source=vm_real_source,
            va_imag_output=va_imag_output,
            va_imag_source=va_imag_source,
            vm_imag_output=vm_imag_output,
            vm_imag_source=vm_imag_source,
        )

    def jacobian(self, voltage: np.ndarray) -> csr_matrix:
        d_vm_data, d_va_data = self.derivatives.derivative_data(voltage)
        data = self.matrix.data
        data[self.va_real_output] = d_va_data[self.va_real_source].real
        data[self.vm_real_output] = d_vm_data[self.vm_real_source].real
        data[self.va_imag_output] = d_va_data[self.va_imag_source].imag
        data[self.vm_imag_output] = d_vm_data[self.vm_imag_source].imag
        return self.matrix


def newton_power_flow(
    ybus: Any,
    sbus: np.ndarray,
    initial_voltage: np.ndarray,
    ref: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    options: dict[str, Any],
    *,
    derivative_workspace: PowerDerivativeWorkspace | None = None,
) -> tuple[np.ndarray, bool, int]:
    """Solve Newton AC power flow using stock PYPOWER equations."""

    workspace = derivative_workspace or PowerDerivativeWorkspace.from_ybus(ybus)
    matrix = workspace.ybus
    jacobian_workspace = JacobianWorkspace.from_derivatives(workspace, pv, pq)

    tolerance = float(options["PF_TOL"])
    max_iterations = int(options["PF_MAX_IT"])
    linear_solver = options["PF_LIN_SOLVER_NR"]

    voltage = np.asarray(initial_voltage, dtype=np.complex128).copy()
    angle = np.angle(voltage)
    magnitude = np.abs(voltage)

    npv = len(pv)
    npq = len(pq)
    j2 = npv
    j3 = j2
    j4 = j2 + npq
    j5 = j4
    j6 = j4 + npq

    mismatch = voltage * np.conj(matrix @ voltage) - sbus
    residual = np.r_[
        mismatch[pv].real,
        mismatch[pq].real,
        mismatch[pq].imag,
    ]

    converged = bool(np.linalg.norm(residual, np.inf) < tolerance)
    iteration = 0

    while not converged and iteration < max_iterations:
        iteration += 1
        jacobian = jacobian_workspace.jacobian(voltage)

        step = -pplinsolve(jacobian, residual, linear_solver)
        if npv:
            angle[pv] += step[:j2]
        if npq:
            angle[pq] += step[j3:j4]
            magnitude[pq] += step[j5:j6]

        voltage = magnitude * np.exp(1j * angle)
        magnitude = np.abs(voltage)
        angle = np.angle(voltage)

        mismatch = voltage * np.conj(matrix @ voltage) - sbus
        residual = np.r_[
            mismatch[pv].real,
            mismatch[pq].real,
            mismatch[pq].imag,
        ]
        converged = bool(np.linalg.norm(residual, np.inf) < tolerance)

    return voltage, converged, iteration


_BUS_NETWORK_COLUMNS = (BUS_I, GS, BS)
_BRANCH_NETWORK_COLUMNS = (
    F_BUS,
    T_BUS,
    BR_R,
    BR_X,
    BR_B,
    TAP,
    SHIFT,
    BR_STATUS,
)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=float, copy=True)
    copied.flags.writeable = False
    return copied


def _network_bus_view(bus: np.ndarray) -> np.ndarray:
    return np.asarray(bus[:, _BUS_NETWORK_COLUMNS], dtype=float)


def _network_branch_view(branch: np.ndarray) -> np.ndarray:
    return np.asarray(branch[:, _BRANCH_NETWORK_COLUMNS], dtype=float)


@dataclass(frozen=True)
class PreparedACNetwork:
    """Admittance matrices valid for one unchanged AC network structure."""

    base_mva: float
    bus_network: np.ndarray
    branch_network: np.ndarray
    ybus: Any
    yf: Any
    yt: Any
    derivatives: PowerDerivativeWorkspace

    @classmethod
    def build(
        cls,
        *,
        base_mva: float,
        bus: np.ndarray,
        branch: np.ndarray,
    ) -> "PreparedACNetwork":
        ybus, yf, yt = makeYbus(float(base_mva), bus, branch)
        return cls(
            base_mva=float(base_mva),
            bus_network=_readonly_copy(_network_bus_view(bus)),
            branch_network=_readonly_copy(_network_branch_view(branch)),
            ybus=ybus,
            yf=yf,
            yt=yt,
            derivatives=PowerDerivativeWorkspace.from_ybus(ybus),
        )

    def require_matches(
        self,
        *,
        base_mva: float,
        bus: np.ndarray,
        branch: np.ndarray,
    ) -> None:
        if float(base_mva) != self.base_mva:
            raise ValueError("Prepared AC network baseMVA does not match the case.")
        if not np.array_equal(
            _network_bus_view(bus),
            self.bus_network,
            equal_nan=True,
        ):
            raise ValueError("Prepared AC network bus admittance data changed.")
        if not np.array_equal(
            _network_branch_view(branch),
            self.branch_network,
            equal_nan=True,
        ):
            raise ValueError("Prepared AC network branch data changed.")


def _prepare_internal_case(casedata: Any) -> dict[str, Any]:
    ppc = loadcase(deepcopy(casedata))
    if not isinstance(ppc, dict):
        raise ValueError("PYPOWER could not load the supplied case.")

    branch = np.asarray(ppc["branch"])
    if branch.shape[1] < QT:
        missing = QT - branch.shape[1] + 1
        ppc["branch"] = np.c_[
            branch,
            np.zeros((branch.shape[0], missing), dtype=float),
        ]

    return ext2int(ppc)


def solve_newton_power_flow(
    casedata: Any,
    options: dict[str, Any],
    *,
    prepared_network: PreparedACNetwork | None = None,
) -> tuple[dict[str, Any], bool, PreparedACNetwork]:
    """Run one stock-equivalent Newton AC solve with optional admittance reuse."""

    if bool(options["PF_DC"]) or int(options["PF_ALG"]) != 1:
        raise ValueError("Prepared AC network solve supports Newton AC power flow only.")

    started = perf_counter()
    ppc = _prepare_internal_case(casedata)
    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)

    ref, pv, pq = bustypes(bus, gen)
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    v0 = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])
    voltage_controlled = np.ones(v0.shape)
    voltage_controlled[pq] = 0
    controlled_generators = np.flatnonzero(voltage_controlled[gbus])
    if len(controlled_generators):
        controlled_buses = gbus[controlled_generators]
        v0[controlled_buses] = (
            gen[on[controlled_generators], VG]
            / np.abs(v0[controlled_buses])
            * v0[controlled_buses]
        )

    if prepared_network is None:
        prepared_network = PreparedACNetwork.build(
            base_mva=base_mva,
            bus=bus,
            branch=branch,
        )
    else:
        prepared_network.require_matches(
            base_mva=base_mva,
            bus=bus,
            branch=branch,
        )

    sbus = makeSbus(base_mva, bus, gen)
    voltage, success, _iterations = newton_power_flow(
        prepared_network.ybus,
        sbus,
        v0,
        ref,
        pv,
        pq,
        options,
        derivative_workspace=prepared_network.derivatives,
    )
    bus, gen, branch = pfsoln(
        base_mva,
        bus,
        gen,
        branch,
        prepared_network.ybus,
        prepared_network.yf,
        prepared_network.yt,
        voltage,
        ref,
        pv,
        pq,
    )

    ppc["bus"] = bus
    ppc["gen"] = gen
    ppc["branch"] = branch
    ppc["success"] = bool(success)
    ppc["et"] = perf_counter() - started

    results = int2ext(ppc)
    off_gen = results["order"]["gen"]["status"]["off"]
    if len(off_gen):
        results["gen"][np.ix_(off_gen, [PG, QG])] = 0

    off_branch = results["order"]["branch"]["status"]["off"]
    if len(off_branch):
        results["branch"][np.ix_(off_branch, [PF, QF, PT, QT])] = 0

    return results, bool(success), prepared_network
