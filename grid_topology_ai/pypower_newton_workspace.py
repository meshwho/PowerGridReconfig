from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

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

    def derivatives(self, voltage: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
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

    tolerance = float(options["PF_TOL"])
    max_iterations = int(options["PF_MAX_IT"])
    linear_solver = options["PF_LIN_SOLVER_NR"]

    voltage = np.asarray(initial_voltage, dtype=np.complex128).copy()
    angle = np.angle(voltage)
    magnitude = np.abs(voltage)

    pvpq = np.r_[pv, pq]
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
        d_vm, d_va = workspace.derivatives(voltage)

        j11 = d_va[np.array([pvpq]).T, pvpq].real
        j12 = d_vm[np.array([pvpq]).T, pq].real
        j21 = d_va[np.array([pq]).T, pvpq].imag
        j22 = d_vm[np.array([pq]).T, pq].imag
        jacobian = vstack(
            [hstack([j11, j12]), hstack([j21, j22])],
            format="csr",
        )

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
