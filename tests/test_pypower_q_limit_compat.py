from __future__ import annotations

import copy

import numpy as np
from pypower.api import case9, ppoption
from pypower.idx_gen import GEN_BUS, QMAX, QMIN

from grid_topology_ai.pypower_compat import (
    _IntegralGenBusMatrix,
    _prepare_case,
    runpf,
)


def test_generator_bus_column_stays_integral_after_copy_and_reorder() -> None:
    matrix = _IntegralGenBusMatrix(
        [
            [1.0, 12.5],
            [3.0, -4.0],
            [2.0, 7.25],
        ]
    )

    copied = copy.deepcopy(matrix)
    reordered = copied[[2, 0, 1], :]

    assert reordered[:, GEN_BUS].dtype.kind in {"i", "u"}
    assert reordered[:, GEN_BUS].tolist() == [2, 1, 3]
    assert isinstance(reordered[0, GEN_BUS], np.integer)


def test_prepare_case_only_wraps_q_limit_solver_input() -> None:
    ppc = case9()

    disabled = _prepare_case(ppc, {"ENFORCE_Q_LIMS": 0})
    enabled = _prepare_case(ppc, {"ENFORCE_Q_LIMS": 1})

    assert disabled is ppc
    assert enabled is not ppc
    assert isinstance(enabled["gen"], _IntegralGenBusMatrix)
    assert isinstance(ppc["gen"], np.ndarray)
    assert not isinstance(ppc["gen"], _IntegralGenBusMatrix)


def test_runpf_handles_enforced_q_limit_with_float_generator_matrix() -> None:
    ppc = case9()

    # The second generator normally supplies positive Q in case9. Tightening
    # only its upper limit forces PYPOWER through the ENFORCE_Q_LIMS branch
    # that indexes buses through the floating-point GEN_BUS column.
    ppc["gen"][1, QMIN] = -300.0
    ppc["gen"][1, QMAX] = 0.0

    result, success = runpf(
        ppc,
        ppoption(
            PF_ALG=1,
            VERBOSE=0,
            OUT_ALL=0,
            ENFORCE_Q_LIMS=1,
        ),
    )

    assert bool(success)
    assert type(result["gen"]) is np.ndarray
    assert np.isfinite(result["bus"]).all()
    assert np.isfinite(result["branch"]).all()
    assert np.isfinite(result["gen"]).all()
