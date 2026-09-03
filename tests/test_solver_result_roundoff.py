from __future__ import annotations

import copy

import numpy as np
import pytest
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    F_BUS,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import BUS_I, VM, VMAX, VMIN
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PMAX,
    PMIN,
    QMAX,
    QMIN,
)

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.physics.constraints import validate_pypower_result
from grid_topology_ai.power_flow import InvalidPhysicalState


def _ppc() -> dict[str, object]:
    bus = np.zeros((2, VMIN + 1), dtype=np.float64)
    bus[:, BUS_I] = [1.0, 2.0]
    bus[:, VM] = 1.0
    bus[:, VMIN] = 0.95
    bus[:, VMAX] = 1.05

    branch = np.zeros((1, ANGMAX + 1), dtype=np.float64)
    branch[0, F_BUS] = 1.0
    branch[0, T_BUS] = 2.0
    branch[0, RATE_A] = 100.0
    branch[0, BR_STATUS] = 1.0
    branch[0, ANGMIN] = -360.0
    branch[0, ANGMAX] = 360.0

    gen = np.zeros((1, 21), dtype=np.float64)
    gen[0, GEN_BUS] = 1.0
    gen[0, GEN_STATUS] = 1.0
    gen[0, PMIN] = 0.0
    gen[0, PMAX] = 100.0
    gen[0, QMIN] = -50.0
    gen[0, QMAX] = 50.0

    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus,
        "branch": branch,
        "gen": gen,
    }


def _result(ppc: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(ppc)
    branch = np.asarray(result["branch"], dtype=np.float64)
    result["branch"] = np.pad(
        branch,
        ((0, 0), (0, QT + 1 - branch.shape[1])),
    )
    return result


def test_result_contract_accepts_machine_roundoff_in_static_solver_data() -> None:
    ppc = _ppc()
    result = _result(ppc)

    np.asarray(result["bus"])[0, VMAX] = np.nextafter(
        np.asarray(result["bus"])[0, VMAX], np.inf
    )
    np.asarray(result["branch"])[0, RATE_A] = np.nextafter(
        np.asarray(result["branch"])[0, RATE_A], np.inf
    )
    np.asarray(result["gen"])[0, PMAX] = np.nextafter(
        np.asarray(result["gen"])[0, PMAX], np.inf
    )

    validate_pypower_result(
        result,
        PhysicsConfig(),
        input_ppc=ppc,
        context="roundoff result",
    )


@pytest.mark.parametrize(
    ("matrix_name", "column", "message"),
    [
        ("bus", VMAX, "immutable bus data"),
        ("branch", RATE_A, "immutable branch data"),
        ("gen", PMAX, "immutable generator data"),
    ],
)
def test_result_contract_rejects_material_static_solver_changes(
    matrix_name: str,
    column: int,
    message: str,
) -> None:
    ppc = _ppc()
    result = _result(ppc)
    np.asarray(result[matrix_name])[0, column] += 1e-8

    with pytest.raises(InvalidPhysicalState, match=message):
        validate_pypower_result(
            result,
            PhysicsConfig(),
            input_ppc=ppc,
            context="material change result",
        )
