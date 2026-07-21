from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from pypower.api import case14, case30, ppoption, runpf
from pypower.idx_brch import F_BUS, PF, PT, QF, QT, T_BUS
from pypower.idx_bus import BUS_I, VA, VM
from pypower.idx_gen import GEN_BUS, PG, QG

from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ac_power_flow"
CASE_FACTORIES: dict[str, Callable[[], dict]] = {
    "case14": case14,
    "case30": case30,
}


def _load_fixture(case_name: str) -> dict:
    path = FIXTURE_DIR / f"{case_name}_newton.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(
    actual: np.ndarray | float,
    expected: list[float] | float,
    *,
    atol: float,
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=atol,
    )


def _branch_row(
    branch: np.ndarray,
    *,
    from_bus: int,
    to_bus: int,
) -> np.ndarray:
    mask = (
        (branch[:, F_BUS].astype(int) == int(from_bus))
        & (branch[:, T_BUS].astype(int) == int(to_bus))
    )
    positions = np.flatnonzero(mask)
    assert positions.size == 1, (
        f"Expected one branch {from_bus}->{to_bus}, found "
        f"{positions.size}."
    )
    return branch[int(positions[0])]


@pytest.mark.parametrize("case_name", ["case14", "case30"])
def test_newton_ac_power_flow_matches_preverified_fixture(
    case_name: str,
) -> None:
    """Guard real AC solutions against silent numerical-contract drift."""

    fixture = _load_fixture(case_name)
    solver = fixture["solver"]
    tolerances = fixture["comparison_tolerances"]

    ppc = CASE_FACTORIES[case_name]()
    result, success = runpf(
        ppc,
        ppoption(
            PF_ALG=int(solver["pf_alg"]),
            PF_TOL=float(solver["tolerance"]),
            PF_MAX_IT=20,
            ENFORCE_Q_LIMS=(
                1 if bool(solver["enforce_q_limits"]) else 0
            ),
            VERBOSE=0,
            OUT_ALL=0,
        ),
    )

    assert bool(success) is True
    assert fixture["schema_version"] == 1
    assert fixture["case"] == case_name

    base_mva = float(result["baseMVA"])
    assert base_mva == pytest.approx(float(fixture["base_mva"]))

    expected_bus = fixture["buses"]
    np.testing.assert_array_equal(
        result["bus"][:, BUS_I].astype(int),
        np.asarray(expected_bus["id"], dtype=int),
    )
    _assert_close(
        result["bus"][:, VM],
        expected_bus["vm_pu"],
        atol=float(tolerances["voltage_magnitude_atol"]),
    )
    _assert_close(
        result["bus"][:, VA],
        expected_bus["va_degrees"],
        atol=float(tolerances["voltage_angle_degrees_atol"]),
    )

    expected_gen = fixture["generators"]
    np.testing.assert_array_equal(
        result["gen"][:, GEN_BUS].astype(int),
        np.asarray(expected_gen["bus_id"], dtype=int),
    )
    _assert_close(
        result["gen"][:, PG] / base_mva,
        expected_gen["pg_pu"],
        atol=float(tolerances["power_pu_atol"]),
    )
    _assert_close(
        result["gen"][:, QG] / base_mva,
        expected_gen["qg_pu"],
        atol=float(tolerances["power_pu_atol"]),
    )

    for anchor in fixture["branch_anchors"]:
        row = _branch_row(
            result["branch"],
            from_bus=int(anchor["from_bus"]),
            to_bus=int(anchor["to_bus"]),
        )
        _assert_close(
            np.asarray([row[PF], row[QF], row[PT], row[QT]])
            / base_mva,
            [
                anchor["pf_pu"],
                anchor["qf_pu"],
                anchor["pt_pu"],
                anchor["qt_pu"],
            ],
            atol=float(tolerances["power_pu_atol"]),
        )

    summary = fixture["summary"]
    _assert_close(
        float(np.min(result["bus"][:, VM])),
        float(summary["min_vm_pu"]),
        atol=float(tolerances["voltage_magnitude_atol"]),
    )
    _assert_close(
        float(np.max(result["bus"][:, VM])),
        float(summary["max_vm_pu"]),
        atol=float(tolerances["voltage_magnitude_atol"]),
    )
    _assert_close(
        float(np.sum(result["gen"][:, PG]) / base_mva),
        float(summary["total_pg_pu"]),
        atol=float(tolerances["power_pu_atol"]),
    )
    _assert_close(
        float(np.sum(result["gen"][:, QG]) / base_mva),
        float(summary["total_qg_pu"]),
        atol=float(tolerances["power_pu_atol"]),
    )

    metrics = calculate_physical_metrics_from_result(
        result,
        power_flow_converged=True,
    )
    assert metrics["power_flow_converged"] is True
    assert metrics["all_values_finite"] is True
    assert metrics["topology_connected"] is True
    assert metrics["num_unrated_active_branches"] == 0
    assert np.isfinite(float(metrics["max_loading_percent"]))
