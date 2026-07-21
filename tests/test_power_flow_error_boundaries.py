from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypower.idx_bus import VM

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.power_flow_errors import PowerFlowFailureKind
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def _adapter() -> SimpleNamespace:
    buses = []
    for bus_id, bus_type in ((10, "REF"), (20, "PQ")):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "load_scenario_idx": 0.0,
                "bus": bus_id,
                "Vm": 1.0,
                "Va": 0.0,
                "PQ": float(bus_type == "PQ"),
                "PV": 0.0,
                "REF": float(bus_type == "REF"),
                "vn_kv": 110.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
            }
        )
        buses.append(row)

    branch = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
    branch.update(
        {
            "scenario": 1,
            "load_scenario_idx": 0.0,
            "idx": 7,
            "from_bus": 10,
            "to_bus": 20,
            "r": 0.01,
            "x": 0.1,
            "b": 0.0,
            "rate_a": 100.0,
            "br_status": 1.0,
            "tap": 0.0,
            "shift": 0.0,
            "ang_min": -30.0,
            "ang_max": 30.0,
        }
    )

    generator = {
        "scenario": 1,
        "idx": 1,
        "bus": 10,
        "p_mw": 50.0,
        "q_mvar": 0.0,
        "min_p_mw": 0.0,
        "max_p_mw": 100.0,
        "min_q_mvar": -50.0,
        "max_q_mvar": 50.0,
        "in_service": 1.0,
    }

    return SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame([branch]),
        gen_df=pd.DataFrame([generator]),
    )


def _backend() -> GridFMPowerFlowBackend:
    return GridFMPowerFlowBackend(
        adapter=_adapter(),
        enable_cache=False,
        store_raw_result=True,
    )


def test_non_convergence_is_returned_as_typed_domain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_runpf(ppc: dict, options: dict) -> tuple[dict, bool]:
        del options
        return copy.deepcopy(ppc), False

    monkeypatch.setattr("grid_topology_ai.pypower_backend.runpf", fake_runpf)

    result = _backend().run_power_flow(scenario_id=1)

    assert result.success is False
    assert result.failure_kind is PowerFlowFailureKind.NOT_CONVERGED
    assert result.next_state is None
    assert result.raw_result is None
    assert "did not converge" in result.message.lower()


def test_invalid_solver_output_is_returned_as_typed_physical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_runpf(ppc: dict, options: dict) -> tuple[dict, bool]:
        del options
        result = copy.deepcopy(ppc)
        result["branch"] = np.pad(result["branch"], ((0, 0), (0, 4)))
        result["bus"][0, VM] = np.nan
        return result, True

    monkeypatch.setattr("grid_topology_ai.pypower_backend.runpf", fake_runpf)

    result = _backend().run_power_flow(scenario_id=1)

    assert result.success is False
    assert result.failure_kind is PowerFlowFailureKind.INVALID_PHYSICAL_STATE
    assert result.next_state is None
    assert result.raw_result is None
    assert "nan" in result.message.lower() or "finite" in result.message.lower()


def test_unexpected_programming_error_is_not_converted_to_domain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_runpf(ppc: dict, options: dict) -> tuple[dict, bool]:
        del ppc, options
        raise TypeError("unexpected programmer bug")

    monkeypatch.setattr("grid_topology_ai.pypower_backend.runpf", fake_runpf)

    with pytest.raises(TypeError, match="unexpected programmer bug"):
        _backend().run_power_flow(scenario_id=1)
