from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypower.idx_brch import BR_STATUS
from pypower.idx_gen import GEN_STATUS, PG, QG, VG

from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, BUS_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.power_flow.problem import (
    GeneratorOperatingPoint,
    build_power_flow_problem_from_state,
    build_scenario_power_flow_template,
)
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


def _adapter() -> SimpleNamespace:
    buses = []
    for bus_id, bus_type in ((10, "REF"), (20, "PV"), (30, "PQ")):
        row = {name: 0.0 for name in BUS_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "bus": bus_id,
                "Pd": 15.0 if bus_id == 30 else 0.0,
                "Qd": 4.0 if bus_id == 30 else 0.0,
                "Vm": 1.01 if bus_id == 10 else 1.0,
                "Va": 0.0,
                "vn_kv": 110.0,
                "min_vm_pu": 0.95,
                "max_vm_pu": 1.05,
                bus_type: 1.0,
            }
        )
        buses.append(row)

    branches = []
    for idx, from_bus, to_bus, rate_a in (
        (7, 10, 20, 100.0),
        (9, 20, 30, 80.0),
    ):
        row = {name: 0.0 for name in BRANCH_FEATURE_COLUMNS}
        row.update(
            {
                "scenario": 1,
                "idx": idx,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "r": 0.01,
                "x": 0.1,
                "b": 0.02,
                "tap": 0.0,
                "shift": 0.0,
                "rate_a": rate_a,
                "br_status": 1.0,
                "ang_min": -30.0,
                "ang_max": 30.0,
            }
        )
        branches.append(row)

    generators = [
        {
            "scenario": 1,
            "idx": 0,
            "bus": 10,
            "p_mw": 40.0,
            "q_mvar": 5.0,
            "min_p_mw": 0.0,
            "max_p_mw": 100.0,
            "min_q_mvar": -50.0,
            "max_q_mvar": 50.0,
            "in_service": 1.0,
        },
        {
            "scenario": 1,
            "idx": 1,
            "bus": 20,
            "p_mw": 25.0,
            "q_mvar": 8.0,
            "min_p_mw": 0.0,
            "max_p_mw": 60.0,
            "min_q_mvar": -20.0,
            "max_q_mvar": 40.0,
            "in_service": 1.0,
        },
    ]
    return SimpleNamespace(
        bus_df=pd.DataFrame(buses),
        branch_df=pd.DataFrame(branches),
        gen_df=pd.DataFrame(generators),
    )


def _state(adapter: SimpleNamespace) -> GridFMState:
    bus = adapter.bus_df.sort_values("bus").reset_index(drop=True)
    branch = adapter.branch_df.sort_values("idx").reset_index(drop=True)
    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=bus[BUS_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        branch_features=branch[BRANCH_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        branch_ids=branch["idx"].to_numpy(dtype=np.int64),
        branch_status=branch["br_status"].to_numpy(dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=bus["bus"].to_numpy(dtype=np.int64),
    )


def _template(adapter: SimpleNamespace, backend: GridFMPowerFlowBackend):
    return build_scenario_power_flow_template(
        scenario_id=1,
        bus_df=adapter.bus_df,
        branch_df=adapter.branch_df,
        gen_df=adapter.gen_df,
        base_mva=backend.base_mva,
    )


def test_numpy_problem_matches_existing_dataframe_builder() -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)

    reference, _frames = backend._build_ppc_from_state(
        state,
        switched_off_branch_id=9,
    )
    problem = build_power_flow_problem_from_state(
        template=_template(adapter, backend),
        state=state,
        branch_id=9,
        target_status=0,
    ).to_ppc()

    assert problem["version"] == reference["version"]
    assert problem["baseMVA"] == reference["baseMVA"]
    np.testing.assert_array_equal(problem["bus"], reference["bus"])
    np.testing.assert_array_equal(problem["branch"], reference["branch"])
    np.testing.assert_array_equal(problem["gen"], reference["gen"])


def test_numpy_problem_carries_exact_generator_operating_point() -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)
    template = _template(adapter, backend)
    original_vg = template.gen[:, VG].copy()

    operating_point = GeneratorOperatingPoint(
        generator_ids=np.asarray([0, 1], dtype=np.int64),
        p_mw=np.asarray([42.5, 22.5], dtype=np.float64),
        q_mvar=np.asarray([7.0, 11.0], dtype=np.float64),
        status=np.asarray([1.0, 0.0], dtype=np.float64),
    )
    problem = build_power_flow_problem_from_state(
        template=template,
        state=state,
        generator_operating_point=operating_point,
    )

    np.testing.assert_array_equal(problem.gen[:, PG], operating_point.p_mw)
    np.testing.assert_array_equal(problem.gen[:, QG], operating_point.q_mvar)
    np.testing.assert_array_equal(problem.gen[:, GEN_STATUS], operating_point.status)
    # Solved parent Vm must not replace the original generator voltage setpoint.
    np.testing.assert_array_equal(problem.gen[:, VG], original_vg)


def test_numpy_problem_transition_does_not_use_pandas_after_template_build(
    monkeypatch,
) -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)
    template = _template(adapter, backend)

    def fail(*_args, **_kwargs):
        raise AssertionError("pandas entered the repeated PF hot path")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail)
    problem = build_power_flow_problem_from_state(
        template=template,
        state=state,
        branch_id=9,
        target_status=0,
    )

    assert problem.branch[1, BR_STATUS] == 0.0


def test_numpy_problem_does_not_mutate_the_scenario_template() -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)
    template = _template(adapter, backend)

    problem = build_power_flow_problem_from_state(
        template=template,
        state=state,
        branch_id=9,
        target_status=0,
    )

    assert not np.shares_memory(problem.bus, template.bus)
    assert not np.shares_memory(problem.branch, template.branch)
    assert not np.shares_memory(problem.gen, template.gen)
    assert template.branch[1, BR_STATUS] == 1.0


def test_numpy_problem_can_close_an_outaged_branch() -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)
    branch_features = state.branch_features.copy()
    status_column = BRANCH_FEATURE_COLUMNS.index("br_status")
    branch_features[1, status_column] = 0.0
    outaged = replace(
        state,
        branch_features=branch_features,
        branch_status=np.asarray([1.0, 0.0], dtype=np.float32),
        outaged_branch_ids=[9],
    )

    problem = build_power_flow_problem_from_state(
        template=_template(adapter, backend),
        state=outaged,
        branch_id=9,
        target_status=1,
    )
    assert problem.branch[1, BR_STATUS] == 1.0


def test_numpy_problem_rejects_wrong_physical_row_identity() -> None:
    adapter = _adapter()
    backend = GridFMPowerFlowBackend(adapter=adapter, enable_cache=False)
    state = _state(adapter)
    mismatched = replace(
        state,
        branch_ids=np.asarray([7, 99], dtype=np.int64),
    )

    with pytest.raises(Exception, match="Branch IDs do not match"):
        build_power_flow_problem_from_state(
            template=_template(adapter, backend),
            state=mismatched,
        )
