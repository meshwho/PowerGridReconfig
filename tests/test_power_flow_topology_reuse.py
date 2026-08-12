from __future__ import annotations

import numpy as np
from pypower.idx_bus import BUS_I, VA, VM

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.pypower_backend import (
    GridFMPowerFlowBackend,
    _GeneratorOperatingPointState,
)
from grid_topology_ai.topology_actions import GridFMAction


def _state(
    *,
    pg: tuple[float, float] = (70.0, 20.0),
    qg: tuple[float, float] = (8.0, 3.0),
    generator_status: tuple[float, float] = (1.0, 1.0),
    branch_status: tuple[float, float] = (1.0, 1.0),
    vm: tuple[float, float] = (1.01, 0.99),
    va: tuple[float, float] = (0.0, -2.0),
) -> _GeneratorOperatingPointState:
    bus_features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    bus_features[:, BUS_FEATURE_COLUMNS.index("Pd")] = [45.0, 25.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Qd")] = [12.0, 7.0]
    bus_features[:, BUS_FEATURE_COLUMNS.index("Vm")] = vm
    bus_features[:, BUS_FEATURE_COLUMNS.index("Va")] = va
    bus_features[:, BUS_FEATURE_COLUMNS.index("min_vm_pu")] = 0.95
    bus_features[:, BUS_FEATURE_COLUMNS.index("max_vm_pu")] = 1.05

    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("r")] = [0.01, 0.02]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("x")] = [0.1, 0.2]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("rate_a")] = [100.0, 100.0]
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = branch_status

    return _GeneratorOperatingPointState(
        scenario_id=7,
        load_scenario_idx=2.0,
        bus_features=bus_features,
        branch_features=branch_features,
        edge_index=np.array(
            [[0, 1], [1, 0]],
            dtype=np.int64,
        ),
        branch_ids=np.array([10, 20], dtype=np.int64),
        branch_status=np.asarray(branch_status, dtype=np.float32),
        metrics={},
        outaged_branch_ids=[
            branch_id
            for branch_id, active in zip((10, 20), branch_status)
            if active <= 0.0
        ],
        bus_ids=np.array([100, 200], dtype=np.int64),
        generator_ids=np.array([0, 1], dtype=np.int64),
        generator_p_mw=np.asarray(pg, dtype=np.float64),
        generator_q_mvar=np.asarray(qg, dtype=np.float64),
        generator_status=np.asarray(generator_status, dtype=np.float64),
    )


def _switch_off(branch_id: int, branch_pos: int) -> GridFMAction:
    return GridFMAction(
        action_id=1 + branch_pos,
        action_type="switch_off_branch",
        branch_id=branch_id,
        branch_pos=branch_pos,
    )


def _backend() -> GridFMPowerFlowBackend:
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        enable_cache=True,
    )
    backend._require_usable_next_state = lambda state: None  # type: ignore[method-assign]
    return backend


def test_exact_cache_hit_has_priority_over_tolerant_reuse() -> None:
    backend = _backend()
    state = _state()
    action = _switch_off(20, 1)
    exact_next = _state(branch_status=(1.0, 0.0), vm=(1.02, 0.98))

    exact_key = backend._make_topology_cache_key_from_state(
        state,
        action=action,
    )
    backend._cache[exact_key] = exact_next

    bucket_key = backend._topology_bucket_key(state, action=action)
    backend._remember_topology_result(
        bucket_key,
        source_state=_state(pg=(70.0 + 0.5e-6, 20.0), qg=(8.0, 3.0)),
        next_state=_state(branch_status=(1.0, 0.0), vm=(0.97, 1.03)),
    )

    result = backend.run_power_flow_from_state(state, action=action)

    assert result.next_state is exact_next
    assert result.message == "Power flow converged. [cache hit]"
    assert backend.exact_cache_hits == 1
    assert backend.tolerant_cache_hits == 0
    assert backend.cache_misses == 0


def test_generator_differences_within_tolerance_reuse_cached_result() -> None:
    backend = _backend()
    action = _switch_off(20, 1)
    cached_source = _state()
    cached_next = _state(branch_status=(1.0, 0.0), vm=(1.02, 0.98))
    bucket_key = backend._topology_bucket_key(cached_source, action=action)
    backend._remember_topology_result(
        bucket_key,
        source_state=cached_source,
        next_state=cached_next,
    )

    current = _state(
        pg=(70.0 + 0.5e-6, 20.0 - 0.5e-6),
        qg=(8.0 - 0.5e-6, 3.0 + 0.5e-6),
    )
    result = backend.run_power_flow_from_state(current, action=action)

    assert result.next_state is cached_next
    assert result.message == "Power flow converged. [tolerant cache hit]"
    assert backend.tolerant_cache_hits == 1
    assert backend.cache_hits == 1
    assert backend.cache_misses == 0


def test_generator_difference_above_tolerance_uses_voltage_warm_start() -> None:
    backend = _backend()
    action = _switch_off(20, 1)
    cached_source = _state()
    cached_next = _state(
        branch_status=(1.0, 0.0),
        vm=(1.035, 0.965),
        va=(1.5, -4.0),
    )
    bucket_key = backend._topology_bucket_key(cached_source, action=action)
    backend._remember_topology_result(
        bucket_key,
        source_state=cached_source,
        next_state=cached_next,
    )

    current = _state(pg=(70.01, 19.99))
    tolerant, warm = backend._select_topology_entry(bucket_key, current)

    assert tolerant is None
    assert warm is not None
    assert warm.next_state is cached_next

    ppc = {"bus": np.zeros((2, 13), dtype=np.float64)}
    ppc["bus"][:, BUS_I] = [100, 200]
    ppc["bus"][:, VM] = 1.0
    backend._pending_warm_start_state = warm.next_state
    backend._pending_warm_start_applied = False
    backend._apply_pending_warm_start(ppc)

    np.testing.assert_allclose(ppc["bus"][:, VM], [1.035, 0.965])
    np.testing.assert_allclose(ppc["bus"][:, VA], [1.5, -4.0])
    assert backend._pending_warm_start_applied


def test_generator_status_difference_is_never_a_tolerant_hit() -> None:
    backend = _backend()
    action = _switch_off(20, 1)
    cached_source = _state()
    cached_next = _state(branch_status=(1.0, 0.0))
    bucket_key = backend._topology_bucket_key(cached_source, action=action)
    backend._remember_topology_result(
        bucket_key,
        source_state=cached_source,
        next_state=cached_next,
    )

    current = _state(generator_status=(1.0, 0.0))
    tolerant, warm = backend._select_topology_entry(bucket_key, current)

    assert tolerant is None
    assert warm is not None
