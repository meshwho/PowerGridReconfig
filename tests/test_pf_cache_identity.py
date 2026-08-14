from __future__ import annotations

import copy

import numpy as np
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)

from grid_topology_ai.pf_cache_identity import (
    exact_pf_problem_fingerprint,
    network_fingerprint,
    topology_fingerprint,
)


def _ppc() -> dict[str, object]:
    bus = np.zeros((2, VMIN + 1), dtype=np.float64)
    bus[:, BUS_I] = [1, 2]
    bus[:, BUS_TYPE] = [3, 1]
    bus[:, PD] = [0.0, 70.0]
    bus[:, QD] = [0.0, 24.0]
    bus[:, GS] = [0.0, 0.0]
    bus[:, BS] = [0.0, 0.0]
    bus[:, VM] = [1.02, 0.99]
    bus[:, VA] = [0.0, -2.0]
    bus[:, BASE_KV] = [230.0, 230.0]
    bus[:, ZONE] = [1.0, 1.0]
    bus[:, VMAX] = [1.10, 1.10]
    bus[:, VMIN] = [0.90, 0.90]

    gen = np.zeros((2, PMIN + 1), dtype=np.float64)
    gen[:, GEN_BUS] = [1, 2]
    gen[:, PG] = [80.0, 15.0]
    gen[:, QG] = [12.0, 3.0]
    gen[:, QMAX] = [50.0, 30.0]
    gen[:, QMIN] = [-50.0, -30.0]
    gen[:, VG] = [1.02, 1.01]
    gen[:, MBASE] = [100.0, 100.0]
    gen[:, GEN_STATUS] = [1.0, 1.0]
    gen[:, PMAX] = [120.0, 40.0]
    gen[:, PMIN] = [0.0, 0.0]

    branch = np.zeros((2, ANGMAX + 1), dtype=np.float64)
    branch[:, F_BUS] = [1, 2]
    branch[:, T_BUS] = [2, 1]
    branch[:, BR_R] = [0.01, 0.02]
    branch[:, BR_X] = [0.10, 0.20]
    branch[:, BR_B] = [0.02, 0.01]
    branch[:, RATE_A] = [100.0, 90.0]
    branch[:, RATE_B] = [100.0, 90.0]
    branch[:, RATE_C] = [100.0, 90.0]
    branch[:, TAP] = [0.0, 0.0]
    branch[:, SHIFT] = [0.0, 0.0]
    branch[:, BR_STATUS] = [1.0, 1.0]
    branch[:, ANGMIN] = [-360.0, -360.0]
    branch[:, ANGMAX] = [360.0, 360.0]

    return {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }


def _fingerprints(ppc: dict[str, object]) -> tuple[str, str, str]:
    branch_ids = np.array([10, 20], dtype=np.int64)
    generator_ids = np.array([100, 200], dtype=np.int64)
    return (
        network_fingerprint(ppc, branch_ids=branch_ids),
        topology_fingerprint(ppc, branch_ids=branch_ids),
        exact_pf_problem_fingerprint(
            ppc,
            physics_fingerprint="physics-v1",
            branch_ids=branch_ids,
            generator_ids=generator_ids,
        ),
    )


def test_exact_identity_changes_with_injections_but_topology_does_not() -> None:
    base = _ppc()
    changed = copy.deepcopy(base)
    changed["bus"][1, PD] += 1.0  # type: ignore[index]

    base_network, base_topology, base_exact = _fingerprints(base)
    changed_network, changed_topology, changed_exact = _fingerprints(changed)

    assert changed_network == base_network
    assert changed_topology == base_topology
    assert changed_exact != base_exact


def test_topology_identity_changes_with_branch_status() -> None:
    base = _ppc()
    changed = copy.deepcopy(base)
    changed["branch"][1, BR_STATUS] = 0.0  # type: ignore[index]

    base_network, base_topology, base_exact = _fingerprints(base)
    changed_network, changed_topology, changed_exact = _fingerprints(changed)

    assert changed_network == base_network
    assert changed_topology != base_topology
    assert changed_exact != base_exact


def test_exact_identity_changes_with_generator_operating_point() -> None:
    base = _ppc()
    changed = copy.deepcopy(base)
    changed["gen"][0, PG] += 0.1  # type: ignore[index]

    assert _fingerprints(changed)[0:2] == _fingerprints(base)[0:2]
    assert _fingerprints(changed)[2] != _fingerprints(base)[2]


def test_exact_identity_ignores_voltage_initial_guess() -> None:
    base = _ppc()
    changed = copy.deepcopy(base)
    changed["bus"][:, VM] = [0.95, 1.05]  # type: ignore[index]
    changed["bus"][:, VA] = [4.0, -8.0]  # type: ignore[index]

    assert _fingerprints(changed) == _fingerprints(base)


def test_network_change_invalidates_all_cache_identities() -> None:
    base = _ppc()
    changed = copy.deepcopy(base)
    changed["branch"][0, BR_X] += 0.001  # type: ignore[index]

    base_values = _fingerprints(base)
    changed_values = _fingerprints(changed)

    assert changed_values[0] != base_values[0]
    assert changed_values[1] != base_values[1]
    assert changed_values[2] != base_values[2]


def test_different_physics_contract_invalidates_exact_identity() -> None:
    ppc = _ppc()
    branch_ids = np.array([10, 20], dtype=np.int64)
    generator_ids = np.array([100, 200], dtype=np.int64)

    first = exact_pf_problem_fingerprint(
        ppc,
        physics_fingerprint="physics-a",
        branch_ids=branch_ids,
        generator_ids=generator_ids,
    )
    second = exact_pf_problem_fingerprint(
        ppc,
        physics_fingerprint="physics-b",
        branch_ids=branch_ids,
        generator_ids=generator_ids,
    )

    assert first != second


def test_row_order_does_not_change_cache_identities() -> None:
    base = _ppc()
    reordered = copy.deepcopy(base)
    reordered["bus"] = reordered["bus"][[1, 0]]  # type: ignore[index]
    reordered["gen"] = reordered["gen"][[1, 0]]  # type: ignore[index]
    reordered["branch"] = reordered["branch"][[1, 0]]  # type: ignore[index]

    base_branch_ids = np.array([10, 20], dtype=np.int64)
    base_generator_ids = np.array([100, 200], dtype=np.int64)
    reordered_branch_ids = base_branch_ids[[1, 0]]
    reordered_generator_ids = base_generator_ids[[1, 0]]

    assert network_fingerprint(
        base,
        branch_ids=base_branch_ids,
    ) == network_fingerprint(
        reordered,
        branch_ids=reordered_branch_ids,
    )
    assert topology_fingerprint(
        base,
        branch_ids=base_branch_ids,
    ) == topology_fingerprint(
        reordered,
        branch_ids=reordered_branch_ids,
    )
    assert exact_pf_problem_fingerprint(
        base,
        physics_fingerprint="physics-v1",
        branch_ids=base_branch_ids,
        generator_ids=base_generator_ids,
    ) == exact_pf_problem_fingerprint(
        reordered,
        physics_fingerprint="physics-v1",
        branch_ids=reordered_branch_ids,
        generator_ids=reordered_generator_ids,
    )
