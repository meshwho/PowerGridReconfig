from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pypower.idx_bus import BUS_I, PD, QD
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.pf_warm_shadow import (
    PersistentWarmStartStore,
    WarmCandidate,
    WarmStartDescriptor,
    WarmStartShadow,
    warm_start_descriptor,
    warm_start_distance,
)


def _ppc(
    *,
    bus_ids=(2, 1),
    pd=(20.0, 40.0),
    qd=(5.0, 10.0),
    pg=(30.0, 50.0),
    qg=(3.0, 7.0),
    status=(1.0, 1.0),
):
    bus = np.zeros((2, 13), dtype=np.float64)
    bus[:, BUS_I] = bus_ids
    bus[:, PD] = pd
    bus[:, QD] = qd

    gen = np.zeros((2, 21), dtype=np.float64)
    gen[:, PG] = pg
    gen[:, QG] = qg
    gen[:, GEN_STATUS] = status

    return {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": np.zeros((2, 13), dtype=np.float64),
    }


def _descriptor(value: float, *, status=(1.0, 1.0)) -> WarmStartDescriptor:
    return WarmStartDescriptor(
        pd=np.array([value, 2.0 * value]),
        qd=np.array([0.5 * value, value]),
        pg=np.array([1.2 * value, 1.8 * value]),
        qg=np.array([0.1 * value, 0.2 * value]),
        gen_status=np.asarray(status, dtype=np.float64),
    )


def test_warm_descriptor_is_row_order_invariant() -> None:
    left = warm_start_descriptor(
        _ppc(),
        generator_ids=np.array([20, 10], dtype=np.int64),
    )

    right_ppc = _ppc(
        bus_ids=(1, 2),
        pd=(40.0, 20.0),
        qd=(10.0, 5.0),
        pg=(50.0, 30.0),
        qg=(7.0, 3.0),
    )
    right = warm_start_descriptor(
        right_ppc,
        generator_ids=np.array([10, 20], dtype=np.int64),
    )

    np.testing.assert_array_equal(left.pd, right.pd)
    np.testing.assert_array_equal(left.qd, right.qd)
    np.testing.assert_array_equal(left.pg, right.pg)
    np.testing.assert_array_equal(left.qg, right.qg)
    np.testing.assert_array_equal(left.gen_status, right.gen_status)


def test_warm_distance_rejects_generator_status_change() -> None:
    distance = warm_start_distance(
        _descriptor(10.0),
        _descriptor(10.0, status=(1.0, 0.0)),
    )

    assert np.isinf(distance)


def test_store_selects_nearest_cross_scenario_candidate(tmp_path) -> None:
    topology_key = "a" * 64
    request_key = "f" * 64

    with_store = PersistentWarmStartStore(
        tmp_path,
        max_candidates_per_topology=4,
    )
    try:
        with_store.put(
            exact_key="1" * 64,
            topology_key=topology_key,
            descriptor=_descriptor(10.0),
            state_payload=b"far",
        )
        with_store.put(
            exact_key="2" * 64,
            topology_key=topology_key,
            descriptor=_descriptor(19.0),
            state_payload=b"near",
        )

        candidate = with_store.nearest(
            topology_key=topology_key,
            descriptor=_descriptor(20.0),
            exclude_exact_key=request_key,
        )
    finally:
        with_store.close()

    assert candidate is not None
    assert candidate.exact_key == "2" * 64
    assert candidate.state_payload == b"near"
    assert candidate.distance < 0.1


def test_store_sees_entries_written_after_another_connection_opened(tmp_path) -> None:
    topology_key = "a" * 64
    writer = PersistentWarmStartStore(tmp_path)
    reader = PersistentWarmStartStore(tmp_path)

    try:
        writer.put(
            exact_key="3" * 64,
            topology_key=topology_key,
            descriptor=_descriptor(15.0),
            state_payload=b"shared",
        )

        candidate = reader.nearest(
            topology_key=topology_key,
            descriptor=_descriptor(15.5),
        )
    finally:
        reader.close()
        writer.close()

    assert candidate is not None
    assert candidate.exact_key == "3" * 64
    assert candidate.state_payload == b"shared"


def test_store_keeps_bounded_candidates_per_topology(tmp_path) -> None:
    topology_key = "b" * 64
    store = PersistentWarmStartStore(
        tmp_path,
        max_candidates_per_topology=2,
    )

    try:
        for index, key in enumerate(("1" * 64, "2" * 64, "3" * 64), start=1):
            store.put(
                exact_key=key,
                topology_key=topology_key,
                descriptor=_descriptor(float(index)),
                state_payload=str(index).encode(),
            )

        candidates, records = store.counts()
    finally:
        store.close()

    assert candidates == 2
    assert records == 0


def test_store_excludes_same_exact_problem_from_warm_lookup(tmp_path) -> None:
    topology_key = "c" * 64
    exact_key = "4" * 64
    store = PersistentWarmStartStore(tmp_path)

    try:
        store.put(
            exact_key=exact_key,
            topology_key=topology_key,
            descriptor=_descriptor(10.0),
            state_payload=b"same",
        )
        candidate = store.nearest(
            topology_key=topology_key,
            descriptor=_descriptor(10.0),
            exclude_exact_key=exact_key,
        )
    finally:
        store.close()

    assert candidate is None


def test_shadow_never_replaces_authoritative_result(monkeypatch) -> None:
    authoritative_state = SimpleNamespace()
    authoritative_result = SimpleNamespace(
        success=True,
        next_state=authoritative_state,
        message="Power flow converged.",
    )

    class FakeBackend:
        def run_power_flow_from_state(self, state, switched_off_branch_id=None, *, action=None):
            return authoritative_result

        @staticmethod
        def _serialize_exact_state(state):
            return b"authoritative"

    class FakeStore:
        def __init__(self):
            self.put_calls = []
            self.records = []

        def nearest(self, **kwargs):
            return WarmCandidate(
                exact_key="1" * 64,
                topology_key="a" * 64,
                distance=0.05,
                state_payload=b"candidate",
            )

        def put(self, **kwargs):
            self.put_calls.append(kwargs)

        def record_shadow(self, **kwargs):
            self.records.append(kwargs)

        def close(self):
            return None

    backend = FakeBackend()
    store = FakeStore()
    shadow = WarmStartShadow(backend, store, sample_rate=1.0)
    shadow.install()

    prepared = (
        {"bus": np.zeros((0, 0))},
        {},
        "f" * 64,
        "a" * 64,
        _descriptor(10.0),
    )
    monkeypatch.setattr(shadow, "_prepare", lambda *args: prepared)
    monkeypatch.setattr(
        shadow,
        "_shadow_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("different basin")
        ),
    )

    request_state = SimpleNamespace(scenario_id=99)
    result = backend.run_power_flow_from_state(request_state)

    assert result is authoritative_result
    assert store.put_calls[0]["state_payload"] == b"authoritative"
    assert store.records[0]["record"]["shadow_success"] is False
    assert store.records[0]["scenario_id"] == 99


def test_shadow_sampling_is_deterministic() -> None:
    backend = SimpleNamespace()
    store = SimpleNamespace(close=lambda: None)
    shadow = WarmStartShadow(backend, store, sample_rate=0.5)

    first = shadow._sample("0" * 64)
    second = shadow._sample("0" * 64)
    high = shadow._sample("f" * 64)

    assert first is True
    assert second is first
    assert high is False
