from __future__ import annotations

import io
import zlib
from pathlib import Path

import numpy as np

from grid_topology_ai import pf_cache_store, pf_warm_shadow_runtime
from grid_topology_ai.data_adapter import BUS_FEATURE_COLUMNS
from grid_topology_ai.pf_cache_store import PersistentPFCacheStore
from grid_topology_ai.pf_warm_shadow import WarmStartDescriptor, _pack_descriptor
from grid_topology_ai.pf_warm_shadow_runtime import (
    BoundedWarmStartStore,
    _compact_warm_state_payload,
)


def _descriptor() -> WarmStartDescriptor:
    return WarmStartDescriptor(
        pd=np.array([10.0, 20.0], dtype=np.float64),
        qd=np.array([4.0, 8.0], dtype=np.float64),
        pg=np.array([30.0], dtype=np.float64),
        qg=np.array([3.0], dtype=np.float64),
        gen_status=np.array([1.0], dtype=np.float64),
    )


def _exact_state_payload() -> bytes:
    features = np.zeros(
        (2, len(BUS_FEATURE_COLUMNS)),
        dtype=np.float64,
    )
    features[:, BUS_FEATURE_COLUMNS.index("Vm")] = [1.02, 0.99]
    features[:, BUS_FEATURE_COLUMNS.index("Va")] = [2.0, -4.0]

    buffer = io.BytesIO()
    np.savez(
        buffer,
        bus_features=features,
        has_bus_ids=np.asarray([1], dtype=np.uint8),
        bus_ids=np.array([1, 2], dtype=np.int64),
    )
    return buffer.getvalue()


def test_exact_budget_eviction_uses_insertion_order_when_timestamps_tie(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = bytes(range(256)) * 8
    packed_bytes = len(zlib.compress(payload, level=1))
    monkeypatch.setattr(pf_cache_store.time, "time", lambda: 1.0)

    old_key = "f" * 64
    new_key = "0" * 64
    with PersistentPFCacheStore(
        tmp_path,
        max_payload_bytes=packed_bytes + 16,
    ) as store:
        assert store.put(old_key, payload) is True
        assert store.put(new_key, payload) is True
        assert store.get(old_key) is None
        assert store.get(new_key) == payload


def test_warm_budget_eviction_uses_insertion_order_when_timestamps_tie(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    payload = _exact_state_payload()
    compact = _compact_warm_state_payload(payload)
    one_entry_bytes = len(compact) + len(
        zlib.compress(_pack_descriptor(descriptor), level=1)
    )
    monkeypatch.setattr(pf_warm_shadow_runtime.time, "time", lambda: 1.0)

    store = BoundedWarmStartStore(
        tmp_path,
        max_payload_bytes=one_entry_bytes + 16,
    )
    try:
        assert store.put(
            exact_key="f" * 64,
            topology_key="a" * 64,
            descriptor=descriptor,
            state_payload=payload,
        ) is True
        assert store.put(
            exact_key="0" * 64,
            topology_key="b" * 64,
            descriptor=descriptor,
            state_payload=payload,
        ) is True

        old = store.nearest(
            topology_key="a" * 64,
            descriptor=descriptor,
        )
        new = store.nearest(
            topology_key="b" * 64,
            descriptor=descriptor,
        )
    finally:
        store.close()

    assert old is None
    assert new is not None
    assert new.exact_key == "0" * 64


def test_warm_topology_cap_keeps_newest_insert_when_timestamps_tie(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    payload = _exact_state_payload()
    topology_key = "a" * 64
    monkeypatch.setattr(pf_warm_shadow_runtime.time, "time", lambda: 1.0)

    store = BoundedWarmStartStore(
        tmp_path,
        max_candidates_per_topology=1,
    )
    try:
        assert store.put(
            exact_key="f" * 64,
            topology_key=topology_key,
            descriptor=descriptor,
            state_payload=payload,
        ) is True
        assert store.put(
            exact_key="0" * 64,
            topology_key=topology_key,
            descriptor=descriptor,
            state_payload=payload,
        ) is True

        candidate = store.nearest(
            topology_key=topology_key,
            descriptor=descriptor,
        )
    finally:
        store.close()

    assert candidate is not None
    assert candidate.exact_key == "0" * 64
