from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from grid_topology_ai.cache.byte_lru import ByteLRUCache


DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES = 8 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128


def _hash_array(digest: Any, values: np.ndarray, *, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def structural_topology_fingerprint(
    state: Any,
    *,
    require_connected_after_switch: bool,
    closeable_branch_ids: tuple[int, ...],
) -> bytes:
    """Return the exact identity of topology-only action validity."""

    branch_ids = np.asarray(state.branch_ids, dtype=np.int64)
    branch_status = np.asarray(state.branch_status, dtype=np.float64)
    edge_index = np.asarray(state.edge_index, dtype=np.int64)

    if branch_ids.ndim != 1 or branch_status.ndim != 1:
        raise ValueError("branch_ids and branch_status must be one-dimensional.")
    if len(branch_ids) != len(branch_status):
        raise ValueError("branch_ids and branch_status length mismatch.")
    if edge_index.shape != (2, len(branch_ids)):
        raise ValueError("edge_index must have shape [2, num_branches].")

    active = np.asarray(branch_status > 0.0, dtype=np.uint8)
    num_buses = int(np.asarray(state.bus_features).shape[0])

    digest = hashlib.sha256()
    digest.update(b"structural-topology-v1\0")
    digest.update(np.asarray([num_buses], dtype=np.int64).tobytes())
    _hash_array(digest, branch_ids, dtype=np.int64)
    _hash_array(digest, edge_index, dtype=np.int64)
    _hash_array(digest, active, dtype=np.uint8)
    digest.update(b"\x01" if require_connected_after_switch else b"\x00")
    _hash_array(
        digest,
        np.asarray(closeable_branch_ids, dtype=np.int64),
        dtype=np.int64,
    )
    return digest.digest()


@dataclass(frozen=True, slots=True)
class _PackedMask:
    values: np.ndarray
    length: int

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "_PackedMask":
        values = np.asarray(mask, dtype=bool)
        if values.ndim != 1:
            raise ValueError("Structural action mask must be one-dimensional.")

        packed = np.packbits(values).copy()
        packed.setflags(write=False)
        return cls(values=packed, length=int(values.size))

    @property
    def owned_bytes(self) -> int:
        return int(self.values.nbytes)

    def unpack(self) -> np.ndarray:
        return np.unpackbits(self.values, count=self.length).astype(bool, copy=False)


class StructuralTopologyCache:
    """Byte-bounded cache for topology-only action masks."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES,
    ) -> None:
        self._cache: ByteLRUCache[bytes, _PackedMask] = ByteLRUCache(max_bytes)
        self.hits = 0
        self.misses = 0

    def lookup(
        self,
        state: Any,
        *,
        require_connected_after_switch: bool,
        closeable_branch_ids: tuple[int, ...],
    ) -> tuple[bytes, np.ndarray | None]:
        key = structural_topology_fingerprint(
            state,
            require_connected_after_switch=require_connected_after_switch,
            closeable_branch_ids=closeable_branch_ids,
        )
        packed = self._cache.get(key)
        if packed is None:
            self.misses += 1
            return key, None

        self.hits += 1
        return key, packed.unpack()

    def store(self, key: bytes, mask: np.ndarray) -> bool:
        packed = _PackedMask.from_mask(mask)
        return self._cache.put(
            key,
            packed,
            size_bytes=packed.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
        )

    def clear(self, *, reset_counters: bool = True) -> None:
        self._cache.clear(reset_evictions=reset_counters)
        if reset_counters:
            self.hits = 0
            self.misses = 0

    def info(self) -> dict[str, int]:
        info = self._cache.info()
        return {
            "size": int(info.entries),
            "bytes": int(info.bytes),
            "max_bytes": int(info.max_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(info.evictions),
        }
