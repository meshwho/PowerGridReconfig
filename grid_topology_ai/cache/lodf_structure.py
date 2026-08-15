from __future__ import annotations

import hashlib

import numpy as np

from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.lodf import LODFStructure, build_lodf_structure


LODF_STRUCTURE_CACHE_SCHEMA_VERSION = 1
DEFAULT_LODF_STRUCTURE_CACHE_BYTES = 16 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128
_LODF_EPS = 1e-9
_STATUS_IDX = BRANCH_FEATURE_COLUMNS.index("br_status")
_X_IDX = BRANCH_FEATURE_COLUMNS.index("x")
_RATE_IDX = BRANCH_FEATURE_COLUMNS.index("rate_a")


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def lodf_structure_fingerprint(state: GridFMState) -> bytes:
    """Return exact identity of the topology math used by LODF screening.

    Dynamic PF, loading, and positive thermal-rating magnitudes are excluded.
    They do not affect PTDF/LODF structure and are read from the current state
    when actions are scored.
    """

    branch_features = np.asarray(state.branch_features)
    edge_index = np.asarray(state.edge_index, dtype=np.int64)
    num_branches = int(branch_features.shape[0])
    num_buses = int(state.bus_features.shape[0])

    if branch_features.ndim != 2:
        raise ValueError("branch_features must be two-dimensional.")
    if edge_index.shape != (2, num_branches):
        raise ValueError("edge_index shape does not match branch_features.")

    status = np.asarray(branch_features[:, _STATUS_IDX], dtype=np.float64)
    reactance = np.asarray(branch_features[:, _X_IDX], dtype=np.float64)
    rate = np.asarray(branch_features[:, _RATE_IDX], dtype=np.float64)
    eligible = (
        (status > 0.0)
        & np.isfinite(reactance)
        & (np.abs(reactance) > _LODF_EPS)
        & np.isfinite(rate)
        & (rate > _LODF_EPS)
    )
    positions = np.flatnonzero(eligible).astype(np.int64, copy=False)

    digest = hashlib.sha256()
    digest.update(
        f"lodf-structure-v{LODF_STRUCTURE_CACHE_SCHEMA_VERSION}".encode("ascii")
    )
    digest.update(np.asarray([num_buses, num_branches], dtype=np.int64).tobytes())
    _hash_array(digest, positions)
    if positions.size:
        _hash_array(
            digest,
            np.asarray(edge_index[:, positions].T, dtype=np.int64),
        )
        _hash_array(
            digest,
            np.asarray(reactance[positions], dtype=np.float64),
        )
    return digest.digest()


class LODFStructureCache:
    """Byte-bounded cache of topology-only LODF matrices."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_LODF_STRUCTURE_CACHE_BYTES,
    ) -> None:
        self._cache: ByteLRUCache[bytes, LODFStructure] = ByteLRUCache(
            max_bytes=int(max_bytes)
        )
        self.hits = 0
        self.misses = 0

    @property
    def max_bytes(self) -> int:
        return int(self._cache.max_bytes)

    def get_or_build(self, state: GridFMState) -> LODFStructure | None:
        key = lodf_structure_fingerprint(state)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        structure = build_lodf_structure(state)
        if structure is None:
            return None

        self._cache.put(
            key,
            structure,
            size_bytes=(
                int(structure.owned_bytes)
                + len(key)
                + _ENTRY_OVERHEAD_BYTES
            ),
        )
        return structure

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
