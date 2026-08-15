from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem


DEFAULT_DC_SCREENING_CACHE_BYTES = 16 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128


def _hash_array(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def dc_screening_fingerprint(
    problem: CanonicalPowerFlowProblem,
    *,
    physics_fingerprint: str,
) -> bytes:
    """Return the exact identity of one DC screening problem."""

    digest = hashlib.sha256()
    digest.update(b"pypower-dc-screening-v1\0")
    digest.update(str(physics_fingerprint).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray([problem.base_mva], dtype=np.float64).tobytes())
    _hash_array(digest, problem.bus)
    _hash_array(digest, problem.branch)
    _hash_array(digest, problem.gen)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class CachedDCScreeningResult:
    success: bool
    max_loading_percent: float
    num_overloaded: int
    num_hard_overloaded: int
    total_overload: float
    hard_overload: float

    @property
    def owned_bytes(self) -> int:
        return 64


class DCScreeningCache:
    """Byte-bounded cache for exact DC screening results."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_DC_SCREENING_CACHE_BYTES,
    ) -> None:
        self._cache: ByteLRUCache[bytes, CachedDCScreeningResult] = ByteLRUCache(
            max_bytes
        )
        self.hits = 0
        self.misses = 0

    def lookup(
        self,
        problem: CanonicalPowerFlowProblem,
        *,
        physics_fingerprint: str,
    ) -> tuple[bytes, CachedDCScreeningResult | None]:
        key = dc_screening_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        result = self._cache.get(key)
        if result is None:
            self.misses += 1
            return key, None

        self.hits += 1
        return key, result

    def store(self, key: bytes, result: CachedDCScreeningResult) -> bool:
        return self._cache.put(
            key,
            result,
            size_bytes=result.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
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
