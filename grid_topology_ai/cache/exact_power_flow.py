from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from pypower.idx_bus import BUS_TYPE, PQ, PV, REF, VA, VM

from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.cache.persistent_exact import (
    PersistentExactPowerFlowCache,
    PersistentPowerFlowFailure,
    PersistentPowerFlowSuccess,
)
from grid_topology_ai.power_flow_errors import PowerFlowFailureKind
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem


EXACT_POWER_FLOW_CACHE_SCHEMA_VERSION = 2
EXACT_POWER_FLOW_SOLVER_CONTRACT = "pypower-ac-physical-input-v2"
SOLVER_INVOCATION_CACHE_SCHEMA_VERSION = 1
SOLVER_INVOCATION_CONTRACT = "pypower-ac-solver-invocation-v1"
DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES = 64 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128
_NEGATIVE_CACHE_MIN_BYTES = 4 * 1024 * 1024
_NEGATIVE_CACHE_MAX_BYTES = 32 * 1024 * 1024


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _physical_bus_cache_input(values: np.ndarray) -> np.ndarray:
    """Normalize only voltage values used as solver starting guesses.

    PYPOWER builds V0 from BUS_VM/BUS_VA and then imposes online-generator VG.
    For normal PQ/PV/REF rows, BUS_VM is therefore a starting value. BUS_VA is
    also a starting value except at REF buses, where it defines the reference
    frame. All other bus inputs remain exact.
    """

    bus = np.ascontiguousarray(values, dtype=np.float64).copy()
    if bus.ndim != 2 or bus.shape[1] <= max(BUS_TYPE, VM, VA):
        raise ValueError("Canonical power-flow bus matrix has an invalid shape.")

    bus_types = np.rint(bus[:, BUS_TYPE]).astype(np.int64)
    solved_bus = np.isin(bus_types, (PQ, PV, REF))
    non_reference_solved_bus = solved_bus & (bus_types != REF)
    bus[solved_bus, VM] = 0.0
    bus[non_reference_solved_bus, VA] = 0.0
    return bus


def exact_power_flow_fingerprint(
    problem: CanonicalPowerFlowProblem,
    *,
    physics_fingerprint: str,
) -> bytes:
    """Identity of one physical AC PF problem, independent of its V0 guess."""

    digest = hashlib.sha256()
    digest.update(
        f"schema:{EXACT_POWER_FLOW_CACHE_SCHEMA_VERSION};".encode("ascii")
    )
    digest.update(EXACT_POWER_FLOW_SOLVER_CONTRACT.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(physics_fingerprint).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray([float(problem.base_mva)], dtype=np.float64).tobytes())
    _hash_array(digest, _physical_bus_cache_input(problem.bus))
    _hash_array(digest, problem.branch)
    _hash_array(digest, problem.gen)
    return digest.digest()


def solver_invocation_fingerprint(
    problem: CanonicalPowerFlowProblem,
    *,
    physics_fingerprint: str,
) -> bytes:
    """Identity of the exact solver invocation, including starting VM/VA."""

    digest = hashlib.sha256()
    digest.update(
        f"schema:{SOLVER_INVOCATION_CACHE_SCHEMA_VERSION};".encode("ascii")
    )
    digest.update(SOLVER_INVOCATION_CONTRACT.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(physics_fingerprint).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray([float(problem.base_mva)], dtype=np.float64).tobytes())
    _hash_array(digest, problem.bus)
    _hash_array(digest, problem.branch)
    _hash_array(digest, problem.gen)
    return digest.digest()


def _readonly_float64_copy(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _negative_cache_budget(positive_max_bytes: int) -> int:
    positive_max_bytes = max(int(positive_max_bytes), 0)
    if positive_max_bytes == 0:
        return 0
    return min(
        max(positive_max_bytes // 4, _NEGATIVE_CACHE_MIN_BYTES),
        _NEGATIVE_CACHE_MAX_BYTES,
    )


@dataclass(frozen=True, slots=True)
class PowerFlowCacheKey:
    """Both identities required by the two-level in-memory PF cache."""

    positive: bytes
    negative: bytes


@dataclass(frozen=True, slots=True)
class CachedPowerFlowSuccess:
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray

    @classmethod
    def from_result_ppc(
        cls,
        result_ppc: Mapping[str, object],
    ) -> "CachedPowerFlowSuccess":
        return cls(
            bus=_readonly_float64_copy(np.asarray(result_ppc["bus"])),
            branch=_readonly_float64_copy(np.asarray(result_ppc["branch"])),
            gen=_readonly_float64_copy(np.asarray(result_ppc["gen"])),
        )

    @classmethod
    def from_persistent(
        cls,
        record: PersistentPowerFlowSuccess,
    ) -> "CachedPowerFlowSuccess":
        return cls(bus=record.bus, branch=record.branch, gen=record.gen)

    @property
    def owned_bytes(self) -> int:
        return int(self.bus.nbytes + self.branch.nbytes + self.gen.nbytes)

    def to_ppc(
        self,
        *,
        base_mva: float,
        copy_arrays: bool,
    ) -> dict[str, object]:
        if copy_arrays:
            bus = self.bus.copy()
            branch = self.branch.copy()
            gen = self.gen.copy()
        else:
            bus = self.bus
            branch = self.branch
            gen = self.gen

        return {
            "version": "2",
            "baseMVA": float(base_mva),
            "bus": bus,
            "branch": branch,
            "gen": gen,
        }


@dataclass(frozen=True, slots=True)
class CachedPowerFlowFailure:
    failure_kind: PowerFlowFailureKind
    message: str

    @property
    def owned_bytes(self) -> int:
        return 64 + len(self.message.encode("utf-8"))


ExactPowerFlowOutcome = CachedPowerFlowSuccess | CachedPowerFlowFailure


class ExactPowerFlowCache:
    """Bounded PF cache with separate success and exact-failure identities.

    Successful solutions are reusable across different solver starting VM/VA
    when every physical input is identical. Non-convergence is cached only for
    the exact solver invocation, including VM/VA, so a bad starting point can
    never poison the physical-result cache.
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES,
        *,
        negative_max_bytes: int | None = None,
        persistent_cache: PersistentExactPowerFlowCache | None = None,
    ) -> None:
        positive_max_bytes = int(max_bytes)
        if negative_max_bytes is None:
            negative_max_bytes = _negative_cache_budget(positive_max_bytes)

        self._success_cache: ByteLRUCache[bytes, CachedPowerFlowSuccess] = (
            ByteLRUCache(max_bytes=positive_max_bytes)
        )
        self._failure_cache: ByteLRUCache[bytes, CachedPowerFlowFailure] = (
            ByteLRUCache(max_bytes=int(negative_max_bytes))
        )
        self._persistent = (
            persistent_cache
            if persistent_cache is not None
            else PersistentExactPowerFlowCache.from_environment()
        )
        self.hits = 0
        self.misses = 0
        self.positive_hits = 0
        self.negative_hits = 0
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

    @property
    def max_bytes(self) -> int:
        return int(self._success_cache.max_bytes)

    @property
    def negative_max_bytes(self) -> int:
        return int(self._failure_cache.max_bytes)

    @staticmethod
    def _positive_key(key: PowerFlowCacheKey | bytes) -> bytes:
        return key.positive if isinstance(key, PowerFlowCacheKey) else key

    @staticmethod
    def _negative_key(key: PowerFlowCacheKey | bytes) -> bytes:
        return key.negative if isinstance(key, PowerFlowCacheKey) else key

    def _put_success(self, key: bytes, outcome: CachedPowerFlowSuccess) -> bool:
        return self._success_cache.put(
            key,
            outcome,
            size_bytes=outcome.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
        )

    def _put_failure(self, key: bytes, outcome: CachedPowerFlowFailure) -> bool:
        return self._failure_cache.put(
            key,
            outcome,
            size_bytes=outcome.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
        )

    def lookup(
        self,
        problem: CanonicalPowerFlowProblem,
        *,
        physics_fingerprint: str,
    ) -> tuple[PowerFlowCacheKey, ExactPowerFlowOutcome | None]:
        positive_key = exact_power_flow_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        negative_key = solver_invocation_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        key = PowerFlowCacheKey(positive=positive_key, negative=negative_key)

        success = self._success_cache.get(positive_key)
        if success is not None:
            self.hits += 1
            self.positive_hits += 1
            self.l1_hits += 1
            return key, success

        self.l1_misses += 1
        if self._persistent is not None:
            persistent = self._persistent.lookup(positive_key)
            if isinstance(persistent, PersistentPowerFlowSuccess):
                success = CachedPowerFlowSuccess.from_persistent(persistent)
                self._put_success(positive_key, success)
                self.hits += 1
                self.positive_hits += 1
                self.l2_hits += 1
                return key, success
            if isinstance(persistent, PersistentPowerFlowFailure):
                # Old persistent negative records were keyed by the physical
                # problem and are unsafe under the split-key contract.
                self._persistent.discard(positive_key)
            self.l2_misses += 1

        failure = self._failure_cache.get(negative_key)
        if failure is not None:
            self.hits += 1
            self.negative_hits += 1
            self.l1_hits += 1
            return key, failure

        self.misses += 1
        return key, None

    def store_success(
        self,
        key: PowerFlowCacheKey | bytes,
        result_ppc: Mapping[str, object],
    ) -> bool:
        positive_key = self._positive_key(key)
        negative_key = self._negative_key(key)
        outcome = CachedPowerFlowSuccess.from_result_ppc(result_ppc)
        stored = self._put_success(positive_key, outcome)
        self._failure_cache.discard(negative_key)
        if self._persistent is not None:
            self._persistent.store_success(
                positive_key,
                bus=outcome.bus,
                branch=outcome.branch,
                gen=outcome.gen,
            )
        return stored

    def store_not_converged(
        self,
        key: PowerFlowCacheKey | bytes,
        message: str,
    ) -> bool:
        outcome = CachedPowerFlowFailure(
            failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
            message=str(message),
        )
        return self._put_failure(self._negative_key(key), outcome)

    def discard(self, key: PowerFlowCacheKey | bytes) -> None:
        positive_key = self._positive_key(key)
        negative_key = self._negative_key(key)
        self._success_cache.discard(positive_key)
        self._failure_cache.discard(negative_key)
        if self._persistent is not None:
            self._persistent.discard(positive_key)

    def clear(self, *, reset_counters: bool = True) -> None:
        """Clear worker-local caches; persistent successes survive recycling."""

        self._success_cache.clear(reset_evictions=reset_counters)
        self._failure_cache.clear(reset_evictions=reset_counters)
        if reset_counters:
            self.reset_counters()

    def reset_performance_counters(self) -> None:
        self.reset_counters()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.positive_hits = 0
        self.negative_hits = 0
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

    def info(self) -> dict[str, object]:
        positive = self._success_cache.info()
        negative = self._failure_cache.info()
        result: dict[str, object] = {
            "size": int(positive.entries + negative.entries),
            "bytes": int(positive.bytes + negative.bytes),
            "max_bytes": int(positive.max_bytes),
            "negative_max_bytes": int(negative.max_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "positive_hits": int(self.positive_hits),
            "negative_hits": int(self.negative_hits),
            "evictions": int(positive.evictions + negative.evictions),
            "positive_evictions": int(positive.evictions),
            "negative_evictions": int(negative.evictions),
            "positive_entries": int(positive.entries),
            "negative_entries": int(negative.entries),
            "positive_bytes": int(positive.bytes),
            "negative_bytes": int(negative.bytes),
            "l1_hits": int(self.l1_hits),
            "l1_misses": int(self.l1_misses),
            "l2_hits": int(self.l2_hits),
            "l2_misses": int(self.l2_misses),
            "l2_enabled": self._persistent is not None,
        }
        if self._persistent is not None:
            result["persistent"] = self._persistent.info()
        return result
