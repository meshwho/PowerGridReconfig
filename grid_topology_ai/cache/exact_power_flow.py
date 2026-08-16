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
DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES = 64 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _physical_bus_cache_input(values: np.ndarray) -> np.ndarray:
    """Return bus input with warm-start-only voltage guesses normalized away.

    PYPOWER builds its AC starting vector from BUS_VM/BUS_VA, then overwrites the
    voltage magnitude at online generator buses from GEN_VG. For normal solved
    PQ/PV/REF buses, BUS_VM is therefore only a starting guess. BUS_VA is also a
    starting guess except at REF buses, whose angle defines the reference frame.

    Keep REF-bus VA and all non-standard bus rows exact. Normalize only the
    starting-point fields that do not define the physical AC problem.
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
    """Return the identity of one physical AC power-flow problem.

    Generator P/Q/status, loads, topology, ratings, voltage limits, generator
    voltage setpoints and the physics contract remain exact. Only warm-start-only
    BUS_VM/BUS_VA values are normalized, while REF-bus VA remains part of the key.
    """

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


def _readonly_float64_copy(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


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
    """Exact physical-input cache: bounded RAM first, optional persistent L2."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES,
        *,
        persistent_cache: PersistentExactPowerFlowCache | None = None,
    ) -> None:
        self._cache: ByteLRUCache[bytes, ExactPowerFlowOutcome] = ByteLRUCache(
            max_bytes=max_bytes
        )
        self._persistent = (
            persistent_cache
            if persistent_cache is not None
            else PersistentExactPowerFlowCache.from_environment()
        )
        self.hits = 0
        self.misses = 0
        self.negative_hits = 0
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

    @property
    def max_bytes(self) -> int:
        return int(self._cache.max_bytes)

    def _put_l1(self, key: bytes, outcome: ExactPowerFlowOutcome) -> bool:
        return self._cache.put(
            key,
            outcome,
            size_bytes=outcome.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
        )

    def lookup(
        self,
        problem: CanonicalPowerFlowProblem,
        *,
        physics_fingerprint: str,
    ) -> tuple[bytes, ExactPowerFlowOutcome | None]:
        key = exact_power_flow_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        outcome = self._cache.get(key)
        if outcome is not None:
            self.hits += 1
            self.l1_hits += 1
            if isinstance(outcome, CachedPowerFlowFailure):
                self.negative_hits += 1
            return key, outcome

        self.l1_misses += 1
        if self._persistent is not None:
            persistent = self._persistent.lookup(key)
            if isinstance(persistent, PersistentPowerFlowSuccess):
                outcome = CachedPowerFlowSuccess.from_persistent(persistent)
            elif isinstance(persistent, PersistentPowerFlowFailure):
                outcome = CachedPowerFlowFailure(
                    failure_kind=PowerFlowFailureKind.NOT_CONVERGED,
                    message=persistent.message,
                )
            else:
                outcome = None

            if outcome is not None:
                self._put_l1(key, outcome)
                self.hits += 1
                self.l2_hits += 1
                if isinstance(outcome, CachedPowerFlowFailure):
                    self.negative_hits += 1
                return key, outcome
            self.l2_misses += 1

        self.misses += 1
        return key, None

    def store_success(
        self,
        key: bytes,
        result_ppc: Mapping[str, object],
    ) -> bool:
        outcome = CachedPowerFlowSuccess.from_result_ppc(result_ppc)
        stored = self._put_l1(key, outcome)
        if self._persistent is not None:
            self._persistent.store_success(
                key,
                bus=outcome.bus,
                branch=outcome.branch,
                gen=outcome.gen,
            )
        return stored

    def store_not_converged(self, key: bytes, message: str) -> bool:
        """Do not reuse solver failure across different voltage starting guesses.

        Successful solutions are keyed by physical inputs with warm-start-only
        BUS_VM/BUS_VA normalized. Non-convergence, however, can depend on that
        starting point, so a failure must not be promoted to a physical-input hit.
        """

        del key, message
        return False

    def discard(self, key: bytes) -> None:
        self._cache.discard(key)
        if self._persistent is not None:
            self._persistent.discard(key)

    def clear(self, *, reset_counters: bool = True) -> None:
        """Clear only L1; persistent exact results survive worker recycling."""

        self._cache.clear(reset_evictions=reset_counters)
        if reset_counters:
            self.reset_counters()

    def reset_performance_counters(self) -> None:
        self.reset_counters()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.negative_hits = 0
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

    def info(self) -> dict[str, object]:
        info = self._cache.info()
        result: dict[str, object] = {
            "size": int(info.entries),
            "bytes": int(info.bytes),
            "max_bytes": int(info.max_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "negative_hits": int(self.negative_hits),
            "evictions": int(info.evictions),
            "l1_hits": int(self.l1_hits),
            "l1_misses": int(self.l1_misses),
            "l2_hits": int(self.l2_hits),
            "l2_misses": int(self.l2_misses),
            "l2_enabled": self._persistent is not None,
        }
        if self._persistent is not None:
            result["persistent"] = self._persistent.info()
        return result
