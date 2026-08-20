from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from pypower.idx_bus import BUS_TYPE, PQ, PV, REF, VA, VM
from pypower.idx_gen import PG, QG

from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.power_flow.problem import CanonicalPowerFlowProblem


POWER_FLOW_WARM_START_ENABLED_ENV = "POWERGRID_ENABLE_PF_WARM_START"
POWER_FLOW_WARM_START_MAX_BYTES_ENV = "POWERGRID_PF_WARM_START_CACHE_MAX_BYTES"
DEFAULT_POWER_FLOW_WARM_START_CACHE_BYTES = 16 * 1024 * 1024
WARM_START_CACHE_SCHEMA_VERSION = 1
WARM_START_CACHE_CONTRACT = "pypower-ac-warm-start-v1"
_DEFAULT_MAX_CANDIDATES_PER_BUCKET = 8
_BUCKET_OVERHEAD_BYTES = 192
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def warm_start_enabled_from_environment() -> bool:
    return (
        os.environ.get(POWER_FLOW_WARM_START_ENABLED_ENV, "").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _configured_max_bytes() -> int:
    raw = os.environ.get(POWER_FLOW_WARM_START_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_POWER_FLOW_WARM_START_CACHE_BYTES

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{POWER_FLOW_WARM_START_MAX_BYTES_ENV} must be a positive integer, "
            f"got {raw!r}."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{POWER_FLOW_WARM_START_MAX_BYTES_ENV} must be > 0, got {value}."
        )
    return value


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _warm_start_bus_input(values: np.ndarray) -> np.ndarray:
    bus = np.ascontiguousarray(values, dtype=np.float64).copy()
    if bus.ndim != 2 or bus.shape[1] <= max(BUS_TYPE, VM, VA):
        raise ValueError("Canonical power-flow bus matrix has an invalid shape.")

    bus_types = np.rint(bus[:, BUS_TYPE]).astype(np.int64)
    solved_bus = np.isin(bus_types, (PQ, PV, REF))
    bus[solved_bus, VM] = 0.0
    bus[solved_bus & (bus_types != REF), VA] = 0.0
    return bus


def _warm_start_gen_input(values: np.ndarray) -> np.ndarray:
    gen = np.ascontiguousarray(values, dtype=np.float64).copy()
    if gen.ndim != 2 or gen.shape[1] <= max(PG, QG):
        raise ValueError("Canonical power-flow generator matrix has an invalid shape.")

    gen[:, PG] = 0.0
    gen[:, QG] = 0.0
    return gen


def warm_start_bucket_fingerprint(
    problem: CanonicalPowerFlowProblem,
    *,
    physics_fingerprint: str,
) -> bytes:
    """Group problems that differ only by generator P/Q and voltage start guess."""

    digest = hashlib.sha256()
    digest.update(f"schema:{WARM_START_CACHE_SCHEMA_VERSION};".encode("ascii"))
    digest.update(WARM_START_CACHE_CONTRACT.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(physics_fingerprint).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray([float(problem.base_mva)], dtype=np.float64).tobytes())
    _hash_array(digest, _warm_start_bus_input(problem.bus))
    _hash_array(digest, problem.branch)
    _hash_array(digest, _warm_start_gen_input(problem.gen))
    return digest.digest()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PowerFlowWarmStart:
    generator_p_mw: np.ndarray
    generator_q_mvar: np.ndarray
    vm: np.ndarray
    va: np.ndarray

    @classmethod
    def from_solution(
        cls,
        problem: CanonicalPowerFlowProblem,
        result_ppc: Mapping[str, object],
    ) -> PowerFlowWarmStart | None:
        gen = np.asarray(problem.gen, dtype=np.float64)
        bus = np.asarray(result_ppc["bus"], dtype=np.float64)
        if gen.ndim != 2 or gen.shape[1] <= max(PG, QG):
            return None
        if bus.ndim != 2 or bus.shape[1] <= max(VM, VA):
            return None

        p_mw = np.asarray(gen[:, PG], dtype=np.float64)
        q_mvar = np.asarray(gen[:, QG], dtype=np.float64)
        vm = np.asarray(bus[:, VM], dtype=np.float64)
        va = np.asarray(bus[:, VA], dtype=np.float64)
        if not all(np.isfinite(values).all() for values in (p_mw, q_mvar, vm, va)):
            return None

        return cls(
            generator_p_mw=_readonly(p_mw),
            generator_q_mvar=_readonly(q_mvar),
            vm=_readonly(vm),
            va=_readonly(va),
        )

    @property
    def owned_bytes(self) -> int:
        return int(
            self.generator_p_mw.nbytes
            + self.generator_q_mvar.nbytes
            + self.vm.nbytes
            + self.va.nbytes
        )

    def distance_to(self, problem: CanonicalPowerFlowProblem) -> float:
        gen = np.asarray(problem.gen, dtype=np.float64)
        if gen.ndim != 2 or gen.shape[0] != self.generator_p_mw.shape[0]:
            return math.inf

        p_mw = np.asarray(gen[:, PG], dtype=np.float64)
        q_mvar = np.asarray(gen[:, QG], dtype=np.float64)
        if not np.isfinite(p_mw).all() or not np.isfinite(q_mvar).all():
            return math.inf

        p_scale = np.maximum(
            np.maximum(np.abs(p_mw), np.abs(self.generator_p_mw)),
            1.0,
        )
        q_scale = np.maximum(
            np.maximum(np.abs(q_mvar), np.abs(self.generator_q_mvar)),
            1.0,
        )
        p_distance = float(np.max(np.abs(p_mw - self.generator_p_mw) / p_scale))
        q_distance = float(np.max(np.abs(q_mvar - self.generator_q_mvar) / q_scale))
        return max(p_distance, q_distance)

    def apply_to_ppc(self, ppc: dict[str, object]) -> bool:
        """Replace only the solver starting voltage; keep physical inputs intact."""

        bus = np.asarray(ppc["bus"])
        if bus.ndim != 2 or bus.shape[0] != self.vm.shape[0]:
            return False
        if bus.shape[1] <= max(BUS_TYPE, VM, VA):
            return False

        bus_types = np.rint(bus[:, BUS_TYPE]).astype(np.int64)
        solved_bus = np.isin(bus_types, (PQ, PV, REF))
        non_reference_solved_bus = solved_bus & (bus_types != REF)

        bus[solved_bus, VM] = self.vm[solved_bus]
        bus[non_reference_solved_bus, VA] = self.va[non_reference_solved_bus]
        return True


WarmStartBucket = tuple[PowerFlowWarmStart, ...]


class PowerFlowWarmStartCache:
    """Bounded nearest-operating-point cache used only to seed real PF solves."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_POWER_FLOW_WARM_START_CACHE_BYTES,
        *,
        max_candidates_per_bucket: int = _DEFAULT_MAX_CANDIDATES_PER_BUCKET,
    ) -> None:
        max_candidates_per_bucket = int(max_candidates_per_bucket)
        if max_candidates_per_bucket <= 0:
            raise ValueError("max_candidates_per_bucket must be positive.")

        self._cache: ByteLRUCache[bytes, WarmStartBucket] = ByteLRUCache(
            max_bytes=int(max_bytes)
        )
        self.max_candidates_per_bucket = max_candidates_per_bucket
        self.hits = 0
        self.misses = 0
        self.stores = 0

    @classmethod
    def from_environment(cls) -> PowerFlowWarmStartCache | None:
        if not warm_start_enabled_from_environment():
            return None
        return cls(max_bytes=_configured_max_bytes())

    def lookup(
        self,
        problem: CanonicalPowerFlowProblem,
        *,
        physics_fingerprint: str,
    ) -> PowerFlowWarmStart | None:
        key = warm_start_bucket_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        bucket = self._cache.get(key)
        if not bucket:
            self.misses += 1
            return None

        candidate = min(bucket, key=lambda item: item.distance_to(problem))
        if not math.isfinite(candidate.distance_to(problem)):
            self.misses += 1
            return None

        self.hits += 1
        return candidate

    def store_success(
        self,
        problem: CanonicalPowerFlowProblem,
        result_ppc: Mapping[str, object],
        *,
        physics_fingerprint: str,
    ) -> bool:
        candidate = PowerFlowWarmStart.from_solution(problem, result_ppc)
        if candidate is None:
            return False

        key = warm_start_bucket_fingerprint(
            problem,
            physics_fingerprint=physics_fingerprint,
        )
        existing = list(self._cache.get(key) or ())
        existing = [
            item
            for item in existing
            if not (
                np.array_equal(item.generator_p_mw, candidate.generator_p_mw)
                and np.array_equal(item.generator_q_mvar, candidate.generator_q_mvar)
            )
        ]
        existing.append(candidate)
        bucket = tuple(existing[-self.max_candidates_per_bucket :])
        size_bytes = (
            len(key)
            + _BUCKET_OVERHEAD_BYTES
            + sum(item.owned_bytes for item in bucket)
        )
        stored = self._cache.put(key, bucket, size_bytes=size_bytes)
        if stored:
            self.stores += 1
        return stored

    def clear(self, *, reset_counters: bool = True) -> None:
        self._cache.clear(reset_evictions=reset_counters)
        if reset_counters:
            self.reset_counters()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def info(self) -> dict[str, object]:
        info = self._cache.info()
        return {
            "enabled": True,
            "buckets": int(info.entries),
            "bytes": int(info.bytes),
            "max_bytes": int(info.max_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "stores": int(self.stores),
            "evictions": int(info.evictions),
        }
