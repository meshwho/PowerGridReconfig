from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np
from pypower.api import ppoption, rundcpf
from pypower.idx_brch import BR_STATUS, PF, PT, RATE_A

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.cache.lodf_structure import LODFStructureCache
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.lodf import (
    build_lodf_structure,
    rank_actions_with_lodf_structure,
)
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend
from grid_topology_ai.power_flow.problem import CanonicalPowerFlowProblem

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


@dataclass(frozen=True)
class DCActionScore:
    """DC screening score for one topology action. Lower is better."""

    action: GridFMAction
    success: bool
    penalty: float
    max_loading_percent: float
    num_overloaded: int
    num_hard_overloaded: int
    total_overload: float
    hard_overload: float
    policy_prior: float


class DCActionScreener:
    """Fast DC ranking before the authoritative AC transition."""

    def __init__(
        self,
        top_k: int = 30,
        candidate_pool: int = 120,
        policy_weight: float = 0.0,
        failure_penalty: float = 1_000_000_000.0,
        enable_cache: bool = True,
        physics_config: PhysicsConfig | None = None,
        cache_max_bytes: int = DEFAULT_DC_SCREENING_CACHE_BYTES,
    ) -> None:
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.policy_weight = float(policy_weight)
        self.failure_penalty = float(failure_penalty)
        self.enable_cache = bool(enable_cache)
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
        self._screening_cache = DCScreeningCache(max_bytes=int(cache_max_bytes))

    @property
    def cache_hits(self) -> int:
        return int(self._screening_cache.hits)

    @property
    def cache_misses(self) -> int:
        return int(self._screening_cache.misses)

    def clear_cache(self) -> None:
        self._screening_cache.clear(reset_counters=True)

    def cache_info(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(self._screening_cache.info())
        info["enabled"] = bool(self.enable_cache)
        total = int(info["hits"]) + int(info["misses"])
        info["hit_rate"] = (
            float(info["hits"]) / float(total)
            if total > 0
            else 0.0
        )
        return info

    @staticmethod
    def supports(action: GridFMAction) -> bool:
        return (
            action.kind == "set_branch_status"
            and action.branch_id is not None
            and action.target_status in (0, 1)
        )

    def screen_actions(
        self,
        *,
        state: GridFMState,
        actions: list[GridFMAction],
        backend: GridFMPowerFlowBackend,
        neural_policy: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[GridFMAction]:
        ranked = self.rank_actions(
            state=state,
            actions=actions,
            backend=backend,
            neural_policy=neural_policy,
        )
        effective_top_k = self.top_k if top_k is None else int(top_k)
        return ranked if effective_top_k <= 0 else ranked[:effective_top_k]

    def rank_actions(
        self,
        *,
        state: GridFMState,
        actions: list[GridFMAction],
        backend: GridFMPowerFlowBackend,
        neural_policy: np.ndarray | None = None,
    ) -> list[GridFMAction]:
        scored = [
            self.score_action(
                state=state,
                action=action,
                backend=backend,
                neural_policy=neural_policy,
            )
            for action in actions
            if self.supports(action)
        ]
        scored.sort(
            key=lambda item: (
                not item.success,
                item.penalty,
                -item.policy_prior,
                item.max_loading_percent,
                item.action.action_id,
            )
        )
        return [item.action for item in scored]

    def score_action(
        self,
        *,
        state: GridFMState,
        action: GridFMAction,
        backend: GridFMPowerFlowBackend,
        neural_policy: np.ndarray | None = None,
    ) -> DCActionScore:
        if not self.supports(action):
            raise ValueError(
                "DCActionScreener supports only branch-status actions."
            )

        policy_prior = self._policy_prior(action, neural_policy)

        try:
            ppc, _frames = backend._build_ppc_from_state(
                state=state,
                action=action,
            )
            problem = CanonicalPowerFlowProblem(
                base_mva=float(ppc["baseMVA"]),
                bus=np.asarray(ppc["bus"], dtype=np.float64),
                branch=np.asarray(ppc["branch"], dtype=np.float64),
                gen=np.asarray(ppc["gen"], dtype=np.float64),
            )
        except Exception:
            return self._failed_score(action=action, policy_prior=policy_prior)

        cache_key: bytes | None = None
        cached: CachedDCScreeningResult | None = None
        if self.enable_cache:
            cache_key, cached = self._screening_cache.lookup(
                problem,
                physics_fingerprint=self.physics_config.fingerprint(),
            )

        if cached is None:
            try:
                result_ppc, success = rundcpf(
                    problem.to_ppc(),
                    ppoption(VERBOSE=0, OUT_ALL=0),
                )
                physical = (
                    self._physical_dc_result(result_ppc)
                    if bool(success)
                    else self._failed_physical_result()
                )
            except Exception:
                return self._failed_score(action=action, policy_prior=policy_prior)

            if self.enable_cache:
                assert cache_key is not None
                self._screening_cache.store(cache_key, physical)
        else:
            physical = cached

        return self._score_from_physical_result(
            action=action,
            physical=physical,
            policy_prior=policy_prior,
        )

    @staticmethod
    def _policy_prior(
        action: GridFMAction,
        neural_policy: np.ndarray | None,
    ) -> float:
        if neural_policy is None:
            return 0.0
        if not 0 <= action.action_id < len(neural_policy):
            return 0.0
        return float(neural_policy[action.action_id])

    def _physical_dc_result(
        self,
        result_ppc: dict[str, Any],
    ) -> CachedDCScreeningResult:
        branch = np.asarray(result_ppc["branch"])
        status = branch[:, BR_STATUS].astype(float)
        rate_a = branch[:, RATE_A].astype(float)
        pf = branch[:, PF].astype(float)
        pt = branch[:, PT].astype(float)

        active = (status > 0.0) & (rate_a > 1e-6)
        if not np.any(active):
            return self._failed_physical_result()

        flow_abs = np.maximum(np.abs(pf), np.abs(pt))
        loading = np.zeros_like(flow_abs, dtype=float)
        loading[active] = 100.0 * flow_abs[active] / rate_a[active]
        active_loading = loading[active]

        overload_threshold = (
            self.physics_config.overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        hard_threshold = (
            self.physics_config.hard_overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        overload = np.where(
            active_loading > overload_threshold,
            active_loading - self.physics_config.overload_limit_percent,
            0.0,
        )
        hard = np.where(
            active_loading > hard_threshold,
            active_loading - self.physics_config.hard_overload_limit_percent,
            0.0,
        )

        return CachedDCScreeningResult(
            success=True,
            max_loading_percent=float(np.max(active_loading)),
            num_overloaded=int(np.sum(active_loading > overload_threshold)),
            num_hard_overloaded=int(np.sum(active_loading > hard_threshold)),
            total_overload=float(np.sum(overload)),
            hard_overload=float(np.sum(hard)),
        )

    @staticmethod
    def _failed_physical_result() -> CachedDCScreeningResult:
        return CachedDCScreeningResult(
            success=False,
            max_loading_percent=float("inf"),
            num_overloaded=9999,
            num_hard_overloaded=9999,
            total_overload=float("inf"),
            hard_overload=float("inf"),
        )

    def _score_from_physical_result(
        self,
        *,
        action: GridFMAction,
        physical: CachedDCScreeningResult,
        policy_prior: float,
    ) -> DCActionScore:
        if not physical.success:
            return self._failed_score(action=action, policy_prior=policy_prior)

        overload_threshold = (
            self.physics_config.overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        max_excess = (
            physical.max_loading_percent
            - self.physics_config.overload_limit_percent
            if physical.max_loading_percent > overload_threshold
            else 0.0
        )
        penalty = (
            2.0 * physical.total_overload
            + 5.0 * physical.hard_overload
            + 10.0 * physical.num_overloaded
            + 30.0 * physical.num_hard_overloaded
            + 0.10 * max_excess
        )
        if self.policy_weight > 0.0 and policy_prior > 0.0:
            penalty -= self.policy_weight * log(policy_prior + 1e-12)

        return DCActionScore(
            action=action,
            success=True,
            penalty=float(penalty),
            max_loading_percent=float(physical.max_loading_percent),
            num_overloaded=int(physical.num_overloaded),
            num_hard_overloaded=int(physical.num_hard_overloaded),
            total_overload=float(physical.total_overload),
            hard_overload=float(physical.hard_overload),
            policy_prior=float(policy_prior),
        )

    def _failed_score(
        self,
        *,
        action: GridFMAction,
        policy_prior: float,
    ) -> DCActionScore:
        return DCActionScore(
            action=action,
            success=False,
            penalty=float(self.failure_penalty),
            max_loading_percent=float("inf"),
            num_overloaded=9999,
            num_hard_overloaded=9999,
            total_overload=float("inf"),
            hard_overload=float("inf"),
            policy_prior=float(policy_prior),
        )


def rank_actions_by_lodf_screening(
    *,
    state: GridFMState,
    actions: list[GridFMAction],
    physics_config: PhysicsConfig | None = None,
    structure_cache: LODFStructureCache | None = None,
) -> list[GridFMAction]:
    """Rank actions with reusable topology-only LODF structure."""
    if not actions:
        return actions

    structure = (
        structure_cache.get_or_build(state)
        if structure_cache is not None
        else build_lodf_structure(state)
    )
    if structure is None:
        return actions

    return rank_actions_with_lodf_structure(
        state=state,
        actions=actions,
        structure=structure,
        physics_config=physics_config,
    )
