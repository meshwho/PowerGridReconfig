from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np
from pypower.api import ppoption, rundcpf
from pypower.idx_brch import BR_STATUS, PF, PT, RATE_A

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


@dataclass(frozen=True)
class DCActionScore:
    """
    DC screening score for one switch action.

    Lower penalty is better.
    """

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
    """
    Fast DC power-flow based action screener.

    Important:
    - This is not a final safety validator.
    - It only ranks / filters candidate topology actions.
    - AC power flow is still used by env.step() and final evaluation.

    Therefore this module should not change correctness of final solved/handoff
    decisions. It only changes which switch actions MCTS expands first.
    """

    def __init__(
        self,
        top_k: int = 30,
        candidate_pool: int = 120,
        policy_weight: float = 0.0,
        failure_penalty: float = 1_000_000_000.0,
        enable_cache: bool = True,
        physics_config: PhysicsConfig | None = None,
    ):
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.policy_weight = float(policy_weight)
        self.failure_penalty = float(failure_penalty)
        self.enable_cache = bool(enable_cache)
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG

        self._cache: dict[tuple, DCActionScore] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "enabled": self.enable_cache,
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def screen_actions(
        self,
        *,
        state: GridFMState,
        actions: list[GridFMAction],
        backend: GridFMPowerFlowBackend,
        neural_policy: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[GridFMAction]:
        """
        Rank actions using DC PF and return the best ones.

        actions:
            switch_off_branch actions only.
        """

        if not actions:
            return []

        effective_top_k = self.top_k if top_k is None else int(top_k)

        if effective_top_k <= 0:
            effective_top_k = len(actions)

        scored: list[DCActionScore] = []

        for action in actions:
            if action.action_type != "switch_off_branch":
                continue

            scored.append(
                self.score_action(
                    state=state,
                    action=action,
                    backend=backend,
                    neural_policy=neural_policy,
                )
            )

        scored.sort(
            key=lambda item: (
                not item.success,
                item.penalty,
                -item.policy_prior,
                item.max_loading_percent,
                item.action.action_id,
            )
        )

        return [item.action for item in scored[:effective_top_k]]

    def score_action(
        self,
        *,
        state: GridFMState,
        action: GridFMAction,
        backend: GridFMPowerFlowBackend,
        neural_policy: np.ndarray | None = None,
    ) -> DCActionScore:
        if action.branch_id is None:
            raise ValueError("DCActionScreener can score only branch switch actions.")

        policy_prior = 0.0

        if neural_policy is not None and 0 <= action.action_id < len(neural_policy):
            policy_prior = float(neural_policy[action.action_id])

        cache_key = self._make_cache_key(
            state=state,
            action=action,
            backend=backend,
            policy_prior=policy_prior,
        )

        if self.enable_cache and cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        if self.enable_cache:
            self.cache_misses += 1

        try:
            ppc, _frames = backend._build_ppc_from_state(
                state=state,
                switched_off_branch_id=int(action.branch_id),
            )

            ppopt = ppoption(
                VERBOSE=0,
                OUT_ALL=0,
            )

            result_ppc, success = rundcpf(ppc, ppopt)
            success = bool(success)

            if not success:
                score = self._failed_score(
                    action=action,
                    policy_prior=policy_prior,
                )
            else:
                score = self._score_dc_result(
                    action=action,
                    result_ppc=result_ppc,
                    policy_prior=policy_prior,
                )

        except Exception:
            score = self._failed_score(
                action=action,
                policy_prior=policy_prior,
            )

        if self.enable_cache:
            self._cache[cache_key] = score

        return score

    def _score_dc_result(
        self,
        *,
        action: GridFMAction,
        result_ppc: dict[str, Any],
        policy_prior: float,
    ) -> DCActionScore:
        branch = result_ppc["branch"]

        status = branch[:, BR_STATUS].astype(float)
        rate_a = branch[:, RATE_A].astype(float)

        # DC PF gives active power flows. Use max(|PF|, |PT|).
        pf = branch[:, PF].astype(float)
        pt = branch[:, PT].astype(float)

        active_mask = (status > 0.0) & (rate_a > 1e-6)

        if not np.any(active_mask):
            return self._failed_score(
                action=action,
                policy_prior=policy_prior,
            )

        flow_abs = np.maximum(np.abs(pf), np.abs(pt))
        loading = np.zeros_like(flow_abs, dtype=float)
        loading[active_mask] = 100.0 * flow_abs[active_mask] / rate_a[active_mask]

        active_loading = loading[active_mask]

        max_loading = float(np.max(active_loading))
        overload_threshold = (
            self.physics_config.overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        hard_overload_threshold = (
            self.physics_config.hard_overload_limit_percent
            + self.physics_config.thermal_tolerance_percent
        )
        overload_vector = np.where(
            active_loading > overload_threshold,
            active_loading - self.physics_config.overload_limit_percent,
            0.0,
        )
        hard_vector = np.where(
            active_loading > hard_overload_threshold,
            active_loading - self.physics_config.hard_overload_limit_percent,
            0.0,
        )

        total_overload = float(np.sum(overload_vector))
        hard_overload = float(np.sum(hard_vector))

        num_overloaded = int(np.sum(active_loading > overload_threshold))
        num_hard_overloaded = int(
            np.sum(active_loading > hard_overload_threshold)
        )
        max_overload_excess = (
            max_loading - self.physics_config.overload_limit_percent
            if max_loading > overload_threshold
            else 0.0
        )

        # Similar idea to MCTS state penalty, but without voltage terms
        # because DC PF does not model voltage magnitudes/reactive power.
        penalty = (
            2.0 * total_overload
            + 5.0 * hard_overload
            + 10.0 * num_overloaded
            + 30.0 * num_hard_overloaded
            + 0.10 * max_overload_excess
        )

        if self.policy_weight > 0.0 and policy_prior > 0.0:
            # Small tie-breaker: prefer actions the neural model also likes.
            penalty -= self.policy_weight * log(policy_prior + 1e-12)

        return DCActionScore(
            action=action,
            success=True,
            penalty=float(penalty),
            max_loading_percent=float(max_loading),
            num_overloaded=int(num_overloaded),
            num_hard_overloaded=int(num_hard_overloaded),
            total_overload=float(total_overload),
            hard_overload=float(hard_overload),
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

    @staticmethod
    def _make_cache_key(
        *,
        state: GridFMState,
        action: GridFMAction,
        backend: GridFMPowerFlowBackend,
        policy_prior: float,
    ) -> tuple:
        outaged = set(int(x) for x in state.outaged_branch_ids)

        if action.branch_id is not None:
            outaged.add(int(action.branch_id))

        return (
            int(state.scenario_id),
            round(float(backend.base_mva), 6),
            tuple(sorted(outaged)),
            int(action.action_id),
            int(action.branch_id) if action.branch_id is not None else -1,
            # Policy prior is intentionally not included.
            # DC physical score does not depend on neural policy.
        )
