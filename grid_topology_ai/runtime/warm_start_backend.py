from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pypower.idx_bus import VA, VM

from grid_topology_ai.cache.power_flow_warm_start import (
    PowerFlowWarmStartCache,
    warm_start_enabled_from_environment,
)
from grid_topology_ai.power_flow_errors import PowerFlowNotConverged
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem
from grid_topology_ai.runtime.scenario_store import (
    MemoryMappedGridFMPowerFlowBackend,
    build_memory_mapped_teacher_context as _build_memory_mapped_teacher_context,
)


_MAX_WARM_START_DISTANCE = 0.20


class WarmStartMemoryMappedGridFMPowerFlowBackend(
    MemoryMappedGridFMPowerFlowBackend
):
    """Mmap backend with bounded warm seeds and no fuzzy result reuse."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._power_flow_warm_start_cache = (
            PowerFlowWarmStartCache.from_environment()
            if bool(self.enable_cache)
            else None
        )
        self._warm_start_applications = 0
        self._warm_start_distance_rejections = 0
        self._warm_start_fallbacks = 0

    def performance_info(self) -> dict[str, object]:
        info = dict(super().performance_info())
        warm_cache = self._power_flow_warm_start_cache
        if warm_cache is None:
            info.update(
                {
                    "warm_start_enabled": False,
                    "warm_start_hits": 0,
                    "warm_start_misses": 0,
                    "warm_start_evictions": 0,
                    "warm_start_distance_rejections": 0,
                    "warm_start_fallbacks": 0,
                }
            )
            return info

        warm_info = warm_cache.info()
        info.update(
            {
                "warm_start_enabled": True,
                "warm_start_hits": int(self._warm_start_applications),
                "warm_start_misses": int(warm_info["misses"]),
                "warm_start_evictions": int(warm_info["evictions"]),
                "warm_start_distance_rejections": int(
                    self._warm_start_distance_rejections
                ),
                "warm_start_fallbacks": int(self._warm_start_fallbacks),
            }
        )
        return info

    def reset_performance_counters(self) -> None:
        super().reset_performance_counters()
        self._warm_start_applications = 0
        self._warm_start_distance_rejections = 0
        self._warm_start_fallbacks = 0
        warm_cache = self._power_flow_warm_start_cache
        if warm_cache is not None:
            warm_cache.reset_counters()

    def clear_cache(self) -> None:
        super().clear_cache()
        warm_cache = self._power_flow_warm_start_cache
        if warm_cache is not None:
            warm_cache.clear(reset_counters=True)
        self._warm_start_applications = 0
        self._warm_start_distance_rejections = 0
        self._warm_start_fallbacks = 0

    def _solve_ppc(
        self,
        ppc: dict[str, Any],
        *,
        context: str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        warm_cache = self._power_flow_warm_start_cache
        if warm_cache is None:
            return super()._solve_ppc(ppc, context=context)

        problem = CanonicalPowerFlowProblem(
            base_mva=float(ppc["baseMVA"]),
            bus=ppc["bus"],
            branch=ppc["branch"],
            gen=ppc["gen"],
        )
        physics_fingerprint = self.physics_config.fingerprint()
        warm_start = warm_cache.lookup(
            problem,
            physics_fingerprint=physics_fingerprint,
        )

        used_warm_start = False
        original_vm: np.ndarray | None = None
        original_va: np.ndarray | None = None
        if warm_start is not None:
            distance = warm_start.distance_to(problem)
            if not np.isfinite(distance) or distance > _MAX_WARM_START_DISTANCE:
                self._warm_start_distance_rejections += 1
            else:
                bus = np.asarray(ppc["bus"])
                original_vm = np.asarray(bus[:, VM], dtype=np.float64).copy()
                original_va = np.asarray(bus[:, VA], dtype=np.float64).copy()
                used_warm_start = bool(warm_start.apply_to_ppc(ppc))
                if used_warm_start:
                    self._warm_start_applications += 1

        try:
            result_ppc, metrics = super()._solve_ppc(ppc, context=context)
        except PowerFlowNotConverged:
            if not used_warm_start or original_vm is None or original_va is None:
                raise

            # Warm start is only an optimization. If it changes convergence,
            # restore the canonical solver starting point and retry once.
            self._warm_start_fallbacks += 1
            bus = np.asarray(ppc["bus"])
            bus[:, VM] = original_vm
            bus[:, VA] = original_va
            result_ppc, metrics = super()._solve_ppc(ppc, context=context)

        warm_cache.store_success(
            problem,
            result_ppc,
            physics_fingerprint=physics_fingerprint,
        )
        return result_ppc, metrics


def build_memory_mapped_teacher_context(
    *,
    runtime_store_dir: str | Path,
    states_dir: str | Path,
    task_config: Mapping[str, Any],
    scenario_ids: Sequence[int],
    memory_registry=None,
) -> dict[str, Any]:
    """Build the mmap context, swapping in warm-start backend only if enabled."""

    context = _build_memory_mapped_teacher_context(
        runtime_store_dir=runtime_store_dir,
        states_dir=states_dir,
        task_config=task_config,
        scenario_ids=scenario_ids,
        memory_registry=memory_registry,
    )
    if not warm_start_enabled_from_environment():
        return context

    previous_backend = context["backend"]
    context["backend"] = WarmStartMemoryMappedGridFMPowerFlowBackend(
        adapter=context["adapter"],
        physics_config=context["physics_config"],
        enable_cache=bool(previous_backend.enable_cache),
        store_raw_result=bool(previous_backend.store_raw_result),
        exact_cache_max_bytes=int(previous_backend.cache_info()["max_bytes"]),
    )
    return context
