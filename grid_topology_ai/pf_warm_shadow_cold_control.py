from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

from grid_topology_ai.pf_warm_shadow import WarmCandidate
from grid_topology_ai.pf_warm_shadow_runtime import (
    BoundedWarmStartShadow,
    BoundedWarmStartStore,
)
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.pypower_compat import (
    get_power_flow_workload_counters,
    runpf,
)


class ColdControlWarmStartShadow(BoundedWarmStartShadow):
    """Temporary paired cold-vs-global-warm validation probe.

    The teacher result remains authoritative.  This class only adds a second,
    cache-free reference solve from the request PPC and compares that result
    with the global warm-start shadow solve.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cold_reference_state: Any | None = None
        self._cold_diagnostics: dict[str, object] = {}

    @staticmethod
    def _counter_delta(
        before: dict[str, object],
        after: dict[str, object],
        name: str,
    ) -> float:
        return float(after.get(name, 0.0)) - float(before.get(name, 0.0))

    def _cold_state(self, state, ppc, frames):
        workload_before = get_power_flow_workload_counters()
        started = perf_counter()

        try:
            cold_ppc = deepcopy(ppc)
            validate_ppc_input(
                cold_ppc,
                self.backend.physics_config,
                context="cross-scenario warm shadow cold control",
            )
            result_ppc, success = runpf(
                cold_ppc,
                self.backend._build_pp_options(),
            )
            if not bool(success):
                raise RuntimeError("cold-control shadow did not converge")

            validate_pypower_result(
                result_ppc,
                self.backend.physics_config,
                input_ppc=cold_ppc,
                context="cross-scenario warm shadow cold control",
            )
            metrics = calculate_physical_metrics_from_result(
                result_ppc,
                power_flow_converged=True,
                physics_config=self.backend.physics_config,
            )
            return self.backend._build_state_from_pypower_result_fast(
                scenario_id=int(state.scenario_id),
                result_ppc=result_ppc,
                previous_state=state,
                original_frames=frames,
                physical_metrics=metrics,
            )
        finally:
            elapsed = perf_counter() - started
            workload_after = get_power_flow_workload_counters()
            self._cold_diagnostics = {
                "comparison_reference": "canonical_cold",
                "cold_path_seconds": float(elapsed),
                "cold_stock_runpf_calls": int(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "stock_runpf_calls",
                    )
                ),
                "cold_q_limit_resolves": int(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "q_limit_resolves",
                    )
                ),
                "cold_stock_runpf_seconds": float(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "stock_runpf_seconds",
                    )
                ),
            }

    def _shadow_state(self, state, ppc, frames, candidate: WarmCandidate):
        self._cold_reference_state = None
        self._cold_diagnostics = {}

        cold_state = self._cold_state(state, ppc, frames)
        self._cold_reference_state = cold_state
        return super()._shadow_state(state, ppc, frames, candidate)

    def _compare(self, authoritative, shadow) -> dict[str, object]:
        cold_state = self._cold_reference_state
        if cold_state is None:
            raise RuntimeError("cold-control reference state is unavailable")

        # Deliberately compare the two diagnostic solves.  The authoritative
        # teacher result is still measured by the parent class but is not used
        # as the physical reference for this experiment.
        record = super()._compare(cold_state, shadow)
        record.update(self._cold_diagnostics)
        record["authoritative_used_for_comparison"] = False
        return record


def install_cold_control_warm_shadow(
    backend: Any,
    cache_root: str | Path,
    *,
    sample_rate: float,
    max_pairs: int,
    max_candidates_per_topology: int,
) -> ColdControlWarmStartShadow:
    store = BoundedWarmStartStore(
        cache_root,
        max_candidates_per_topology=max_candidates_per_topology,
        max_shadow_records=max_pairs,
    )
    shadow = ColdControlWarmStartShadow(
        backend,
        store,
        sample_rate=sample_rate,
    )
    shadow.install()
    return shadow
