from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pypower.idx_bus import VA, VM

from grid_topology_ai.pf_warm_shadow import (
    PersistentWarmStartStore,
    WarmCandidate,
    WarmStartShadow,
    _key,
)
from grid_topology_ai.pypower_compat import get_power_flow_workload_counters


class BoundedWarmStartStore(PersistentWarmStartStore):
    """Warm-state store with a process-safe global shadow-record limit."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_candidates_per_topology: int = 16,
        max_shadow_records: int = 50_000,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            root,
            max_candidates_per_topology=max_candidates_per_topology,
            timeout=timeout,
        )
        self.max_shadow_records = int(max_shadow_records)
        if self.max_shadow_records <= 0:
            raise ValueError("max_shadow_records must be positive")

    def shadow_limit_reached(self) -> bool:
        row = self._db.execute("SELECT COUNT(*) FROM shadow_records").fetchone()
        count = 0 if row is None else int(row[0])
        return count >= self.max_shadow_records

    def record_shadow(
        self,
        *,
        request_exact_key: str,
        topology_key: str,
        candidate: WarmCandidate,
        scenario_id: int,
        record: dict[str, object],
    ) -> bool:
        request_key = _key(request_exact_key)
        topology_key = _key(topology_key)

        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                "SELECT COUNT(*) FROM shadow_records"
            ).fetchone()
            count = 0 if row is None else int(row[0])
            if count >= self.max_shadow_records:
                self._db.rollback()
                return False

            self._db.execute(
                """
                INSERT INTO shadow_records(
                    request_exact_key,
                    topology_key,
                    candidate_exact_key,
                    distance,
                    scenario_id,
                    record_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_key,
                    topology_key,
                    candidate.exact_key,
                    float(candidate.distance),
                    int(scenario_id),
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    float(time.time()),
                ),
            )
            self._db.commit()
            return True
        except Exception:
            self._db.rollback()
            raise


class BoundedWarmStartShadow(WarmStartShadow):
    store: BoundedWarmStartStore

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._authoritative_diagnostics: dict[str, object] = {}
        self._shadow_diagnostics: dict[str, object] = {}

    @staticmethod
    def _workload_snapshot() -> dict[str, object]:
        return get_power_flow_workload_counters()

    @staticmethod
    def _counter_delta(
        before: dict[str, object],
        after: dict[str, object],
        name: str,
    ) -> float:
        return float(after.get(name, 0.0)) - float(before.get(name, 0.0))

    def install(self) -> None:
        if self._run is not None:
            return

        super().install()
        original_run = self._run
        if original_run is None:
            raise RuntimeError("warm shadow did not capture the authoritative PF path")

        def measured_run(*args, **kwargs):
            workload_before = self._workload_snapshot()
            warm_before = int(getattr(self.backend, "warm_start_hits", 0))
            cold_before = int(getattr(self.backend, "cold_start_misses", 0))
            started = perf_counter()
            try:
                return original_run(*args, **kwargs)
            finally:
                elapsed = perf_counter() - started
                workload_after = self._workload_snapshot()
                warm_after = int(getattr(self.backend, "warm_start_hits", 0))
                cold_after = int(getattr(self.backend, "cold_start_misses", 0))
                self._authoritative_diagnostics = {
                    "authoritative_path_seconds": float(elapsed),
                    "authoritative_stock_runpf_calls": int(
                        self._counter_delta(
                            workload_before,
                            workload_after,
                            "stock_runpf_calls",
                        )
                    ),
                    "authoritative_q_limit_resolves": int(
                        self._counter_delta(
                            workload_before,
                            workload_after,
                            "q_limit_resolves",
                        )
                    ),
                    "authoritative_stock_runpf_seconds": float(
                        self._counter_delta(
                            workload_before,
                            workload_after,
                            "stock_runpf_seconds",
                        )
                    ),
                    "authoritative_used_legacy_warm_start": bool(
                        warm_after > warm_before
                    ),
                    "authoritative_used_cold_start": bool(
                        cold_after > cold_before
                    ),
                }

        self._run = measured_run

    def _sample(self, exact_key: str) -> bool:
        try:
            if self.store.shadow_limit_reached():
                return False
        except sqlite3.Error:
            return False
        return super()._sample(exact_key)

    def _seed_diagnostics(
        self,
        ppc: dict[str, Any],
        warm_state: Any,
    ) -> dict[str, object]:
        request_bus = np.asarray(ppc["bus"], dtype=np.float64)
        seeded_ppc = {
            "bus": np.array(request_bus, dtype=np.float64, copy=True),
        }
        self._seed(seeded_ppc, warm_state)
        seeded_bus = np.asarray(seeded_ppc["bus"], dtype=np.float64)

        vm_delta = self._delta(request_bus[:, VM], seeded_bus[:, VM])
        va_delta = self._delta(request_bus[:, VA], seeded_bus[:, VA])
        return {
            "max_initial_vm_delta_pu": float(vm_delta),
            "max_initial_va_delta_deg": float(va_delta),
            "initial_seed_distinct": bool(vm_delta > 0.0 or va_delta > 0.0),
        }

    def _shadow_state(self, state, ppc, frames, candidate: WarmCandidate):
        warm_state = self.backend._deserialize_exact_state(
            candidate.state_payload,
            state,
        )
        seed_diagnostics = self._seed_diagnostics(ppc, warm_state)
        workload_before = self._workload_snapshot()
        started = perf_counter()

        try:
            return super()._shadow_state(state, ppc, frames, candidate)
        finally:
            elapsed = perf_counter() - started
            workload_after = self._workload_snapshot()
            self._shadow_diagnostics = {
                **seed_diagnostics,
                "shadow_path_seconds": float(elapsed),
                "shadow_stock_runpf_calls": int(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "stock_runpf_calls",
                    )
                ),
                "shadow_q_limit_resolves": int(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "q_limit_resolves",
                    )
                ),
                "shadow_stock_runpf_seconds": float(
                    self._counter_delta(
                        workload_before,
                        workload_after,
                        "stock_runpf_seconds",
                    )
                ),
            }

    def _compare(self, authoritative, shadow) -> dict[str, object]:
        record = super()._compare(authoritative, shadow)
        record.update(self._authoritative_diagnostics)
        record.update(self._shadow_diagnostics)
        return record


def install_runtime_warm_shadow(
    backend: Any,
    cache_root: str | Path,
    *,
    sample_rate: float,
    max_pairs: int,
    max_candidates_per_topology: int,
) -> BoundedWarmStartShadow:
    store = BoundedWarmStartStore(
        cache_root,
        max_candidates_per_topology=max_candidates_per_topology,
        max_shadow_records=max_pairs,
    )
    shadow = BoundedWarmStartShadow(
        backend,
        store,
        sample_rate=sample_rate,
    )
    shadow.install()
    return shadow
