from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from grid_topology_ai.pf_warm_shadow import (
    PersistentWarmStartStore,
    WarmCandidate,
    WarmStartShadow,
    _key,
)


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

    def _sample(self, exact_key: str) -> bool:
        try:
            if self.store.shadow_limit_reached():
                return False
        except sqlite3.Error:
            return False
        return super()._sample(exact_key)


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
