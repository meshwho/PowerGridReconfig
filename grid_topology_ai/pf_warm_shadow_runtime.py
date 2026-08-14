from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
import zlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pypower.idx_bus import BUS_I, VA, VM

from grid_topology_ai.data_adapter import BUS_FEATURE_COLUMNS
from grid_topology_ai.pf_warm_shadow import (
    WarmCandidate,
    WarmStartDescriptor,
    WarmStartShadow,
    _key,
    _pack_descriptor,
    _unpack_descriptor,
    warm_start_distance,
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


_SCHEMA_VERSION = 2
_PAYLOAD_VERSION = 1
_WARM_SEED_PAYLOAD_VERSION = 1
_DEFAULT_MAX_PAYLOAD_BYTES = 1024**3
_COMPRESSION_LEVEL = 1
_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}


@dataclass(frozen=True)
class _WarmSeed:
    bus_ids: np.ndarray | None
    vm: np.ndarray
    va: np.ndarray


def _compact_warm_state_payload(payload: bytes) -> bytes:
    """Reduce a full exact-state payload to the values required by a warm start."""

    raw = bytes(payload)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if "warm_payload_version" in data.files:
                version = int(
                    np.asarray(data["warm_payload_version"]).reshape(-1)[0]
                )
                if version != _WARM_SEED_PAYLOAD_VERSION:
                    raise ValueError("unsupported warm-seed payload version")
                return raw

            bus_features = np.asarray(data["bus_features"], dtype=np.float64)
            if bus_features.ndim != 2:
                raise ValueError("cached bus features must be two-dimensional")
            vm = np.asarray(
                bus_features[:, _BUS_COL["Vm"]], dtype=np.float64
            )
            va = np.asarray(
                bus_features[:, _BUS_COL["Va"]], dtype=np.float64
            )
            has_bus_ids = bool(
                int(np.asarray(data["has_bus_ids"]).reshape(-1)[0])
            )
            bus_ids = (
                np.asarray(data["bus_ids"], dtype=np.int64)
                if has_bus_ids
                else np.empty(0, dtype=np.int64)
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid exact-state payload for warm cache") from exc

    if not np.isfinite(vm).all() or not np.isfinite(va).all():
        raise ValueError("warm seed contains non-finite voltage values")
    if has_bus_ids and len(bus_ids) != len(vm):
        raise ValueError("warm seed bus ID count mismatch")

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        warm_payload_version=np.asarray(
            [_WARM_SEED_PAYLOAD_VERSION], dtype=np.int16
        ),
        has_bus_ids=np.asarray([has_bus_ids], dtype=np.uint8),
        bus_ids=bus_ids,
        vm=vm,
        va=va,
    )
    return buffer.getvalue()


def _decode_warm_seed(payload: bytes) -> _WarmSeed:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            version = int(
                np.asarray(data["warm_payload_version"]).reshape(-1)[0]
            )
            if version != _WARM_SEED_PAYLOAD_VERSION:
                raise ValueError("unsupported warm-seed payload version")
            has_bus_ids = bool(
                int(np.asarray(data["has_bus_ids"]).reshape(-1)[0])
            )
            vm = np.asarray(data["vm"], dtype=np.float64).copy()
            va = np.asarray(data["va"], dtype=np.float64).copy()
            bus_ids = (
                np.asarray(data["bus_ids"], dtype=np.int64).copy()
                if has_bus_ids
                else None
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid warm-seed payload") from exc

    if vm.ndim != 1 or va.ndim != 1 or len(vm) != len(va):
        raise ValueError("warm seed voltage arrays must have matching lengths")
    if not np.isfinite(vm).all() or not np.isfinite(va).all():
        raise ValueError("warm seed contains non-finite voltage values")
    if bus_ids is not None and len(bus_ids) != len(vm):
        raise ValueError("warm seed bus ID count mismatch")
    return _WarmSeed(bus_ids=bus_ids, vm=vm, va=va)


class BoundedWarmStartStore:
    """Process-safe compact warm bank with a hard compressed-byte budget."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_candidates_per_topology: int = 16,
        max_shadow_records: int = 50_000,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        timeout: float = 30.0,
    ) -> None:
        self.root = Path(root).expanduser()
        self.directory = self.root / "warm"
        self.database_path = self.directory / "cache_v2.sqlite3"
        self.max_candidates_per_topology = int(max_candidates_per_topology)
        self.max_shadow_records = int(max_shadow_records)
        self.max_payload_bytes = int(max_payload_bytes)

        if self.max_candidates_per_topology <= 0:
            raise ValueError("max_candidates_per_topology must be positive")
        if self.max_shadow_records <= 0:
            raise ValueError("max_shadow_records must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")

        self.directory.mkdir(parents=True, exist_ok=True)
        new_database = not self.database_path.exists()
        self._db = sqlite3.connect(self.database_path, timeout=float(timeout))
        if new_database:
            self._db.execute("PRAGMA auto_vacuum=FULL")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA wal_autocheckpoint=1000")
        self._create_schema()

    def _create_schema(self) -> None:
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                exact_key TEXT PRIMARY KEY,
                topology_key TEXT NOT NULL,
                descriptor BLOB NOT NULL,
                state_payload BLOB NOT NULL,
                state_sha256 TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS candidates_topology "
            "ON candidates(topology_key, created_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS candidates_created "
            "ON candidates(created_at, exact_key)"
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_exact_key TEXT NOT NULL,
                topology_key TEXT NOT NULL,
                candidate_exact_key TEXT NOT NULL,
                distance REAL NOT NULL,
                scenario_id INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._db.execute(
            "INSERT OR IGNORE INTO metadata(name, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._db.execute(
            "INSERT OR IGNORE INTO metadata(name, value) "
            "VALUES('candidate_payload_bytes', '0')"
        )
        row = self._db.execute(
            "SELECT COALESCE(SUM(payload_bytes), 0) FROM candidates"
        ).fetchone()
        total = 0 if row is None else int(row[0])
        self._db.execute(
            "UPDATE metadata SET value = ? WHERE name = 'candidate_payload_bytes'",
            (str(total),),
        )
        self._db.commit()
        row = self._db.execute(
            "SELECT value FROM metadata WHERE name='schema_version'"
        ).fetchone()
        if row is None or int(row[0]) != _SCHEMA_VERSION:
            raise ValueError("unsupported warm-cache schema version")

    def _payload_total_locked(self) -> int:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE name = 'candidate_payload_bytes'"
        ).fetchone()
        if row is None:
            raise RuntimeError("warm-cache payload counter is missing")
        return int(row[0])

    def _set_payload_total_locked(self, value: int) -> None:
        self._db.execute(
            "UPDATE metadata SET value = ? WHERE name = 'candidate_payload_bytes'",
            (str(max(int(value), 0)),),
        )

    def _evict_locked(self, topology_key: str, total: int) -> int:
        rows = self._db.execute(
            """
            SELECT exact_key, payload_bytes
            FROM candidates
            WHERE topology_key = ?
            ORDER BY created_at DESC, exact_key DESC
            """,
            (topology_key,),
        ).fetchall()
        for exact_key, payload_bytes in rows[self.max_candidates_per_topology :]:
            self._db.execute(
                "DELETE FROM candidates WHERE exact_key = ?",
                (str(exact_key),),
            )
            total -= int(payload_bytes)

        while total > self.max_payload_bytes:
            victim = self._db.execute(
                """
                SELECT exact_key, payload_bytes
                FROM candidates
                ORDER BY created_at ASC, exact_key ASC
                LIMIT 1
                """
            ).fetchone()
            if victim is None:
                break
            self._db.execute(
                "DELETE FROM candidates WHERE exact_key = ?",
                (str(victim[0]),),
            )
            total -= int(victim[1])

        return max(total, 0)

    def put(
        self,
        *,
        exact_key: str,
        topology_key: str,
        descriptor: WarmStartDescriptor,
        state_payload: bytes,
    ) -> bool:
        exact_key = _key(exact_key)
        topology_key = _key(topology_key)
        compact_payload = _compact_warm_state_payload(state_payload)
        packed_descriptor = zlib.compress(
            _pack_descriptor(descriptor), level=_COMPRESSION_LEVEL
        )
        payload_bytes = len(packed_descriptor) + len(compact_payload)
        if payload_bytes > self.max_payload_bytes:
            return False

        self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = self._db.execute(
                "SELECT 1 FROM candidates WHERE exact_key = ?",
                (exact_key,),
            ).fetchone()
            if existing is not None:
                self._db.rollback()
                return False

            self._db.execute(
                """
                INSERT INTO candidates(
                    exact_key,
                    topology_key,
                    descriptor,
                    state_payload,
                    state_sha256,
                    payload_version,
                    payload_bytes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exact_key,
                    topology_key,
                    sqlite3.Binary(packed_descriptor),
                    sqlite3.Binary(compact_payload),
                    hashlib.sha256(compact_payload).hexdigest(),
                    _PAYLOAD_VERSION,
                    payload_bytes,
                    float(time.time()),
                ),
            )
            total = self._payload_total_locked() + payload_bytes
            total = self._evict_locked(topology_key, total)
            self._set_payload_total_locked(total)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        row = self._db.execute(
            "SELECT 1 FROM candidates WHERE exact_key = ?",
            (exact_key,),
        ).fetchone()
        return row is not None

    def nearest(
        self,
        *,
        topology_key: str,
        descriptor: WarmStartDescriptor,
        exclude_exact_key: str | None = None,
    ) -> WarmCandidate | None:
        topology_key = _key(topology_key)
        excluded = None if exclude_exact_key is None else _key(exclude_exact_key)
        rows = self._db.execute(
            """
            SELECT exact_key, descriptor, state_payload, state_sha256,
                   payload_version
            FROM candidates
            WHERE topology_key = ?
            ORDER BY created_at DESC, exact_key
            LIMIT ?
            """,
            (topology_key, self.max_candidates_per_topology),
        ).fetchall()

        best: WarmCandidate | None = None
        for exact_key, packed, state_payload, state_sha, version in rows:
            exact_key = str(exact_key)
            if exact_key == excluded or int(version) != _PAYLOAD_VERSION:
                continue
            payload = bytes(state_payload)
            if hashlib.sha256(payload).hexdigest() != str(state_sha):
                continue
            try:
                candidate_descriptor = _unpack_descriptor(
                    zlib.decompress(bytes(packed))
                )
            except (KeyError, OSError, TypeError, ValueError, zlib.error):
                continue
            distance = warm_start_distance(descriptor, candidate_descriptor)
            if np.isfinite(distance) and (
                best is None or distance < best.distance
            ):
                best = WarmCandidate(
                    exact_key=exact_key,
                    topology_key=topology_key,
                    distance=float(distance),
                    state_payload=payload,
                )
        return best

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

    def counts(self) -> tuple[int, int]:
        candidates = int(
            self._db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        )
        records = int(
            self._db.execute("SELECT COUNT(*) FROM shadow_records").fetchone()[0]
        )
        return candidates, records

    def storage_info(self) -> dict[str, int]:
        row = self._db.execute("SELECT COUNT(*) FROM candidates").fetchone()
        payload_row = self._db.execute(
            "SELECT value FROM metadata WHERE name = 'candidate_payload_bytes'"
        ).fetchone()
        database_bytes = 0
        for path in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            try:
                database_bytes += int(path.stat().st_size)
            except OSError:
                pass
        return {
            "candidates": 0 if row is None else int(row[0]),
            "payload_bytes": 0 if payload_row is None else int(payload_row[0]),
            "max_payload_bytes": int(self.max_payload_bytes),
            "database_bytes": database_bytes,
        }

    def close(self) -> None:
        self._db.close()


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

    @staticmethod
    def _apply_seed(ppc: dict[str, Any], seed: _WarmSeed) -> None:
        bus = np.asarray(ppc["bus"])
        if seed.bus_ids is None:
            if len(bus) != len(seed.vm):
                raise ValueError("warm seed bus count mismatch")
            bus[:, VM] = seed.vm
            bus[:, VA] = seed.va
            return

        by_id = {
            int(bus_id): index
            for index, bus_id in enumerate(seed.bus_ids)
        }
        ppc_bus_ids = np.rint(bus[:, BUS_I]).astype(np.int64)
        try:
            positions = np.asarray(
                [by_id[int(bus_id)] for bus_id in ppc_bus_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("warm seed bus IDs do not match request") from exc
        bus[:, VM] = seed.vm[positions]
        bus[:, VA] = seed.va[positions]

    def _seed_diagnostics(
        self,
        ppc: dict[str, Any],
        seed: _WarmSeed,
    ) -> dict[str, object]:
        request_bus = np.asarray(ppc["bus"], dtype=np.float64)
        seeded_ppc = {
            "bus": np.array(request_bus, dtype=np.float64, copy=True),
        }
        self._apply_seed(seeded_ppc, seed)
        seeded_bus = np.asarray(seeded_ppc["bus"], dtype=np.float64)

        vm_delta = self._delta(request_bus[:, VM], seeded_bus[:, VM])
        va_delta = self._delta(request_bus[:, VA], seeded_bus[:, VA])
        return {
            "max_initial_vm_delta_pu": float(vm_delta),
            "max_initial_va_delta_deg": float(va_delta),
            "initial_seed_distinct": bool(vm_delta > 0.0 or va_delta > 0.0),
        }

    def _shadow_state(self, state, ppc, frames, candidate: WarmCandidate):
        seed = _decode_warm_seed(candidate.state_payload)
        seed_diagnostics = self._seed_diagnostics(ppc, seed)
        workload_before = self._workload_snapshot()
        started = perf_counter()

        try:
            warm_ppc = deepcopy(ppc)
            self._apply_seed(warm_ppc, seed)
            validate_ppc_input(
                warm_ppc,
                self.backend.physics_config,
                context="cross-scenario warm shadow",
            )
            result_ppc, success = runpf(
                warm_ppc,
                self.backend._build_pp_options(),
            )
            if not bool(success):
                raise RuntimeError("warm shadow did not converge")
            validate_pypower_result(
                result_ppc,
                self.backend.physics_config,
                input_ppc=warm_ppc,
                context="cross-scenario warm shadow",
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
