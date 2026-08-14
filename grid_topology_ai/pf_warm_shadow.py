from __future__ import annotations

import hashlib
import io
import json
import math
import sqlite3
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pypower.idx_bus import BUS_I, PD, QD, VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.pf_cache_identity import topology_fingerprint
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.pypower_compat import runpf


_SCHEMA_VERSION = 1
_PAYLOAD_VERSION = 1
_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_COL = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}


@dataclass(frozen=True)
class WarmStartDescriptor:
    pd: np.ndarray
    qd: np.ndarray
    pg: np.ndarray
    qg: np.ndarray
    gen_status: np.ndarray


@dataclass(frozen=True)
class WarmCandidate:
    exact_key: str
    topology_key: str
    distance: float
    state_payload: bytes


def _key(value: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64:
        raise ValueError("cache key must be a full SHA-256 digest")
    bytes.fromhex(value)
    return value


def warm_start_descriptor(
    ppc: dict[str, Any],
    *,
    generator_ids: np.ndarray | None = None,
) -> WarmStartDescriptor:
    bus = np.asarray(ppc["bus"], dtype=np.float64)
    gen = np.asarray(ppc["gen"], dtype=np.float64)

    bus_ids = np.rint(bus[:, BUS_I]).astype(np.int64)
    if not np.allclose(bus[:, BUS_I], bus_ids) or np.unique(bus_ids).size != len(bus_ids):
        raise ValueError("invalid PYPOWER bus IDs")
    bus_order = np.argsort(bus_ids, kind="stable")

    if generator_ids is None:
        gen_order = np.arange(len(gen))
    else:
        ids = np.asarray(generator_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != len(gen) or np.unique(ids).size != len(ids):
            raise ValueError("invalid generator IDs")
        gen_order = np.argsort(ids, kind="stable")

    values = (
        bus[bus_order, PD],
        bus[bus_order, QD],
        gen[gen_order, PG],
        gen[gen_order, QG],
        gen[gen_order, GEN_STATUS],
    )
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("warm-start descriptor contains non-finite values")

    return WarmStartDescriptor(
        *(np.asarray(value, dtype=np.float64).copy() for value in values)
    )


def _scaled_max_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if not left.size:
        return 0.0
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0)
    return float(np.max(np.abs(left - right))) / scale


def warm_start_distance(
    left: WarmStartDescriptor,
    right: WarmStartDescriptor,
) -> float:
    if not np.array_equal(left.gen_status, right.gen_status):
        return float("inf")
    return max(
        _scaled_max_delta(left.pd, right.pd),
        _scaled_max_delta(left.qd, right.qd),
        _scaled_max_delta(left.pg, right.pg),
        _scaled_max_delta(left.qg, right.qg),
    )


def _pack_descriptor(value: WarmStartDescriptor) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        pd=value.pd,
        qd=value.qd,
        pg=value.pg,
        qg=value.qg,
        gen_status=value.gen_status,
    )
    return buffer.getvalue()


def _unpack_descriptor(payload: bytes) -> WarmStartDescriptor:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return WarmStartDescriptor(
            pd=np.asarray(data["pd"], dtype=np.float64).copy(),
            qd=np.asarray(data["qd"], dtype=np.float64).copy(),
            pg=np.asarray(data["pg"], dtype=np.float64).copy(),
            qg=np.asarray(data["qg"], dtype=np.float64).copy(),
            gen_status=np.asarray(data["gen_status"], dtype=np.float64).copy(),
        )


class PersistentWarmStartStore:
    def __init__(
        self,
        root: str | Path,
        *,
        max_candidates_per_topology: int = 16,
        timeout: float = 30.0,
    ) -> None:
        self.root = Path(root).expanduser()
        self.directory = self.root / "warm"
        self.database_path = self.directory / "cache.sqlite3"
        self.max_candidates_per_topology = int(max_candidates_per_topology)
        if self.max_candidates_per_topology <= 0:
            raise ValueError("max_candidates_per_topology must be positive")

        self.directory.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.database_path, timeout=float(timeout))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
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
                created_at REAL NOT NULL
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS candidates_topology "
            "ON candidates(topology_key, created_at DESC)"
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
        self._db.commit()
        row = self._db.execute(
            "SELECT value FROM metadata WHERE name='schema_version'"
        ).fetchone()
        if row is None or int(row[0]) != _SCHEMA_VERSION:
            raise ValueError("unsupported warm-cache schema version")

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
        payload = bytes(state_payload)
        cursor = self._db.execute(
            """
            INSERT OR IGNORE INTO candidates(
                exact_key, topology_key, descriptor, state_payload,
                state_sha256, payload_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exact_key,
                topology_key,
                sqlite3.Binary(_pack_descriptor(descriptor)),
                sqlite3.Binary(payload),
                hashlib.sha256(payload).hexdigest(),
                _PAYLOAD_VERSION,
                float(time.time()),
            ),
        )
        self._db.execute(
            """
            DELETE FROM candidates
            WHERE topology_key = ?
              AND exact_key NOT IN (
                SELECT exact_key FROM candidates
                WHERE topology_key = ?
                ORDER BY created_at DESC, exact_key
                LIMIT ?
              )
            """,
            (topology_key, topology_key, self.max_candidates_per_topology),
        )
        self._db.commit()
        return cursor.rowcount == 1

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
            SELECT exact_key, descriptor, state_payload, state_sha256, payload_version
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
                candidate_descriptor = _unpack_descriptor(bytes(packed))
            except (KeyError, OSError, TypeError, ValueError):
                continue
            distance = warm_start_distance(descriptor, candidate_descriptor)
            if math.isfinite(distance) and (
                best is None or distance < best.distance
            ):
                best = WarmCandidate(
                    exact_key=exact_key,
                    topology_key=topology_key,
                    distance=float(distance),
                    state_payload=payload,
                )
        return best

    def record_shadow(
        self,
        *,
        request_exact_key: str,
        topology_key: str,
        candidate: WarmCandidate,
        scenario_id: int,
        record: dict[str, object],
    ) -> None:
        self._db.execute(
            """
            INSERT INTO shadow_records(
                request_exact_key, topology_key, candidate_exact_key,
                distance, scenario_id, record_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _key(request_exact_key),
                _key(topology_key),
                candidate.exact_key,
                float(candidate.distance),
                int(scenario_id),
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False),
                float(time.time()),
            ),
        )
        self._db.commit()

    def counts(self) -> tuple[int, int]:
        candidates = int(self._db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        records = int(self._db.execute("SELECT COUNT(*) FROM shadow_records").fetchone()[0])
        return candidates, records

    def close(self) -> None:
        self._db.close()


class WarmStartShadow:
    def __init__(
        self,
        backend: Any,
        store: PersistentWarmStartStore,
        *,
        sample_rate: float = 0.05,
    ) -> None:
        self.backend = backend
        self.store = store
        self.sample_rate = float(sample_rate)
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        self._run: Callable[..., Any] | None = None

    def install(self) -> None:
        if self._run is not None:
            return
        self._run = self.backend.run_power_flow_from_state
        self.backend.run_power_flow_from_state = self.run_power_flow_from_state
        self.backend._warm_start_shadow = self

    def close(self) -> None:
        self.store.close()

    def _sample(self, exact_key: str) -> bool:
        if self.sample_rate <= 0.0:
            return False
        if self.sample_rate >= 1.0:
            return True
        return int(exact_key[:16], 16) / float((1 << 64) - 1) < self.sample_rate

    @staticmethod
    def _ids(frames: dict[str, Any], name: str) -> np.ndarray:
        return frames[name].sort_values("idx")["idx"].to_numpy(dtype=np.int64)

    def _prepare(self, state, switched_off_branch_id, action):
        ppc, frames = self.backend._build_ppc_from_state(
            state=state,
            switched_off_branch_id=switched_off_branch_id,
            action=action,
        )
        branch_ids = self._ids(frames, "branch")
        generator_ids = self._ids(frames, "gen")
        exact_key = self.backend._exact_problem_fingerprint(ppc, frames)
        topology_key = topology_fingerprint(ppc, branch_ids=branch_ids)
        descriptor = warm_start_descriptor(ppc, generator_ids=generator_ids)
        return ppc, frames, exact_key, topology_key, descriptor

    @staticmethod
    def _seed(ppc: dict[str, Any], state: GridFMState) -> None:
        bus = np.asarray(ppc["bus"])
        features = np.asarray(state.bus_features)
        vm = np.asarray(features[:, _BUS_COL["Vm"]], dtype=np.float64)
        va = np.asarray(features[:, _BUS_COL["Va"]], dtype=np.float64)
        state_ids = getattr(state, "bus_ids", None)

        if state_ids is None:
            if len(bus) != len(vm):
                raise ValueError("warm-state bus count mismatch")
            bus[:, VM], bus[:, VA] = vm, va
            return

        by_id = {
            int(bus_id): index
            for index, bus_id in enumerate(np.asarray(state_ids, dtype=np.int64))
        }
        try:
            order = [by_id[int(bus_id)] for bus_id in np.rint(bus[:, BUS_I]).astype(int)]
        except KeyError as exc:
            raise ValueError("warm-state bus IDs do not match request") from exc
        bus[:, VM] = vm[order]
        bus[:, VA] = va[order]

    def _shadow_state(self, state, ppc, frames, candidate: WarmCandidate):
        warm_state = self.backend._deserialize_exact_state(candidate.state_payload, state)
        warm_ppc = deepcopy(ppc)
        self._seed(warm_ppc, warm_state)
        validate_ppc_input(
            warm_ppc,
            self.backend.physics_config,
            context="cross-scenario warm shadow",
        )
        result_ppc, success = runpf(warm_ppc, self.backend._build_pp_options())
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

    @staticmethod
    def _delta(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return float("inf")
        return 0.0 if not left.size else float(np.max(np.abs(left - right)))

    def _compare(self, authoritative: GridFMState, shadow: GridFMState) -> dict[str, object]:
        left_bus = np.asarray(authoritative.bus_features)
        right_bus = np.asarray(shadow.bus_features)
        left_branch = np.asarray(authoritative.branch_features)
        right_branch = np.asarray(shadow.branch_features)

        left_op = self.backend._generator_operating_point(authoritative)
        right_op = self.backend._generator_operating_point(shadow)
        pg_delta = qg_delta = float("inf")
        same_gen_status = False
        if left_op is not None and right_op is not None and np.array_equal(left_op[0], right_op[0]):
            pg_delta = self._delta(left_op[1], right_op[1])
            qg_delta = self._delta(left_op[2], right_op[2])
            same_gen_status = np.array_equal(left_op[3], right_op[3])

        left_assessment = assess_physical_state(authoritative.metrics)
        right_assessment = assess_physical_state(shadow.metrics)
        return {
            "shadow_success": True,
            "max_vm_delta_pu": self._delta(left_bus[:, _BUS_COL["Vm"]], right_bus[:, _BUS_COL["Vm"]]),
            "max_va_delta_deg": self._delta(left_bus[:, _BUS_COL["Va"]], right_bus[:, _BUS_COL["Va"]]),
            "max_pg_delta_mw": pg_delta,
            "max_qg_delta_mvar": qg_delta,
            "max_branch_p_delta_mw": max(
                self._delta(left_branch[:, _BRANCH_COL["pf"]], right_branch[:, _BRANCH_COL["pf"]]),
                self._delta(left_branch[:, _BRANCH_COL["pt"]], right_branch[:, _BRANCH_COL["pt"]]),
            ),
            "max_branch_q_delta_mvar": max(
                self._delta(left_branch[:, _BRANCH_COL["qf"]], right_branch[:, _BRANCH_COL["qf"]]),
                self._delta(left_branch[:, _BRANCH_COL["qt"]], right_branch[:, _BRANCH_COL["qt"]]),
            ),
            "same_branch_status": bool(np.array_equal(authoritative.branch_status, shadow.branch_status)),
            "same_generator_status": bool(same_gen_status),
            "same_secure_classification": bool(
                left_assessment.physically_secure == right_assessment.physically_secure
            ),
            "same_hard_overload_classification": bool(
                left_assessment.hard_overload_free == right_assessment.hard_overload_free
            ),
        }

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: Any = None,
    ):
        if self._run is None:
            raise RuntimeError("warm shadow is not installed")

        prepared = candidate = None
        try:
            prepared = self._prepare(state, switched_off_branch_id, action)
            ppc, frames, exact_key, topology_key, descriptor = prepared
            if self._sample(exact_key):
                candidate = self.store.nearest(
                    topology_key=topology_key,
                    descriptor=descriptor,
                    exclude_exact_key=exact_key,
                )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            prepared = None

        authoritative = self._run(
            state,
            switched_off_branch_id,
            action=action,
        )
        if prepared is None:
            return authoritative

        ppc, frames, exact_key, topology_key, descriptor = prepared
        next_state = getattr(authoritative, "next_state", None)
        if bool(getattr(authoritative, "success", False)) and next_state is not None:
            try:
                self.store.put(
                    exact_key=exact_key,
                    topology_key=topology_key,
                    descriptor=descriptor,
                    state_payload=self.backend._serialize_exact_state(next_state),
                )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass

        if (
            candidate is None
            or next_state is None
            or not bool(getattr(authoritative, "success", False))
            or "cache hit" in str(getattr(authoritative, "message", "")).lower()
        ):
            return authoritative

        try:
            shadow = self._shadow_state(state, ppc, frames, candidate)
            record = self._compare(next_state, shadow)
        except Exception as exc:
            record = {
                "shadow_success": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }

        try:
            self.store.record_shadow(
                request_exact_key=exact_key,
                topology_key=topology_key,
                candidate=candidate,
                scenario_id=int(state.scenario_id),
                record=record,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            pass

        return authoritative
