from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, RATE_A
from pypower.idx_bus import BUS_I, VA, VM
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai._pypower_backend_core import *  # noqa: F401,F403
from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as _CoreGridFMPowerFlowBackend,
)
from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
)
from grid_topology_ai.pf_cache_identity import exact_pf_problem_fingerprint
from grid_topology_ai.pf_cache_store import PersistentPFCacheStore
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.power_flow_errors import (
    InvalidPhysicalState,
    PowerFlowNotConverged,
)
from grid_topology_ai.power_flow_state_builder import PowerFlowStateBuilder
from grid_topology_ai.pypower_compat import (
    get_power_flow_workload_counters,
    runpf,
)
from grid_topology_ai.topology_actions import GridFMAction


_BUS_COL = {name: index for index, name in enumerate(BUS_FEATURE_COLUMNS)}
_BRANCH_COL = {name: index for index, name in enumerate(BRANCH_FEATURE_COLUMNS)}
_GENERATOR_COLUMNS = (
    "bus",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
)
_TOPOLOGY_CACHE_MAX_ENTRIES = 8
_EXACT_CACHE_PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class _GeneratorOperatingPointState(GridFMState):
    """Solved state carrying the exact per-generator operating point."""

    generator_ids: np.ndarray | None = None
    generator_p_mw: np.ndarray | None = None
    generator_q_mvar: np.ndarray | None = None
    generator_status: np.ndarray | None = None


@dataclass(frozen=True)
class _TopologyCacheEntry:
    """One solved target topology together with its source generator state."""

    generator_ids: np.ndarray
    generator_p_mw: np.ndarray
    generator_q_mvar: np.ndarray
    generator_status: np.ndarray
    next_state: GridFMState


# Preserve the public module path used by pickled results and type displays.
GridFMPowerFlowResult.__module__ = __name__


class GridFMPowerFlowBackend(_CoreGridFMPowerFlowBackend):
    """PYPOWER backend with canonical initial and fast transition builders."""

    def __init__(
        self,
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        enable_cache: bool = True,
        store_raw_result: bool = False,
        persistent_cache_root: str | Path | None = None,
        persistent_cache_read_only: bool = False,
    ) -> None:
        self._require_matching_physics_contract(adapter, physics_config)
        super().__init__(
            adapter=adapter,
            physics_config=physics_config,
            enable_cache=enable_cache,
            store_raw_result=store_raw_result,
        )
        self._state_builder = PowerFlowStateBuilder(self.physics_config)
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0
        self.exact_cache_hits = 0
        self.tolerant_cache_hits = 0
        self.warm_start_hits = 0
        self.cold_start_misses = 0
        self.global_exact_cache_hits = 0
        self.persistent_exact_cache_hits = 0
        self.persistent_exact_cache_misses = 0
        self.persistent_exact_cache_writes = 0
        self.persistent_exact_cache_errors = 0
        self._global_exact_cache: dict[str, GridFMState] = {}
        self._persistent_exact_cache: PersistentPFCacheStore | None = None
        self._topology_cache: dict[tuple, list[_TopologyCacheEntry]] = {}
        self._pending_warm_start_state: GridFMState | None = None
        self._pending_warm_start_applied = False

        if self.enable_cache and persistent_cache_root is not None:
            self._persistent_exact_cache = PersistentPFCacheStore(
                persistent_cache_root,
                namespace="exact",
                read_only=bool(persistent_cache_read_only),
            )

    def performance_info(self) -> dict[str, object]:
        """Return backend-local cache and PYPOWER workload counters."""

        info = dict(self.cache_info())
        misses = int(info["misses"])
        stock_calls = int(self.stock_runpf_calls)
        persistent_entries = 0
        persistent_bytes = 0

        if self._persistent_exact_cache is not None:
            try:
                store_info = self._persistent_exact_cache.info()
            except (OSError, sqlite3.Error, ValueError):
                self.persistent_exact_cache_errors += 1
            else:
                persistent_entries = int(store_info.entries)
                persistent_bytes = int(store_info.database_bytes)

        info.update(
            {
                "exact_cache_hits": int(self.exact_cache_hits),
                "tolerant_cache_hits": int(self.tolerant_cache_hits),
                "warm_start_hits": int(self.warm_start_hits),
                "cold_start_misses": int(self.cold_start_misses),
                "global_exact_cache_hits": int(self.global_exact_cache_hits),
                "global_exact_cache_size": int(len(self._global_exact_cache)),
                "persistent_exact_cache_enabled": (
                    self._persistent_exact_cache is not None
                ),
                "persistent_exact_cache_hits": int(
                    self.persistent_exact_cache_hits
                ),
                "persistent_exact_cache_misses": int(
                    self.persistent_exact_cache_misses
                ),
                "persistent_exact_cache_writes": int(
                    self.persistent_exact_cache_writes
                ),
                "persistent_exact_cache_errors": int(
                    self.persistent_exact_cache_errors
                ),
                "persistent_exact_cache_entries": persistent_entries,
                "persistent_exact_cache_bytes": persistent_bytes,
                "topology_cache_buckets": int(len(self._topology_cache)),
                "topology_cache_entries": int(
                    sum(len(entries) for entries in self._topology_cache.values())
                ),
                "stock_runpf_calls": stock_calls,
                "q_limit_resolves": int(self.q_limit_resolves),
                "solves_per_cache_miss": (
                    float(stock_calls) / float(misses)
                    if misses > 0
                    else 0.0
                ),
            }
        )
        return info

    def reset_performance_counters(self) -> None:
        """Reset counters without discarding cached power-flow states."""

        self.cache_hits = 0
        self.cache_misses = 0
        self.stock_runpf_calls = 0
        self.q_limit_resolves = 0
        self.exact_cache_hits = 0
        self.tolerant_cache_hits = 0
        self.warm_start_hits = 0
        self.cold_start_misses = 0
        self.global_exact_cache_hits = 0
        self.persistent_exact_cache_hits = 0
        self.persistent_exact_cache_misses = 0
        self.persistent_exact_cache_writes = 0
        self.persistent_exact_cache_errors = 0

    def clear_cache(self) -> None:
        """Clear volatile caches without deleting persistent PF results."""

        super().clear_cache()
        self._global_exact_cache.clear()
        self._topology_cache.clear()
        self._pending_warm_start_state = None
        self._pending_warm_start_applied = False
        self.exact_cache_hits = 0
        self.tolerant_cache_hits = 0
        self.warm_start_hits = 0
        self.cold_start_misses = 0
        self.global_exact_cache_hits = 0
        self.persistent_exact_cache_hits = 0
        self.persistent_exact_cache_misses = 0
        self.persistent_exact_cache_writes = 0
        self.persistent_exact_cache_errors = 0

    def close(self) -> None:
        store = self._persistent_exact_cache
        self._persistent_exact_cache = None
        if store is not None:
            store.close()

    @staticmethod
    def _require_matching_physics_contract(
        adapter: GridFMAdapter,
        physics_config: PhysicsConfig,
    ) -> None:
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")

        adapter_config = getattr(adapter, "physics_config", None)
        if adapter_config is None:
            return
        if not isinstance(adapter_config, PhysicsConfig):
            raise TypeError("adapter.physics_config must be a PhysicsConfig.")
        if adapter_config.fingerprint() != physics_config.fingerprint():
            raise ValueError(
                "Adapter and power-flow backend must use the same "
                "PhysicsConfig fingerprint."
            )

    def _build_pp_options(self) -> dict[str, object]:
        options = super()._build_pp_options()
        options["OPF_VIOLATION"] = self.physics_config.generator_q_tolerance_mvar
        return options

    @staticmethod
    def _generator_operating_point(
        state: GridFMState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        values = (
            getattr(state, "generator_ids", None),
            getattr(state, "generator_p_mw", None),
            getattr(state, "generator_q_mvar", None),
            getattr(state, "generator_status", None),
        )

        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise InvalidPhysicalState(
                "Generator operating point is incomplete in GridFMState."
            )

        ids = np.asarray(values[0], dtype=np.int64)
        pg = np.asarray(values[1], dtype=np.float64)
        qg = np.asarray(values[2], dtype=np.float64)
        status = np.asarray(values[3], dtype=np.float64)

        if ids.ndim != 1:
            raise InvalidPhysicalState("Generator IDs must be one-dimensional.")
        if any(array.shape != ids.shape for array in (pg, qg, status)):
            raise InvalidPhysicalState(
                "Generator operating-point arrays must have matching shapes."
            )
        if np.unique(ids).size != ids.size:
            raise InvalidPhysicalState("Generator IDs must be unique.")
        if not np.isfinite(pg).all() or not np.isfinite(qg).all():
            raise InvalidPhysicalState(
                "Generator operating point contains NaN or infinity."
            )
        if not np.isfinite(status).all() or np.any(
            (status != 0.0) & (status != 1.0)
        ):
            raise InvalidPhysicalState(
                "Generator status must contain only 0 or 1."
            )

        return ids, pg, qg, status

    @staticmethod
    def _with_generator_operating_point(
        state: GridFMState,
        *,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
    ) -> GridFMState:
        gen_df = original_frames["gen"].sort_values("idx").reset_index(drop=True)
        gen_result = np.asarray(result_ppc["gen"])

        if gen_result.ndim != 2 or gen_result.shape[0] != len(gen_df):
            raise InvalidPhysicalState(
                "PYPOWER gen result does not match the source frame."
            )

        return _GeneratorOperatingPointState(
            scenario_id=state.scenario_id,
            load_scenario_idx=state.load_scenario_idx,
            bus_features=state.bus_features,
            branch_features=state.branch_features,
            edge_index=state.edge_index,
            branch_ids=state.branch_ids,
            branch_status=state.branch_status,
            metrics=state.metrics,
            outaged_branch_ids=state.outaged_branch_ids,
            bus_ids=state.bus_ids,
            generator_ids=gen_df["idx"].to_numpy(dtype=np.int64, copy=True),
            generator_p_mw=np.asarray(
                gen_result[:, PG], dtype=np.float64
            ).copy(),
            generator_q_mvar=np.asarray(
                gen_result[:, QG], dtype=np.float64
            ).copy(),
            generator_status=np.asarray(
                gen_result[:, GEN_STATUS], dtype=np.float64
            ).copy(),
        )

    def _power_flow_input_fingerprint(self, state: GridFMState) -> str:
        fingerprint = super()._power_flow_input_fingerprint(state)
        operating_point = self._generator_operating_point(state)

        if operating_point is None:
            return fingerprint

        digest = hashlib.sha256(fingerprint.encode("ascii"))
        for values in operating_point:
            self._hash_array(digest, values)
        return digest.hexdigest()

    def _topology_bucket_key(
        self,
        state: GridFMState,
        *,
        action: GridFMAction | None = None,
        switched_off_branch_id: int | None = None,
    ) -> tuple:
        """Return a topology key that intentionally excludes generator P/Q."""

        base_input_fingerprint = (
            _CoreGridFMPowerFlowBackend._power_flow_input_fingerprint(self, state)
        )
        return (
            self.physics_config.fingerprint(),
            int(state.scenario_id),
            base_input_fingerprint,
            self._resulting_topology_signature(
                state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            ),
        )

    def _select_topology_entry(
        self,
        bucket_key: tuple,
        state: GridFMState,
    ) -> tuple[_TopologyCacheEntry | None, _TopologyCacheEntry | None]:
        """Find a tolerant reuse first, otherwise the nearest warm-start state."""

        operating_point = self._generator_operating_point(state)
        if operating_point is None:
            return None, None

        ids, pg, qg, status = operating_point
        entries = list(self._topology_cache.get(bucket_key, ()))
        if not entries:
            return None, None

        p_tolerance = float(self.physics_config.generator_p_tolerance_mw)
        q_tolerance = float(self.physics_config.generator_q_tolerance_mvar)
        p_scale = max(p_tolerance, 1e-12)
        q_scale = max(q_tolerance, 1e-12)

        valid_entries: list[_TopologyCacheEntry] = []
        tolerant_entry: _TopologyCacheEntry | None = None
        warm_entry: _TopologyCacheEntry | None = None
        warm_distance = float("inf")

        for entry in entries:
            try:
                self._require_usable_next_state(entry.next_state)
            except InvalidPhysicalState:
                continue

            valid_entries.append(entry)
            if not np.array_equal(ids, entry.generator_ids):
                continue

            max_delta_p = float(np.max(np.abs(pg - entry.generator_p_mw)))
            max_delta_q = float(np.max(np.abs(qg - entry.generator_q_mvar)))
            same_status = np.array_equal(status, entry.generator_status)

            if (
                same_status
                and max_delta_p <= p_tolerance
                and max_delta_q <= q_tolerance
            ):
                tolerant_entry = entry
                break

            distance = max(max_delta_p / p_scale, max_delta_q / q_scale)
            if not same_status:
                distance += 1e12

            if distance < warm_distance:
                warm_distance = distance
                warm_entry = entry

        if len(valid_entries) != len(entries):
            if valid_entries:
                self._topology_cache[bucket_key] = valid_entries
            else:
                self._topology_cache.pop(bucket_key, None)

        if tolerant_entry is not None:
            return tolerant_entry, tolerant_entry
        return None, warm_entry

    def _remember_topology_result(
        self,
        bucket_key: tuple,
        source_state: GridFMState,
        next_state: GridFMState,
    ) -> None:
        operating_point = self._generator_operating_point(source_state)
        if operating_point is None:
            return

        ids, pg, qg, status = operating_point
        entry = _TopologyCacheEntry(
            generator_ids=ids.copy(),
            generator_p_mw=pg.copy(),
            generator_q_mvar=qg.copy(),
            generator_status=status.copy(),
            next_state=next_state,
        )

        entries = list(self._topology_cache.get(bucket_key, ()))
        for index, existing in enumerate(entries):
            if (
                np.array_equal(existing.generator_ids, entry.generator_ids)
                and np.array_equal(existing.generator_p_mw, entry.generator_p_mw)
                and np.array_equal(existing.generator_q_mvar, entry.generator_q_mvar)
                and np.array_equal(existing.generator_status, entry.generator_status)
            ):
                entries[index] = entry
                self._topology_cache[bucket_key] = entries
                return

        entries.append(entry)
        if len(entries) > _TOPOLOGY_CACHE_MAX_ENTRIES:
            entries = entries[-_TOPOLOGY_CACHE_MAX_ENTRIES:]
        self._topology_cache[bucket_key] = entries

    def _apply_pending_warm_start(self, ppc: dict[str, Any]) -> None:
        warm_state = self._pending_warm_start_state
        if warm_state is None:
            return

        bus = np.asarray(ppc["bus"])
        warm_features = np.asarray(warm_state.bus_features)
        if warm_features.ndim != 2 or bus.ndim != 2:
            return

        vm = np.asarray(
            warm_features[:, BUS_FEATURE_COLUMNS.index("Vm")],
            dtype=np.float64,
        )
        va = np.asarray(
            warm_features[:, BUS_FEATURE_COLUMNS.index("Va")],
            dtype=np.float64,
        )
        if not np.isfinite(vm).all() or not np.isfinite(va).all():
            return

        warm_bus_ids = getattr(warm_state, "bus_ids", None)
        if warm_bus_ids is None:
            if len(vm) != len(bus):
                return
            bus[:, VM] = vm
            bus[:, VA] = va
            self._pending_warm_start_applied = True
            return

        warm_bus_ids = np.asarray(warm_bus_ids, dtype=np.int64)
        if len(warm_bus_ids) != len(vm):
            return

        by_id = {
            int(bus_id): position
            for position, bus_id in enumerate(warm_bus_ids)
        }
        ppc_bus_ids = np.rint(bus[:, BUS_I]).astype(np.int64)
        try:
            positions = np.asarray(
                [by_id[int(bus_id)] for bus_id in ppc_bus_ids],
                dtype=np.int64,
            )
        except KeyError:
            return

        bus[:, VM] = vm[positions]
        bus[:, VA] = va[positions]
        self._pending_warm_start_applied = True

    @staticmethod
    def _json_cache_value(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported cache metadata value: {type(value).__name__}")

    @classmethod
    def _serialize_exact_state(cls, state: GridFMState) -> bytes:
        metrics = json.dumps(
            state.metrics,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=cls._json_cache_value,
        ).encode("utf-8")
        bus_ids = getattr(state, "bus_ids", None)
        operating_point = cls._generator_operating_point(state)

        values: dict[str, np.ndarray] = {
            "payload_version": np.asarray(
                [_EXACT_CACHE_PAYLOAD_VERSION], dtype=np.int16
            ),
            "bus_features": np.asarray(state.bus_features),
            "branch_features": np.asarray(state.branch_features),
            "edge_index": np.asarray(state.edge_index, dtype=np.int64),
            "branch_ids": np.asarray(state.branch_ids, dtype=np.int64),
            "branch_status": np.asarray(state.branch_status),
            "outaged_branch_ids": np.asarray(
                state.outaged_branch_ids, dtype=np.int64
            ),
            "metrics_json": np.frombuffer(metrics, dtype=np.uint8),
            "has_bus_ids": np.asarray([bus_ids is not None], dtype=np.uint8),
            "bus_ids": (
                np.asarray(bus_ids, dtype=np.int64)
                if bus_ids is not None
                else np.empty(0, dtype=np.int64)
            ),
            "has_generator_state": np.asarray(
                [operating_point is not None], dtype=np.uint8
            ),
        }

        if operating_point is None:
            values.update(
                {
                    "generator_ids": np.empty(0, dtype=np.int64),
                    "generator_p_mw": np.empty(0, dtype=np.float64),
                    "generator_q_mvar": np.empty(0, dtype=np.float64),
                    "generator_status": np.empty(0, dtype=np.float64),
                }
            )
        else:
            generator_ids, pg, qg, status = operating_point
            values.update(
                {
                    "generator_ids": generator_ids,
                    "generator_p_mw": pg,
                    "generator_q_mvar": qg,
                    "generator_status": status,
                }
            )

        buffer = io.BytesIO()
        np.savez(buffer, **values)
        return buffer.getvalue()

    @staticmethod
    def _id_positions(
        cached_ids: np.ndarray,
        requested_ids: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        cached = np.asarray(cached_ids, dtype=np.int64)
        requested = np.asarray(requested_ids, dtype=np.int64)
        if cached.ndim != 1 or requested.ndim != 1:
            raise InvalidPhysicalState(f"Cached {name} must be one-dimensional.")
        if len(cached) != len(requested):
            raise InvalidPhysicalState(f"Cached {name} length mismatch.")
        if np.unique(cached).size != cached.size:
            raise InvalidPhysicalState(f"Cached {name} contains duplicate IDs.")

        by_id = {int(value): index for index, value in enumerate(cached)}
        try:
            return np.asarray(
                [by_id[int(value)] for value in requested], dtype=np.int64
            )
        except KeyError as exc:
            raise InvalidPhysicalState(
                f"Cached {name} does not match the requested physical state."
            ) from exc

    @classmethod
    def _rebind_exact_state(
        cls,
        cached_state: GridFMState,
        request_state: GridFMState,
    ) -> GridFMState:
        cached_bus_ids = getattr(cached_state, "bus_ids", None)
        request_bus_ids = getattr(request_state, "bus_ids", None)

        if cached_bus_ids is None and request_bus_ids is None:
            if cached_state.bus_features.shape[0] != request_state.bus_features.shape[0]:
                raise InvalidPhysicalState("Cached bus count mismatch.")
            bus_features = np.asarray(cached_state.bus_features).copy()
            bus_ids = None
        elif cached_bus_ids is None or request_bus_ids is None:
            raise InvalidPhysicalState(
                "Cached and requested states disagree on bus ID metadata."
            )
        else:
            bus_positions = cls._id_positions(
                np.asarray(cached_bus_ids, dtype=np.int64),
                np.asarray(request_bus_ids, dtype=np.int64),
                name="bus IDs",
            )
            bus_features = np.asarray(cached_state.bus_features)[
                bus_positions
            ].copy()
            bus_ids = np.asarray(request_bus_ids, dtype=np.int64).copy()

        request_branch_ids = np.asarray(request_state.branch_ids, dtype=np.int64)
        branch_positions = cls._id_positions(
            np.asarray(cached_state.branch_ids, dtype=np.int64),
            request_branch_ids,
            name="branch IDs",
        )
        branch_features = np.asarray(cached_state.branch_features)[
            branch_positions
        ].copy()
        branch_status = np.asarray(cached_state.branch_status)[
            branch_positions
        ].copy()
        outaged_branch_ids = [
            int(branch_id)
            for branch_id, status in zip(request_branch_ids, branch_status)
            if float(status) <= 0.5
        ]

        operating_point = cls._generator_operating_point(cached_state)
        common = {
            "scenario_id": int(request_state.scenario_id),
            "load_scenario_idx": float(request_state.load_scenario_idx),
            "bus_features": bus_features,
            "branch_features": branch_features,
            "edge_index": np.asarray(request_state.edge_index, dtype=np.int64).copy(),
            "branch_ids": request_branch_ids.copy(),
            "branch_status": branch_status,
            "metrics": dict(cached_state.metrics),
            "outaged_branch_ids": outaged_branch_ids,
            "bus_ids": bus_ids,
        }

        if operating_point is None:
            return GridFMState(**common)

        generator_ids, pg, qg, status = operating_point
        order = np.argsort(generator_ids, kind="stable")
        return _GeneratorOperatingPointState(
            **common,
            generator_ids=generator_ids[order].copy(),
            generator_p_mw=pg[order].copy(),
            generator_q_mvar=qg[order].copy(),
            generator_status=status[order].copy(),
        )

    @classmethod
    def _deserialize_exact_state(
        cls,
        payload: bytes,
        request_state: GridFMState,
    ) -> GridFMState:
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                version = int(np.asarray(data["payload_version"]).reshape(-1)[0])
                if version != _EXACT_CACHE_PAYLOAD_VERSION:
                    raise ValueError("Unsupported exact PF cache payload version.")

                metrics_bytes = np.asarray(data["metrics_json"], dtype=np.uint8)
                metrics = json.loads(metrics_bytes.tobytes().decode("utf-8"))
                has_bus_ids = bool(
                    int(np.asarray(data["has_bus_ids"]).reshape(-1)[0])
                )
                has_generator_state = bool(
                    int(np.asarray(data["has_generator_state"]).reshape(-1)[0])
                )

                common = {
                    "scenario_id": int(request_state.scenario_id),
                    "load_scenario_idx": float(request_state.load_scenario_idx),
                    "bus_features": np.asarray(data["bus_features"]).copy(),
                    "branch_features": np.asarray(data["branch_features"]).copy(),
                    "edge_index": np.asarray(
                        data["edge_index"], dtype=np.int64
                    ).copy(),
                    "branch_ids": np.asarray(
                        data["branch_ids"], dtype=np.int64
                    ).copy(),
                    "branch_status": np.asarray(data["branch_status"]).copy(),
                    "metrics": dict(metrics),
                    "outaged_branch_ids": [
                        int(value)
                        for value in np.asarray(
                            data["outaged_branch_ids"], dtype=np.int64
                        ).tolist()
                    ],
                    "bus_ids": (
                        np.asarray(data["bus_ids"], dtype=np.int64).copy()
                        if has_bus_ids
                        else None
                    ),
                }

                if has_generator_state:
                    cached_state: GridFMState = _GeneratorOperatingPointState(
                        **common,
                        generator_ids=np.asarray(
                            data["generator_ids"], dtype=np.int64
                        ).copy(),
                        generator_p_mw=np.asarray(
                            data["generator_p_mw"], dtype=np.float64
                        ).copy(),
                        generator_q_mvar=np.asarray(
                            data["generator_q_mvar"], dtype=np.float64
                        ).copy(),
                        generator_status=np.asarray(
                            data["generator_status"], dtype=np.float64
                        ).copy(),
                    )
                else:
                    cached_state = GridFMState(**common)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid exact PF cache payload.") from exc

        return cls._rebind_exact_state(cached_state, request_state)

    def _exact_problem_fingerprint(
        self,
        ppc: dict[str, Any],
        frames: dict[str, pd.DataFrame],
    ) -> str:
        branch_ids = frames["branch"].sort_values("idx")["idx"].to_numpy(
            dtype=np.int64
        )
        generator_ids = frames["gen"].sort_values("idx")["idx"].to_numpy(
            dtype=np.int64
        )
        return exact_pf_problem_fingerprint(
            ppc,
            physics_fingerprint=self.physics_config.fingerprint(),
            branch_ids=branch_ids,
            generator_ids=generator_ids,
        )

    def _lookup_global_exact(
        self,
        cache_key: str,
        request_state: GridFMState,
    ) -> tuple[GridFMState | None, str | None]:
        cached = self._global_exact_cache.get(cache_key)
        if cached is not None:
            try:
                rebound = self._rebind_exact_state(cached, request_state)
                self._require_usable_next_state(rebound)
            except InvalidPhysicalState:
                self._global_exact_cache.pop(cache_key, None)
            else:
                return rebound, "memory"

        store = self._persistent_exact_cache
        if store is None:
            return None, None

        try:
            payload = store.get(
                cache_key,
                payload_version=_EXACT_CACHE_PAYLOAD_VERSION,
            )
        except (OSError, sqlite3.Error, ValueError):
            self.persistent_exact_cache_errors += 1
            return None, None

        if payload is None:
            self.persistent_exact_cache_misses += 1
            return None, None

        try:
            rebound = self._deserialize_exact_state(payload, request_state)
            self._require_usable_next_state(rebound)
        except (InvalidPhysicalState, ValueError):
            self.persistent_exact_cache_errors += 1
            return None, None

        self.persistent_exact_cache_hits += 1
        self._global_exact_cache[cache_key] = rebound
        return rebound, "persistent"

    def _remember_global_exact(
        self,
        cache_key: str,
        next_state: GridFMState,
    ) -> None:
        self._global_exact_cache[cache_key] = next_state
        store = self._persistent_exact_cache
        if store is None or store.read_only:
            return

        try:
            payload = self._serialize_exact_state(next_state)
            inserted = store.put(
                cache_key,
                payload,
                payload_version=_EXACT_CACHE_PAYLOAD_VERSION,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            self.persistent_exact_cache_errors += 1
            return

        if inserted:
            self.persistent_exact_cache_writes += 1

    def _make_cache_key_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> tuple:
        """Preserve the public legacy argument order."""

        return super()._make_cache_key_from_state(
            state=state,
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )

    def _build_ppc_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        """Build a case from the current topology and generator operating point."""

        if action is not None:
            switched_off_branch_id = None

        ppc, frames = super()._build_ppc_from_state(
            state=state,
            switched_off_branch_id=switched_off_branch_id,
            action=action,
        )

        operating_point = self._generator_operating_point(state)
        if operating_point is None:
            self._apply_pending_warm_start(ppc)
            return ppc, frames

        generator_ids, pg, qg, status = operating_point
        gen_df = frames["gen"].sort_values("idx").reset_index(drop=True).copy()
        frame_ids = gen_df["idx"].to_numpy(dtype=np.int64)

        if not np.array_equal(frame_ids, generator_ids):
            raise InvalidPhysicalState(
                "Generator operating point does not match scenario generator IDs."
            )

        gen_df["p_mw"] = pg
        gen_df["q_mvar"] = qg
        gen_df["in_service"] = status

        source_bus_df = self.adapter.bus_df[
            self.adapter.bus_df["scenario"] == int(state.scenario_id)
        ].copy()
        if source_bus_df.empty:
            raise InvalidPhysicalState(
                f"Scenario {state.scenario_id} not found in bus_data."
            )
        source_bus_df = source_bus_df.sort_values("bus").reset_index(drop=True)

        # Keep the original voltage-control setpoint while carrying forward the
        # solved generator P/Q/status from the parent topology state.
        ppc["gen"] = self._build_gen_matrix(gen_df, source_bus_df)
        frames["gen"] = gen_df
        self._apply_pending_warm_start(ppc)
        return ppc, frames

    def run_power_flow_from_state(
        self,
        state: GridFMState,
        switched_off_branch_id: int | None = None,
        *,
        action: GridFMAction | None = None,
    ) -> GridFMPowerFlowResult:
        """Run from a solved state using exact, tolerant, then warm cache reuse."""

        branch_id, target_status = self._resolve_branch_status_action(
            action=action,
            switched_off_branch_id=switched_off_branch_id,
        )
        effective_switched_off = branch_id if target_status == 0 else None

        bucket_key: tuple | None = None
        warm_entry: _TopologyCacheEntry | None = None
        tolerant_entry: _TopologyCacheEntry | None = None
        global_exact_key: str | None = None

        if self.enable_cache:
            exact_key = self._make_topology_cache_key_from_state(
                state=state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            )
            cached_next_state = self._cache.get(exact_key)
            if cached_next_state is not None:
                try:
                    self._require_usable_next_state(cached_next_state)
                except InvalidPhysicalState:
                    del self._cache[exact_key]
                else:
                    self.cache_hits += 1
                    self.exact_cache_hits += 1
                    return GridFMPowerFlowResult(
                        success=True,
                        scenario_id=int(state.scenario_id),
                        switched_off_branch_id=effective_switched_off,
                        next_state=cached_next_state,
                        raw_result=None,
                        message="Power flow converged. [cache hit]",
                        switched_branch_id=branch_id,
                        target_status=target_status,
                    )

            bucket_key = self._topology_bucket_key(
                state,
                action=action,
                switched_off_branch_id=switched_off_branch_id,
            )
            tolerant_entry, warm_entry = self._select_topology_entry(
                bucket_key,
                state,
            )

            try:
                key_ppc, key_frames = self._build_ppc_from_state(
                    state=state,
                    switched_off_branch_id=switched_off_branch_id,
                    action=action,
                )
                global_exact_key = self._exact_problem_fingerprint(
                    key_ppc,
                    key_frames,
                )
            except (InvalidPhysicalState, ValueError):
                global_exact_key = None
            else:
                global_cached, cache_level = self._lookup_global_exact(
                    global_exact_key,
                    state,
                )
                if global_cached is not None:
                    self.cache_hits += 1
                    self.exact_cache_hits += 1
                    self.global_exact_cache_hits += 1
                    message = (
                        "Power flow converged. [persistent exact cache hit]"
                        if cache_level == "persistent"
                        else "Power flow converged. [global exact cache hit]"
                    )
                    return GridFMPowerFlowResult(
                        success=True,
                        scenario_id=int(state.scenario_id),
                        switched_off_branch_id=effective_switched_off,
                        next_state=global_cached,
                        raw_result=None,
                        message=message,
                        switched_branch_id=branch_id,
                        target_status=target_status,
                    )

            if tolerant_entry is not None:
                self.cache_hits += 1
                self.tolerant_cache_hits += 1
                return GridFMPowerFlowResult(
                    success=True,
                    scenario_id=int(state.scenario_id),
                    switched_off_branch_id=effective_switched_off,
                    next_state=tolerant_entry.next_state,
                    raw_result=None,
                    message="Power flow converged. [tolerant cache hit]",
                    switched_branch_id=branch_id,
                    target_status=target_status,
                )

        self._pending_warm_start_state = (
            None if warm_entry is None else warm_entry.next_state
        )
        self._pending_warm_start_applied = False
        misses_before = int(self.cache_misses)

        try:
            result = super().run_power_flow_from_state(
                state=state,
                switched_off_branch_id=switched_off_branch_id,
                action=action,
            )
            warm_start_applied = bool(self._pending_warm_start_applied)
        finally:
            self._pending_warm_start_state = None
            self._pending_warm_start_applied = False

        if self.enable_cache and self.cache_misses > misses_before:
            if warm_start_applied:
                self.warm_start_hits += 1
            else:
                self.cold_start_misses += 1

        if (
            self.enable_cache
            and result.success
            and result.next_state is not None
        ):
            if global_exact_key is not None:
                self._remember_global_exact(
                    global_exact_key,
                    result.next_state,
                )

            if bucket_key is not None:
                self._remember_topology_result(
                    bucket_key,
                    source_state=state,
                    next_state=result.next_state,
                )

        return replace(
            result,
            switched_off_branch_id=effective_switched_off,
            switched_branch_id=branch_id,
            target_status=target_status,
        )

    def _solve_ppc(
        self,
        ppc: dict[str, Any],
        *,
        context: str,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        """
        Solve through this public module so monkeypatched ``runpf`` remains
        observable in tests and diagnostics after the implementation split.
        """

        validate_ppc_input(ppc, self.physics_config, context=context)
        before = get_power_flow_workload_counters()

        try:
            result_ppc, success = runpf(ppc, self._build_pp_options())
        finally:
            after = get_power_flow_workload_counters()
            self.stock_runpf_calls += max(
                int(after["stock_runpf_calls"])
                - int(before["stock_runpf_calls"]),
                0,
            )
            self.q_limit_resolves += max(
                int(after["q_limit_resolves"])
                - int(before["q_limit_resolves"]),
                0,
            )

        if not bool(success):
            raise PowerFlowNotConverged(
                f"PYPOWER power flow did not converge ({context})."
            )

        validate_pypower_result(
            result_ppc,
            self.physics_config,
            input_ppc=ppc,
            context=context,
        )
        metrics = calculate_physical_metrics_from_result(
            result_ppc,
            power_flow_converged=True,
            physics_config=self.physics_config,
        )
        return result_ppc, metrics

    def _build_state_from_pypower_result(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        state = self._build_canonical_state(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )
        return self._with_generator_operating_point(
            state,
            result_ppc=result_ppc,
            original_frames=original_frames,
        )

    def _build_state_from_pypower_result_fast(
        self,
        scenario_id: int,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None = None,
    ) -> GridFMState:
        """Build a repeated transition through the NumPy-oriented core path."""

        state = super()._build_state_from_pypower_result_fast(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            previous_state=previous_state,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )

        bus_features = state.bus_features.copy()
        self._update_fast_generator_features(
            bus_features=bus_features,
            result_ppc=result_ppc,
            previous_state=previous_state,
            original_frames=original_frames,
        )

        branch_features = state.branch_features.copy()
        branch_result = np.asarray(result_ppc["branch"])
        rate_a = np.asarray(branch_result[:, RATE_A], dtype=np.float64)
        status = np.asarray(branch_result[:, BR_STATUS], dtype=np.float64)

        if np.any(rate_a < 0.0):
            raise InvalidPhysicalState("Branch RATE_A must be non-negative.")

        branch_features[:, _BRANCH_COL["unlimited_rating"]] = (
            rate_a == 0.0
        ).astype(np.float32)

        if not np.isfinite(bus_features).all():
            raise InvalidPhysicalState(
                "Bus features cannot be represented in float32."
            )
        if not np.isfinite(branch_features).all():
            raise InvalidPhysicalState(
                "Branch features cannot be represented in float32."
            )

        rated = (status > 0.0) & (rate_a > 0.0)
        loading = branch_features[:, _BRANCH_COL["loading_percent"]]
        vm = np.asarray(result_ppc["bus"][:, VM], dtype=np.float64)

        metrics = dict(state.metrics)
        metrics["num_generators"] = int(len(original_frames["gen"]))
        metrics["mean_loading_percent"] = (
            float(np.mean(loading[rated]))
            if np.any(rated)
            else 0.0
        )
        metrics["min_vm_pu"] = float(np.min(vm))
        metrics["max_vm_pu"] = float(np.max(vm))

        state = replace(
            state,
            bus_features=bus_features,
            branch_features=branch_features,
            metrics=metrics,
            bus_ids=previous_state.bus_ids,
        )
        return self._with_generator_operating_point(
            state,
            result_ppc=result_ppc,
            original_frames=original_frames,
        )

    @staticmethod
    def _update_fast_generator_features(
        *,
        bus_features: np.ndarray,
        result_ppc: dict[str, Any],
        previous_state: GridFMState,
        original_frames: dict[str, pd.DataFrame],
    ) -> None:
        gen_df = original_frames["gen"]
        missing = set(_GENERATOR_COLUMNS) - set(gen_df.columns)
        if missing:
            raise InvalidPhysicalState(
                f"Generator data is missing schema columns: {sorted(missing)}."
            )

        gen_result = np.asarray(result_ppc["gen"])
        if gen_result.ndim != 2 or gen_result.shape[0] != len(gen_df):
            raise InvalidPhysicalState(
                "PYPOWER gen result does not match the source frame."
            )

        if previous_state.bus_ids is not None:
            bus_ids = np.asarray(previous_state.bus_ids, dtype=np.int64)
        else:
            bus_ids = original_frames["bus"]["bus"].to_numpy(dtype=np.int64)

        if len(bus_ids) != bus_features.shape[0]:
            raise InvalidPhysicalState(
                "Bus IDs do not match the source state."
            )

        bus_pos = {
            int(bus_id): position
            for position, bus_id in enumerate(bus_ids)
        }
        try:
            gen_bus_pos = np.asarray(
                [
                    bus_pos[int(bus_id)]
                    for bus_id in gen_df["bus"].to_numpy(dtype=np.int64)
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise InvalidPhysicalState(
                f"Generator references unknown bus {exc.args[0]}."
            ) from exc

        status = np.asarray(gen_result[:, GEN_STATUS], dtype=np.float64)
        if not np.isfinite(status).all():
            raise InvalidPhysicalState(
                "Generator in_service contains NaN or infinity."
            )
        if np.any((status != 0.0) & (status != 1.0)):
            raise InvalidPhysicalState(
                "Generator in_service must contain only 0 or 1."
            )

        pg = np.asarray(gen_result[:, PG], dtype=np.float64)
        qg = np.asarray(gen_result[:, QG], dtype=np.float64)
        p_min = gen_df["min_p_mw"].to_numpy(dtype=np.float64)
        p_max = gen_df["max_p_mw"].to_numpy(dtype=np.float64)
        q_min = gen_df["min_q_mvar"].to_numpy(dtype=np.float64)
        q_max = gen_df["max_q_mvar"].to_numpy(dtype=np.float64)

        active = status > 0.0
        numeric = np.column_stack((pg, qg, p_min, p_max, q_min, q_max))
        if active.any() and not np.isfinite(numeric[active]).all():
            raise InvalidPhysicalState(
                "Active generator limits and outputs must be finite."
            )

        n_bus = bus_features.shape[0]
        active_pos = gen_bus_pos[active]
        sums = {
            "Pg": np.zeros(n_bus, dtype=np.float64),
            "Qg": np.zeros(n_bus, dtype=np.float64),
            "gen_online_count": np.zeros(n_bus, dtype=np.float64),
            "gen_p_min_mw": np.zeros(n_bus, dtype=np.float64),
            "gen_p_max_mw": np.zeros(n_bus, dtype=np.float64),
            "gen_q_min_mvar": np.zeros(n_bus, dtype=np.float64),
            "gen_q_max_mvar": np.zeros(n_bus, dtype=np.float64),
            "gen_p_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
            "gen_q_limit_violation_count": np.zeros(n_bus, dtype=np.float64),
        }
        minima = {
            "gen_min_p_down_margin_mw": np.full(n_bus, np.inf),
            "gen_min_p_up_margin_mw": np.full(n_bus, np.inf),
            "gen_min_q_down_margin_mvar": np.full(n_bus, np.inf),
            "gen_min_q_up_margin_mvar": np.full(n_bus, np.inf),
        }

        if active.any():
            pg_a = pg[active]
            qg_a = qg[active]
            p_min_a = p_min[active]
            p_max_a = p_max[active]
            q_min_a = q_min[active]
            q_max_a = q_max[active]

            p_down = pg_a - p_min_a
            p_up = p_max_a - pg_a
            q_down = qg_a - q_min_a
            q_up = q_max_a - qg_a

            for name, values in (
                ("Pg", pg_a),
                ("Qg", qg_a),
                ("gen_p_min_mw", p_min_a),
                ("gen_p_max_mw", p_max_a),
                ("gen_q_min_mvar", q_min_a),
                ("gen_q_max_mvar", q_max_a),
            ):
                np.add.at(sums[name], active_pos, values)

            np.add.at(sums["gen_online_count"], active_pos, 1.0)
            np.add.at(
                sums["gen_p_limit_violation_count"],
                active_pos,
                ((p_down < 0.0) | (p_up < 0.0)).astype(np.float64),
            )
            np.add.at(
                sums["gen_q_limit_violation_count"],
                active_pos,
                ((q_down < 0.0) | (q_up < 0.0)).astype(np.float64),
            )
            for name, values in (
                ("gen_min_p_down_margin_mw", p_down),
                ("gen_min_p_up_margin_mw", p_up),
                ("gen_min_q_down_margin_mvar", q_down),
                ("gen_min_q_up_margin_mvar", q_up),
            ):
                np.minimum.at(minima[name], active_pos, values)

        available = sums["gen_online_count"] > 0.0
        for values in minima.values():
            values[~available] = 0.0

        sums.update(minima)
        sums["gen_available"] = available.astype(np.float64)
        sums["gen_p_down_margin_mw"] = (
            sums["Pg"] - sums["gen_p_min_mw"]
        )
        sums["gen_p_up_margin_mw"] = (
            sums["gen_p_max_mw"] - sums["Pg"]
        )
        sums["gen_q_down_margin_mvar"] = (
            sums["Qg"] - sums["gen_q_min_mvar"]
        )
        sums["gen_q_up_margin_mvar"] = (
            sums["gen_q_max_mvar"] - sums["Qg"]
        )

        for name, values in sums.items():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                feature = values.astype(np.float32)
            if not np.isfinite(feature).all():
                raise InvalidPhysicalState(
                    f"Bus feature {name} cannot be represented in float32."
                )
            bus_features[:, _BUS_COL[name]] = feature

    def _build_canonical_state(
        self,
        *,
        scenario_id: int,
        result_ppc: dict[str, Any],
        original_frames: dict[str, pd.DataFrame],
        physical_metrics: dict[str, object] | None,
    ) -> GridFMState:
        return self._state_builder.build(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )