from __future__ import annotations

import dataclasses
import functools
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


_MB = 1024.0 * 1024.0


def process_memory_snapshot() -> dict[str, float | None]:
    """Return process-memory counters without making psutil a hard dependency."""

    try:
        import psutil
    except Exception:
        return {
            "rss_mb": None,
            "private_mb": None,
            "uss_mb": None,
            "vms_mb": None,
        }

    try:
        process = psutil.Process()
        info = process.memory_info()
    except Exception:
        return {
            "rss_mb": None,
            "private_mb": None,
            "uss_mb": None,
            "vms_mb": None,
        }

    private_bytes = getattr(info, "private", None)
    uss_bytes = None
    try:
        full_info = process.memory_full_info()
    except Exception:
        full_info = None

    if full_info is not None:
        if private_bytes is None:
            private_bytes = getattr(full_info, "private", None)
        uss_bytes = getattr(full_info, "uss", None)

    def to_mb(value: Any) -> float | None:
        if value is None:
            return None
        return float(value) / _MB

    return {
        "rss_mb": to_mb(getattr(info, "rss", None)),
        "private_mb": to_mb(private_bytes),
        "uss_mb": to_mb(uss_bytes),
        "vms_mb": to_mb(getattr(info, "vms", None)),
    }


def estimate_object_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate owned Python/NumPy bytes for diagnostics at scenario boundaries."""

    if seen is None:
        seen = set()

    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    if isinstance(value, np.ndarray):
        return int(sys.getsizeof(value))

    size = int(sys.getsizeof(value))
    if isinstance(value, dict):
        for key, item in value.items():
            size += estimate_object_bytes(key, seen)
            size += estimate_object_bytes(item, seen)
        return size
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            size += estimate_object_bytes(item, seen)
        return size
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            size += estimate_object_bytes(getattr(value, field.name), seen)
    return size


def _json_safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, bool, int, float)) or item is None:
            result[str(key)] = item
    return result


def _cache_containers(component: Any) -> dict[str, Any]:
    if component is None:
        return {}

    containers: dict[str, Any] = {}
    for name, value in vars(component).items():
        if "cache" not in name.lower():
            continue
        if isinstance(value, (dict, list, tuple, set, frozenset)):
            containers[name] = value
    return containers


def cache_snapshot(backend: Any, action_space: Any) -> dict[str, Any]:
    """Capture cache counters and owned-byte diagnostics."""

    backend_info: dict[str, Any] = {}
    if backend is not None:
        info_fn = getattr(backend, "performance_info", None)
        if not callable(info_fn):
            info_fn = getattr(backend, "cache_info", None)
        if callable(info_fn):
            try:
                backend_info = _json_safe_mapping(info_fn())
            except Exception:
                backend_info = {}

    action_info: dict[str, Any] = {}
    info_fn = getattr(action_space, "cache_info", None)
    if callable(info_fn):
        try:
            action_info = _json_safe_mapping(info_fn())
        except Exception:
            action_info = {}

    backend_caches = _cache_containers(backend)
    action_caches = _cache_containers(action_space)

    backend_owned = backend_info.get("bytes")
    if not isinstance(backend_owned, (int, float)):
        backend_owned = estimate_object_bytes(tuple(backend_caches.values()))

    action_owned = action_info.get("bytes")
    if not isinstance(action_owned, (int, float)):
        action_owned = estimate_object_bytes(tuple(action_caches.values()))

    return {
        "backend": backend_info,
        "action_space": action_info,
        "estimated_bytes": {
            "backend": int(backend_owned),
            "action_space": int(action_owned),
        },
        "containers": {
            "backend": {
                name: len(value)
                for name, value in backend_caches.items()
            },
            "action_space": {
                name: len(value)
                for name, value in action_caches.items()
            },
        },
    }


def cache_counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Return per-scenario counters for the active cache implementations."""

    result: dict[str, Any] = {}
    for component in ("backend", "action_space"):
        start = before.get(component, {})
        end = after.get(component, {})
        delta: dict[str, int | float] = {}
        for name in ("hits", "misses", "negative_hits", "evictions"):
            old_value = start.get(name)
            new_value = end.get(name)
            if isinstance(old_value, (int, float)) and isinstance(
                new_value,
                (int, float),
            ):
                delta[name] = new_value - old_value
        result[component] = delta
    return result


class TeacherRuntimeProfiler:
    """Process-local timing probes for teacher performance diagnostics."""

    def __init__(self) -> None:
        self._elapsed: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._patches: list[tuple[Any, str, Any]] = []

    def reset(self) -> None:
        self._elapsed.clear()
        self._calls.clear()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._elapsed[name] += time.perf_counter() - started
            self._calls[name] += 1

    def patch(self, owner: Any, attribute: str, name: str) -> None:
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any):
            with self.measure(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, wrapped)
        self._patches.append((owner, attribute, original))

    def install_default_probes(self) -> None:
        """Instrument hot paths used by the current impact teacher."""

        if self._patches:
            raise RuntimeError("Teacher runtime probes are already installed.")

        from grid_topology_ai.action_space import GridFMActionSpace
        from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
        from grid_topology_ai.search.impact_beam_search import ImpactBeamSearchPlanner
        from grid_topology_ai.state_store import GridFMStateStore
        from scripts.self_play import generate_impact_teacher_parallel_fast as teacher

        self.patch(
            GridFMPowerFlowBackend,
            "run_power_flow_from_state",
            "pf.transition_total",
        )
        self.patch(
            GridFMPowerFlowBackend,
            "_build_ppc_from_state",
            "pf.problem_build",
        )
        self.patch(
            GridFMPowerFlowBackend,
            "_solve_ppc",
            "pf.solve",
        )
        self.patch(
            GridFMActionSpace,
            "structural_action_mask",
            "action.structural_mask",
        )
        self.patch(
            GridFMActionSpace,
            "operational_action_mask",
            "action.operational_mask",
        )
        self.patch(
            ImpactBeamSearchPlanner,
            "search",
            "teacher.beam_search",
        )
        self.patch(
            teacher,
            "rank_actions_by_lodf_screening",
            "screening.lodf",
        )
        self.patch(
            GridFMStateStore,
            "save_state",
            "io.state_write",
        )

    def restore(self) -> None:
        while self._patches:
            owner, attribute, original = self._patches.pop()
            setattr(owner, attribute, original)

    @contextmanager
    def installed(self) -> Iterator["TeacherRuntimeProfiler"]:
        self.install_default_probes()
        try:
            yield self
        finally:
            self.restore()

    def snapshot(self) -> dict[str, Any]:
        return {
            "timers_sec": {
                name: float(self._elapsed[name])
                for name in sorted(self._elapsed)
            },
            "calls": {
                name: int(self._calls[name])
                for name in sorted(self._calls)
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.snapshot(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
