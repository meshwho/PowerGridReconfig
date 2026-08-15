from __future__ import annotations

from pathlib import Path

from grid_topology_ai._pypower_backend_core import (
    GridFMPowerFlowBackend as PhysicalCoreBackend,
)
from grid_topology_ai.cache import ExactPowerFlowCache
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


_OLD_CACHE_SYMBOLS = (
    "_TopologyCacheEntry",
    "_topology_cache",
    "_pending_warm_start_state",
    "_pending_warm_start_applied",
    "_topology_bucket_key",
    "_select_topology_entry",
    "_remember_topology_result",
    "_apply_pending_warm_start",
    "_make_cache_key_from_state",
    "_make_topology_cache_key_from_state",
    "_power_flow_input_fingerprint",
    "_scenario_static_input_fingerprint",
    "_scenario_static_input_fingerprints",
)


def test_physical_core_contains_no_power_flow_cache_implementation() -> None:
    backend = PhysicalCoreBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(),
        enable_cache=True,
    )

    assert "_cache" not in vars(backend)
    assert not hasattr(PhysicalCoreBackend, "cache_info")
    assert not hasattr(PhysicalCoreBackend, "clear_cache")
    for name in _OLD_CACHE_SYMBOLS:
        assert not hasattr(PhysicalCoreBackend, name), name


def test_public_backend_has_only_exact_cache_component() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(),
        enable_cache=True,
        exact_cache_max_bytes=4096,
    )

    assert isinstance(backend._exact_power_flow_cache, ExactPowerFlowCache)
    assert backend._exact_power_flow_cache.__class__.__module__.startswith(
        "grid_topology_ai.cache."
    )
    for name in _OLD_CACHE_SYMBOLS:
        assert not hasattr(backend, name), name


def test_power_flow_modules_do_not_reintroduce_approximate_cache_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        root / "grid_topology_ai" / "_pypower_backend_core.py",
        root / "grid_topology_ai" / "pypower_backend.py",
    )
    forbidden_fragments = (
        "tolerant_cache",
        "warm_start_hits",
        "cold_start_misses",
        "nearest_operating",
        "_TopologyCacheEntry",
        "_pending_warm_start",
        "_make_topology_cache_key_from_state",
    )

    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path.name}"
