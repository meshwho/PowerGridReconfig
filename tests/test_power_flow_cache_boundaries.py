from __future__ import annotations

from pathlib import Path

from grid_topology_ai.cache import ExactPowerFlowCache
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


_OLD_CACHE_SYMBOLS = (
    "_TopologyCacheEntry",
    "_topology_cache",
    "_topology_bucket_key",
    "_select_topology_entry",
    "_remember_topology_result",
    "_make_cache_key_from_state",
    "_make_topology_cache_key_from_state",
    "_power_flow_input_fingerprint",
    "_scenario_static_input_fingerprint",
    "_scenario_static_input_fingerprints",
)


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


def test_power_flow_module_does_not_reintroduce_approximate_cache_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "grid_topology_ai" / "power_flow" / "backend.py"
    forbidden_fragments = (
        "tolerant_cache",
        "cold_start_misses",
        "nearest_operating",
        "_TopologyCacheEntry",
        "_make_topology_cache_key_from_state",
    )

    source = source_path.read_text(encoding="utf-8")
    for fragment in forbidden_fragments:
        assert fragment not in source, fragment
