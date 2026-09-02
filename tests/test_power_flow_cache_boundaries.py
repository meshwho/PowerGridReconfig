from __future__ import annotations

from grid_topology_ai.cache import ExactPowerFlowCache
from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


def test_public_backend_uses_exact_cache_component() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(),
        enable_cache=True,
        exact_cache_max_bytes=4096,
    )

    assert isinstance(backend._exact_power_flow_cache, ExactPowerFlowCache)
    assert backend._exact_power_flow_cache.__class__.__module__ == "grid_topology_ai.cache"
