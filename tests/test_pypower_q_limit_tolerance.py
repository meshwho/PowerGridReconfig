from __future__ import annotations

from grid_topology_ai.config import PhysicsConfig
from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


def test_q_limit_tolerance_comes_from_physics_config() -> None:
    config = PhysicsConfig(generator_q_tolerance_mvar=2.5e-7)
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=config,
    )

    options = backend._build_pp_options()

    assert options["OPF_VIOLATION"] == config.generator_q_tolerance_mvar
