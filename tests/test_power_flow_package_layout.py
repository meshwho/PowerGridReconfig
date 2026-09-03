"""Contracts for the canonical power-flow package layout."""

import importlib

from grid_topology_ai.power_flow.backend import GridFMPowerFlowResult


CANONICAL_MODULES = (
    "grid_topology_ai.power_flow.backend",
    "grid_topology_ai.power_flow.problem",
    "grid_topology_ai.power_flow.solver",
    "grid_topology_ai.power_flow.workspace",
)


def test_canonical_power_flow_modules_are_importable() -> None:
    for module_name in CANONICAL_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_power_flow_result_has_canonical_identity() -> None:
    assert GridFMPowerFlowResult.__module__ == "grid_topology_ai.power_flow.backend"
