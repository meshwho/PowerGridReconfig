"""Contracts for the canonical power-flow package layout."""

import importlib
import importlib.util

from grid_topology_ai.power_flow.backend import GridFMPowerFlowResult


CANONICAL_MODULES = (
    "grid_topology_ai.power_flow.backend",
    "grid_topology_ai.power_flow.problem",
    "grid_topology_ai.power_flow.solver",
    "grid_topology_ai.power_flow.network_workspace",
    "grid_topology_ai.power_flow.newton_workspace",
    "grid_topology_ai.power_flow.errors",
)

LEGACY_MODULES = (
    "grid_topology_ai.pypower_backend",
    "grid_topology_ai.power_flow_problem",
    "grid_topology_ai.pypower_compat",
    "grid_topology_ai.pypower_network_workspace",
    "grid_topology_ai.pypower_newton_workspace",
    "grid_topology_ai.power_flow_errors",
)


def test_canonical_power_flow_modules_are_importable() -> None:
    for module_name in CANONICAL_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_legacy_power_flow_modules_are_absent() -> None:
    for module_name in LEGACY_MODULES:
        assert importlib.util.find_spec(module_name) is None


def test_power_flow_result_has_canonical_identity() -> None:
    assert GridFMPowerFlowResult.__module__ == "grid_topology_ai.power_flow.backend"
