from __future__ import annotations

import importlib
import importlib.util

from grid_topology_ai.state import GridFMState


def test_canonical_state_package_layout():
    canonical_modules = (
        "grid_topology_ai.state",
        "grid_topology_ai.state.io",
    )
    legacy_modules = (
        "grid_topology_ai.state_builder",
        "grid_topology_ai.state_schema",
        "grid_topology_ai.state_topology",
        "grid_topology_ai.state_artifact_schema",
        "grid_topology_ai.state_fingerprint",
        "grid_topology_ai.state_store",
    )

    for module_name in canonical_modules:
        importlib.import_module(module_name)

    for module_name in legacy_modules:
        assert importlib.util.find_spec(module_name) is None


def test_grid_fm_state_has_canonical_state_identity():
    assert GridFMState.__module__ == "grid_topology_ai.state"
