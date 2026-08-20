from __future__ import annotations

import importlib
import importlib.util

from grid_topology_ai.data_adapter import GridFMState


def test_canonical_state_modules_are_available() -> None:
    module_names = (
        "grid_topology_ai.state.builder",
        "grid_topology_ai.state.schema",
        "grid_topology_ai.state.topology",
        "grid_topology_ai.state.artifacts",
        "grid_topology_ai.state.fingerprint",
        "grid_topology_ai.state.store",
    )

    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name


def test_legacy_state_modules_are_absent() -> None:
    module_names = (
        "grid_topology_ai.state_builder",
        "grid_topology_ai.state_schema",
        "grid_topology_ai.state_topology",
        "grid_topology_ai.state_artifact_schema",
        "grid_topology_ai.state_fingerprint",
        "grid_topology_ai.state_store",
    )

    for module_name in module_names:
        assert importlib.util.find_spec(module_name) is None


def test_grid_fm_state_preserves_public_module_identity() -> None:
    assert GridFMState.__module__ == "grid_topology_ai.data_adapter"
