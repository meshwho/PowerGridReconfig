from __future__ import annotations

import importlib

from grid_topology_ai.state import GridFMState


def test_canonical_state_package_layout():
    assert importlib.import_module("grid_topology_ai.state") is not None


def test_grid_fm_state_has_canonical_state_identity():
    assert GridFMState.__module__ == "grid_topology_ai.state"
