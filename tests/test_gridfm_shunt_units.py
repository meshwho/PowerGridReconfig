from __future__ import annotations

import numpy as np
import pandas as pd
from pypower.idx_bus import BS, GS

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def test_gridfm_per_unit_shunts_are_restored_for_pypower() -> None:
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(base_mva=100.0),
        enable_cache=False,
    )
    bus_df = pd.DataFrame(
        {
            "bus": [1, 2],
            "PQ": [1.0, 1.0],
            "PV": [0.0, 0.0],
            "REF": [0.0, 0.0],
            "Pd": [0.0, 0.0],
            "Qd": [0.0, 0.0],
            "GS": [0.25, -0.10],
            "BS": [-0.40, 0.05],
            "Vm": [1.0, 1.0],
            "Va": [0.0, 0.0],
            "vn_kv": [230.0, 230.0],
            "max_vm_pu": [1.1, 1.1],
            "min_vm_pu": [0.9, 0.9],
        }
    )

    bus = backend._build_bus_matrix(bus_df)

    np.testing.assert_allclose(bus[:, GS], [25.0, -10.0])
    np.testing.assert_allclose(bus[:, BS], [-40.0, 5.0])
