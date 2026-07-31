from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.physical_constraints import (
    calculate_physical_metrics_from_result,
)
from grid_topology_ai.state_builder import GridFMState, GridFMStateBuilder


@dataclass(frozen=True, slots=True)
class PowerFlowStateBuilder:
    """Compatibility wrapper for solved-state construction."""

    physics_config: PhysicsConfig

    def __post_init__(self) -> None:
        if not isinstance(self.physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig.")

    def build(
        self,
        *,
        scenario_id: int,
        result_ppc: Mapping[str, Any],
        original_frames: Mapping[str, pd.DataFrame],
        physical_metrics: Mapping[str, object] | None = None,
    ) -> GridFMState:
        return GridFMStateBuilder(
            physics_config=self.physics_config,
            result_metrics_calculator=(
                calculate_physical_metrics_from_result
            ),
        ).build_from_pypower_result(
            scenario_id=scenario_id,
            result_ppc=result_ppc,
            original_frames=original_frames,
            physical_metrics=physical_metrics,
        )
