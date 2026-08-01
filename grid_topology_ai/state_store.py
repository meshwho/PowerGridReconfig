from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.state_schema import state_feature_schema_provenance


class GridFMStateStore:
    """Save one validated graph state per compressed NPZ file."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_state(
        self,
        state: GridFMState,
        state_id: str,
        action_mask: np.ndarray | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        output_path = self.output_dir / f"{state_id}.npz"

        if action_mask is None:
            action_mask_array = np.array([], dtype=np.int8)
        else:
            action_mask_array = np.asarray(action_mask, dtype=np.int8)

        bus_ids = self._validated_bus_ids(state)
        metadata = dict(extra_metadata or {})
        metadata.update(
            {
                "scenario_id": int(state.scenario_id),
                "load_scenario_idx": float(state.load_scenario_idx),
                "outaged_branch_ids": [
                    int(value) for value in state.outaged_branch_ids
                ],
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
                **state_feature_schema_provenance(),
            }
        )

        np.savez_compressed(
            output_path,
            bus_features=state.bus_features.astype(np.float32),
            branch_features=state.branch_features.astype(np.float32),
            edge_index=state.edge_index.astype(np.int64),
            bus_ids=bus_ids,
            branch_ids=state.branch_ids.astype(np.int64),
            branch_status=state.branch_status.astype(np.float32),
            action_mask=action_mask_array,
            metrics_json=np.array(json.dumps(state.metrics)),
            metadata_json=np.array(json.dumps(metadata)),
        )

        return output_path

    @staticmethod
    def _validated_bus_ids(state: GridFMState) -> np.ndarray:
        if state.bus_ids is None:
            raise ValueError(
                "GridFMState.bus_ids is required when saving schema-v2 states."
            )

        try:
            values = np.asarray(state.bus_ids, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("GridFMState.bus_ids must be numeric.") from exc

        expected_shape = (int(state.bus_features.shape[0]),)
        if values.shape != expected_shape:
            raise ValueError(
                "GridFMState.bus_ids shape mismatch: expected "
                f"{expected_shape}, observed {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("GridFMState.bus_ids must contain finite values.")
        if not np.equal(values, np.rint(values)).all():
            raise ValueError(
                "GridFMState.bus_ids must contain integer-valued IDs."
            )

        bus_ids = values.astype(np.int64)
        if np.unique(bus_ids).size != bus_ids.size:
            raise ValueError("GridFMState.bus_ids must be unique.")
        return bus_ids
