from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


class GridFMStateStore:
    """
    Save GridFMState objects as compressed NPZ files.

    One NPZ file = one graph state.

    This file will later be used by:
    - supervised policy pretraining;
    - value function training;
    - GNN / GAT model;
    - MCTS / AlphaZero-style pipeline.
    """

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
        """
        Save one state as .npz.

        Parameters
        ----------
        state:
            GridFMState object.

        state_id:
            Name of the saved state file, for example:
                scenario_000007

        action_mask:
            Boolean array of valid actions.
            Shape:
                [1 + num_branches]

        extra_metadata:
            Optional additional metadata.
        """

        output_path = self.output_dir / f"{state_id}.npz"

        if action_mask is None:
            action_mask_array = np.array([], dtype=np.int8)
        else:
            action_mask_array = action_mask.astype(np.int8)

        metadata = {
            "scenario_id": int(state.scenario_id),
            "load_scenario_idx": float(state.load_scenario_idx),
            "outaged_branch_ids": [int(x) for x in state.outaged_branch_ids],
            "physical_objective_schema_version": (
                PHYSICAL_OBJECTIVE_SCHEMA_VERSION
            ),
        }

        if extra_metadata is not None:
            metadata.update(extra_metadata)

        np.savez_compressed(
            output_path,
            bus_features=state.bus_features.astype(np.float32),
            branch_features=state.branch_features.astype(np.float32),
            edge_index=state.edge_index.astype(np.int64),
            branch_ids=state.branch_ids.astype(np.int64),
            branch_status=state.branch_status.astype(np.float32),
            action_mask=action_mask_array,
            metrics_json=np.array(json.dumps(state.metrics)),
            metadata_json=np.array(json.dumps(metadata)),
        )

        return output_path
