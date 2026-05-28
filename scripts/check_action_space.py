from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GridFM topology switching action space."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to gridfm raw output directory.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Optional scenario ID. If omitted, the first useful scenario is used.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking GridFM action space")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)
    useful_ids = adapter.useful_scenario_ids()

    if not useful_ids:
        print("No useful emergency scenarios found.")
        return

    scenario_id = args.scenario if args.scenario is not None else useful_ids[0]

    state = adapter.build_state(scenario_id)

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        min_loading_for_switch_percent=0.0,
    )

    all_actions = action_space.build_all_actions(state)
    valid_actions = action_space.valid_actions(state)
    invalid_actions = action_space.invalid_actions(state)
    mask = action_space.valid_action_mask(state)

    print(f"Raw directory:            {raw_dir.resolve()}")
    print(f"Scenario ID:              {scenario_id}")
    print(f"Useful scenario IDs:      {useful_ids}")
    print(f"Total actions:            {len(all_actions)}")
    print(f"Valid actions:            {len(valid_actions)}")
    print(f"Invalid actions:          {len(invalid_actions)}")
    print(f"Mask shape:               {mask.shape}")
    print(f"Outaged branches:         {state.outaged_branch_ids}")

    print("\nState metrics:")
    for key, value in state.metrics.items():
        print(f"  {key}: {value}")

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    print("\nFirst 20 valid actions:")
    for action in valid_actions[:20]:
        if action.action_type == "do_nothing":
            print(
                f"  action_id={action.action_id:3d} | "
                f"type={action.action_type}"
            )
        else:
            loading = float(state.branch_features[action.branch_pos, loading_idx])
            print(
                f"  action_id={action.action_id:3d} | "
                f"type={action.action_type:17s} | "
                f"branch_id={action.branch_id:4d} | "
                f"branch_pos={action.branch_pos:4d} | "
                f"loading={loading:8.2f}%"
            )

    print("\nTop 15 loaded active branches and whether switch-off is valid:")

    branch_loadings = state.branch_features[:, loading_idx]
    active_mask = state.branch_status > 0

    active_positions = np.where(active_mask)[0]
    sorted_positions = active_positions[
        np.argsort(branch_loadings[active_positions])[::-1]
    ]

    for branch_pos in sorted_positions[:15]:
        action_id = 1 + int(branch_pos)
        branch_id = int(state.branch_ids[branch_pos])
        loading = float(branch_loadings[branch_pos])
        is_valid = bool(mask[action_id])

        from_bus = int(state.edge_index[0, branch_pos])
        to_bus = int(state.edge_index[1, branch_pos])

        print(
            f"  branch_id={branch_id:4d} | "
            f"pos={branch_pos:4d} | "
            f"{from_bus:3d}->{to_bus:3d} | "
            f"loading={loading:8.2f}% | "
            f"switch_valid={is_valid}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()