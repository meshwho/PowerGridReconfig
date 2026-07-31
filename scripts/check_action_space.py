from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    print_topology_action_config,
    topology_action_config_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check GridFM topology switching action space."
    )
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--scenario", type=int, default=None)
    add_topology_action_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    topology_config = topology_action_config_from_args(args)
    adapter = GridFMAdapter(raw_dir)
    useful_ids = adapter.useful_scenario_ids()
    if not useful_ids:
        print("No useful emergency scenarios found.")
        return

    scenario_id = args.scenario if args.scenario is not None else useful_ids[0]
    state = adapter.build_state(scenario_id)
    action_space = GridFMActionSpace(
        **action_space_kwargs(topology_config, enable_cache=False)
    )
    all_actions = action_space.build_all_actions(state)
    valid_actions = action_space.valid_actions(state)
    invalid_actions = action_space.invalid_actions(state)
    mask = action_space.valid_action_mask(state)
    loading_index = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    print("=" * 100)
    print("Checking GridFM action space")
    print("=" * 100)
    print(f"Raw directory:       {raw_dir.resolve()}")
    print(f"Scenario ID:         {scenario_id}")
    print(f"Total actions:       {len(all_actions)}")
    print(f"Valid actions:       {len(valid_actions)}")
    print(f"Invalid actions:     {len(invalid_actions)}")
    print(f"Outaged branches:    {state.outaged_branch_ids}")
    print_topology_action_config(topology_config)

    print("\nFirst 20 valid actions:")
    for action in valid_actions[:20]:
        if action.kind == "stop":
            print(f"  action_id={action.action_id:3d} | type={action.action_type}")
            continue
        loading = float(state.branch_features[action.branch_pos, loading_index])
        print(
            f"  action_id={action.action_id:3d} | type={action.action_type:17s} | "
            f"target_status={action.target_status} | branch_id={action.branch_id:4d} | "
            f"branch_pos={action.branch_pos:4d} | loading={loading:8.2f}%"
        )

    branch_loadings = state.branch_features[:, loading_index]
    active_positions = np.flatnonzero(state.branch_status > 0)
    active_positions = active_positions[np.argsort(branch_loadings[active_positions])[::-1]]
    print("\nTop 15 loaded active branches and whether opening is valid:")
    for branch_pos in active_positions[:15]:
        action_id = 1 + int(branch_pos)
        print(
            f"  branch_id={int(state.branch_ids[branch_pos]):4d} | "
            f"pos={int(branch_pos):4d} | loading={float(branch_loadings[branch_pos]):8.2f}% | "
            f"open_valid={bool(mask[action_id])}"
        )

    print("\nInactive branches and whether closing is valid:")
    for branch_pos in np.flatnonzero(state.branch_status <= 0):
        action_id = 1 + int(branch_pos)
        branch_id = int(state.branch_ids[branch_pos])
        print(
            f"  branch_id={branch_id:4d} | pos={int(branch_pos):4d} | "
            f"configured_closeable={branch_id in topology_config.closeable_branch_ids} | "
            f"close_valid={bool(mask[action_id])}"
        )


if __name__ == "__main__":
    main()
