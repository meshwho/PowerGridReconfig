from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.gridfm_adapter import GridFMAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GridFM adapter and build GNN-ready states."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to gridfm raw output directory.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking GridFM adapter")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)

    scenario_ids = adapter.scenario_ids()
    useful_ids = adapter.useful_scenario_ids()

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Total scenarios: {len(scenario_ids)}")
    print(f"Useful emergency scenarios: {len(useful_ids)}")
    print(f"Useful scenario IDs: {useful_ids}")

    if not useful_ids:
        print("No useful scenarios found.")
        return

    scenario_id = useful_ids[0]
    state = adapter.build_state(scenario_id)

    print("\nExample state:")
    print(f"Scenario ID:          {state.scenario_id}")
    print(f"Load scenario idx:    {state.load_scenario_idx}")
    print(f"Bus features shape:   {state.bus_features.shape}")
    print(f"Branch features shape:{state.branch_features.shape}")
    print(f"Edge index shape:     {state.edge_index.shape}")
    print(f"Branch ids shape:     {state.branch_ids.shape}")
    print(f"Branch status shape:  {state.branch_status.shape}")
    print(f"Outaged branches:     {state.outaged_branch_ids}")

    print("\nMetrics:")
    for key, value in state.metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()