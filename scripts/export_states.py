from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.state_store import GridFMStateStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GridFM states as NPZ tensors for GNN/RL training."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--transitions",
        type=str,
        default=None,
        help=(
            "Optional transitions CSV. If provided, only scenarios present "
            "in this file will be exported."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/states",
        help="Directory where NPZ state files will be saved.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional CSV summary output path.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print("=" * 100)
    print("Exporting GridFM states")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Output dir:    {output_dir.resolve()}")

    adapter = GridFMAdapter(raw_dir)
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    state_store = GridFMStateStore(output_dir)

    if args.transitions is not None:
        transitions = pd.read_csv(args.transitions)
        scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())
        print(f"Using scenario IDs from transitions file: {args.transitions}")
    else:
        scenario_ids = adapter.useful_scenario_ids()
        print("Using all useful scenario IDs from GridFMAdapter.")

    print(f"Scenario IDs: {scenario_ids}")

    rows = []

    for scenario_id in scenario_ids:
        state = adapter.build_state(scenario_id)
        action_mask = action_space.valid_action_mask(state)

        state_id = f"scenario_{scenario_id:06d}"

        state_path = state_store.save_state(
            state=state,
            state_id=state_id,
            action_mask=action_mask,
            extra_metadata={
                "source": "gridfm-datakit",
                "state_type": "initial_emergency_state",
            },
        )

        rows.append(
            {
                "scenario_id": int(scenario_id),
                "state_id": state_id,
                "state_path": str(state_path),
                "num_buses": int(state.bus_features.shape[0]),
                "num_branches": int(state.branch_features.shape[0]),
                "num_bus_features": int(state.bus_features.shape[1]),
                "num_branch_features": int(state.branch_features.shape[1]),
                "num_actions": int(len(action_mask)),
                "num_valid_actions": int(action_mask.sum()),
                "max_loading_percent": float(state.metrics["max_loading_percent"]),
                "num_overloaded_branches": int(
                    state.metrics["num_overloaded_branches"]
                ),
                "num_hard_overloaded_branches": int(
                    state.metrics["num_hard_overloaded_branches"]
                ),
                "num_outaged_branches": int(state.metrics["num_outaged_branches"]),
                "outaged_branch_ids": state.outaged_branch_ids,
            }
        )

        print(
            f"Saved {state_id}: "
            f"buses={state.bus_features.shape[0]}, "
            f"branches={state.branch_features.shape[0]}, "
            f"valid_actions={int(action_mask.sum())}, "
            f"max_loading={state.metrics['max_loading_percent']:.2f}%"
        )

    summary = pd.DataFrame(rows)

    if args.summary is None:
        summary_path = output_dir / "states_summary.csv"
    else:
        summary_path = Path(args.summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print("\nSaved summary:")
    print(summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()