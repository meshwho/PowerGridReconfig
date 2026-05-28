from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def print_metrics(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PYPOWER backend for GridFM scenarios."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Scenario ID. If omitted, the first useful scenario is used.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking GridFM PYPOWER backend")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)

    useful_ids = adapter.useful_scenario_ids()

    if not useful_ids:
        print("No useful emergency scenarios found.")
        return

    scenario_id = args.scenario if args.scenario is not None else useful_ids[0]

    state = adapter.build_state(scenario_id)

    print(f"Raw directory:       {raw_dir.resolve()}")
    print(f"Useful scenario IDs: {useful_ids}")
    print(f"Scenario ID:         {scenario_id}")
    print(f"Outaged branches:    {state.outaged_branch_ids}")

    print_metrics("Original GridFM state metrics", state.metrics)

    # First, check that PYPOWER can reproduce the same scenario without
    # any additional switching action.
    no_action_result = backend.run_power_flow(
        scenario_id=scenario_id,
        switched_off_branch_id=None,
    )

    print("\nNo-action PYPOWER result:")
    print(f"  success: {no_action_result.success}")
    print(f"  message: {no_action_result.message}")

    if no_action_result.next_state is not None:
        print_metrics(
            "PYPOWER no-action next_state metrics",
            no_action_result.next_state.metrics,
        )

    # Now select one valid branch switch-off action.
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    valid_actions = action_space.valid_actions(state)

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    switch_actions = [
        action
        for action in valid_actions
        if action.action_type == "switch_off_branch"
    ]

    if not switch_actions:
        print("\nNo valid switch-off actions found.")
        return

    # Choose the valid action whose branch currently has the largest loading.
    best_test_action = max(
        switch_actions,
        key=lambda action: float(
            state.branch_features[action.branch_pos, loading_idx]
        ),
    )

    branch_loading = float(
        state.branch_features[best_test_action.branch_pos, loading_idx]
    )

    print("\nTest topology action:")
    print(f"  action_id:   {best_test_action.action_id}")
    print(f"  branch_id:   {best_test_action.branch_id}")
    print(f"  branch_pos:  {best_test_action.branch_pos}")
    print(f"  loading:     {branch_loading:.2f} %")

    action_result = backend.run_power_flow(
        scenario_id=scenario_id,
        switched_off_branch_id=best_test_action.branch_id,
    )

    print("\nAction PYPOWER result:")
    print(f"  success: {action_result.success}")
    print(f"  message: {action_result.message}")

    if action_result.next_state is not None:
        print_metrics(
            "PYPOWER after-action next_state metrics",
            action_result.next_state.metrics,
        )

        original_max = state.metrics["max_loading_percent"]
        new_max = action_result.next_state.metrics["max_loading_percent"]

        print("\nComparison:")
        print(f"  original max loading: {original_max:.2f} %")
        print(f"  new max loading:      {new_max:.2f} %")
        print(f"  delta:                {new_max - original_max:+.2f} %")

    print("\nDone.")


if __name__ == "__main__":
    main()