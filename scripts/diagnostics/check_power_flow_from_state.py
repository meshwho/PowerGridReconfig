from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward


def print_metrics(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check multi-step power flow from current GridFMState."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=7,
        help="Scenario ID to test.",
    )

    parser.add_argument(
        "--first-branch",
        type=int,
        default=122,
        help="First branch ID to switch off.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking run_power_flow_from_state")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    reward_fn = GridFMReward()
    action_space = GridFMActionSpace(require_connected_after_switch=True)

    state_0 = adapter.build_state(args.scenario)

    print_metrics("Initial state", state_0.metrics)
    print(f"Initial outaged branches: {state_0.outaged_branch_ids}")

    # Step 1: use old method from original scenario.
    result_1 = backend.run_power_flow(
        scenario_id=args.scenario,
        switched_off_branch_id=args.first_branch,
    )

    print("\nStep 1 result:")
    print(f"  success: {result_1.success}")
    print(f"  message: {result_1.message}")

    if not result_1.success or result_1.next_state is None:
        print("Step 1 failed.")
        return

    state_1 = result_1.next_state

    reward_1 = reward_fn.compute(
        before_state=state_0,
        after_state=state_1,
        action_is_switching=True,
        power_flow_success=result_1.success,
    )

    print_metrics("State after step 1", state_1.metrics)
    print(f"Outaged branches after step 1: {state_1.outaged_branch_ids}")
    print(f"Step 1 reward: {reward_1.reward:.4f}")
    print(f"Step 1 done:   {reward_1.done}")

    # Step 2: now use the NEW method from already modified state.
    valid_actions = action_space.valid_actions(state_1)

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    switch_actions = [
        action
        for action in valid_actions
        if action.action_type == "switch_off_branch"
        and action.branch_id != args.first_branch
    ]

    if not switch_actions:
        print("\nNo valid second switch actions.")
        return

    # For this check, choose the currently most loaded valid branch.
    second_action = max(
        switch_actions,
        key=lambda action: float(
            state_1.branch_features[action.branch_pos, loading_idx]
        ),
    )

    print("\nSelected second action:")
    print(f"  action_id:  {second_action.action_id}")
    print(f"  branch_id:  {second_action.branch_id}")
    print(f"  branch_pos: {second_action.branch_pos}")
    print(
        "  loading:    "
        f"{float(state_1.branch_features[second_action.branch_pos, loading_idx]):.2f}%"
    )

    result_2 = backend.run_power_flow_from_state(
        state=state_1,
        switched_off_branch_id=second_action.branch_id,
    )

    print("\nStep 2 result:")
    print(f"  success: {result_2.success}")
    print(f"  message: {result_2.message}")

    if not result_2.success or result_2.next_state is None:
        print("Step 2 failed.")
        return

    state_2 = result_2.next_state

    reward_2 = reward_fn.compute(
        before_state=state_1,
        after_state=state_2,
        action_is_switching=True,
        power_flow_success=result_2.success,
    )

    print_metrics("State after step 2", state_2.metrics)
    print(f"Outaged branches after step 2: {state_2.outaged_branch_ids}")
    print(f"Step 2 reward: {reward_2.reward:.4f}")
    print(f"Step 2 done:   {reward_2.done}")

    print("\nComparison:")
    print(
        f"  max loading: "
        f"{state_0.metrics['max_loading_percent']:.2f}%"
        f" -> {state_1.metrics['max_loading_percent']:.2f}%"
        f" -> {state_2.metrics['max_loading_percent']:.2f}%"
    )
    print(
        f"  overloaded branches: "
        f"{state_0.metrics['num_overloaded_branches']}"
        f" -> {state_1.metrics['num_overloaded_branches']}"
        f" -> {state_2.metrics['num_overloaded_branches']}"
    )
    print(
        f"  hard overloaded branches: "
        f"{state_0.metrics['num_hard_overloaded_branches']}"
        f" -> {state_1.metrics['num_hard_overloaded_branches']}"
        f" -> {state_2.metrics['num_hard_overloaded_branches']}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()