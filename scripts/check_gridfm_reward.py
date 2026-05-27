from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.gridfm_action_space import GridFMActionSpace
from grid_topology_ai.gridfm_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from grid_topology_ai.gridfm_pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.gridfm_reward import GridFMReward


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check reward for one GridFM topology switching transition."
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
        help="Scenario ID.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking GridFM reward")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    reward_fn = GridFMReward()

    state = adapter.build_state(args.scenario)

    action_space = GridFMActionSpace(require_connected_after_switch=True)
    valid_actions = action_space.valid_actions(state)

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    switch_actions = [
        action for action in valid_actions if action.action_type == "switch_off_branch"
    ]

    if not switch_actions:
        print("No valid switch-off actions found.")
        return

    # Test the same style of action as before:
    # valid switch-off action with maximum current loading.
    action = max(
        switch_actions,
        key=lambda a: float(state.branch_features[a.branch_pos, loading_idx]),
    )

    print(f"Scenario:      {args.scenario}")
    print(f"Action ID:     {action.action_id}")
    print(f"Action type:   {action.action_type}")
    print(f"Branch ID:     {action.branch_id}")
    print(f"Branch pos:    {action.branch_pos}")
    print(
        f"Branch loading:{float(state.branch_features[action.branch_pos, loading_idx]):.2f}%"
    )

    result = backend.run_power_flow(
        scenario_id=args.scenario,
        switched_off_branch_id=action.branch_id,
    )

    reward = reward_fn.compute(
        before_state=state,
        after_state=result.next_state,
        action_is_switching=True,
        power_flow_success=result.success,
    )

    print("\nPower flow:")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")

    print("\nReward breakdown:")
    print(f"  reward:                    {reward.reward:.4f}")
    print(f"  before_penalty:            {reward.before_penalty:.4f}")
    print(f"  after_penalty:             {reward.after_penalty:.4f}")
    print(f"  improvement:               {reward.improvement:.4f}")
    print(f"  switching_penalty:         {reward.switching_penalty:.4f}")
    print(f"  done:                      {reward.done}")
    print(f"  success:                   {reward.success}")

    print("\nLoading comparison:")
    print(f"  before max loading:        {reward.before_max_loading:.2f}%")
    print(f"  after max loading:         {reward.after_max_loading:.2f}%")
    print(f"  before total overload:     {reward.before_total_overload:.2f}")
    print(f"  after total overload:      {reward.after_total_overload:.2f}")
    print(f"  before overloaded count:   {reward.before_num_overloaded}")
    print(f"  after overloaded count:    {reward.after_num_overloaded}")
    print(f"  before hard overloaded:    {reward.before_num_hard_overloaded}")
    print(f"  after hard overloaded:     {reward.after_num_hard_overloaded}")

    print("\nVoltage comparison:")
    print(f"  before voltage penalty:    {reward.before_voltage_penalty:.2f}")
    print(f"  after voltage penalty:     {reward.after_voltage_penalty:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()