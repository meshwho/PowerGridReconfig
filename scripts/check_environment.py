from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward


def print_state(title: str, env: TopologySwitchingEnv) -> None:
    state = env.current_state

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if state is None:
        print("No current state.")
        return

    print(f"Scenario ID:       {state.scenario_id}")
    print(f"Step count:        {env.step_count}")
    print(f"Done:              {env.done}")
    print(f"Solved:            {env.solved}")
    print(f"Termination:       {env.termination_reason}")
    print(f"Switched branches: {env.switched_branch_ids}")
    print(f"Outaged branches:  {state.outaged_branch_ids}")

    print("\nMetrics:")
    for key, value in state.metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check multi-step TopologySwitchingEnv."
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

    parser.add_argument(
        "--first-branch",
        type=int,
        default=122,
        help="First branch to switch off.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum episode steps.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Checking TopologySwitchingEnv")
    print("=" * 100)

    adapter = GridFMAdapter(raw_dir)
    backend = GridFMPowerFlowBackend(adapter)
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    reward_fn = GridFMReward()

    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.max_steps,
    )

    env.reset(args.scenario)
    print_state("Initial environment state", env)

    first_action = env.action_by_branch_id(args.first_branch)

    print("\nFirst action:")
    print(f"  action_id:  {first_action.action_id}")
    print(f"  branch_id:  {first_action.branch_id}")
    print(f"  branch_pos: {first_action.branch_pos}")

    result_1 = env.step(first_action)

    print("\nStep 1 result:")
    print(f"  reward:             {result_1.reward:.4f}")
    print(f"  done:               {result_1.done}")
    print(f"  solved:             {result_1.solved}")
    print(f"  power_flow_success: {result_1.power_flow_success}")
    print(f"  termination_reason: {result_1.info['termination_reason']}")

    print_state("Environment state after step 1", env)

    if env.done:
        print("\nEpisode ended after step 1.")
        return

    # Choose the currently most loaded valid switch action.
    # This is not a good planner - it is only a check that the env can step twice.
    state_1 = env.current_state
    assert state_1 is not None

    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    switch_actions = [
        action
        for action in env.valid_actions()
        if action.action_type == "switch_off_branch"
    ]

    second_action = max(
        switch_actions,
        key=lambda action: float(
            state_1.branch_features[action.branch_pos, loading_idx]
        ),
    )

    print("\nSecond action selected by naive max-loading heuristic:")
    print(f"  action_id:  {second_action.action_id}")
    print(f"  branch_id:  {second_action.branch_id}")
    print(f"  branch_pos: {second_action.branch_pos}")
    print(
        "  loading:    "
        f"{float(state_1.branch_features[second_action.branch_pos, loading_idx]):.2f}%"
    )

    result_2 = env.step(second_action)

    print("\nStep 2 result:")
    print(f"  reward:             {result_2.reward:.4f}")
    print(f"  done:               {result_2.done}")
    print(f"  solved:             {result_2.solved}")
    print(f"  power_flow_success: {result_2.power_flow_success}")
    print(f"  termination_reason: {result_2.info['termination_reason']}")

    print_state("Environment state after step 2", env)

    print("\nDone.")


if __name__ == "__main__":
    main()