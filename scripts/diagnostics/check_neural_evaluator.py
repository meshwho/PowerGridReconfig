from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check trained neural policy-value evaluator."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy-value checkpoint.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=7,
        help="Scenario ID to evaluate.",
    )

    parser.add_argument(
        "--top-actions",
        type=int,
        default=10,
        help="Number of top policy actions to print.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Checking neural policy-value evaluator")
    print("=" * 100)

    raw_dir = Path(args.raw_dir)
    checkpoint = Path(args.checkpoint)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Checkpoint:    {checkpoint.resolve()}")
    print(f"Scenario:      {args.scenario}")

    adapter = GridFMAdapter(
        raw_dir,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )
    action_space = GridFMActionSpace(require_connected_after_switch=True)
    evaluator = NeuralPolicyValueEvaluator(
        checkpoint,
        physics_config=DEFAULT_PHYSICS_CONFIG,
    )

    state = adapter.build_state(args.scenario)
    action_mask = action_space.valid_action_mask(state)

    policy, value = evaluator.evaluate(
        state=state,
        action_mask=action_mask,
    )

    print("\nState metrics:")
    print(f"  max_loading_percent:          {state.metrics['max_loading_percent']:.4f}")
    print(f"  num_overloaded_branches:      {state.metrics['num_overloaded_branches']}")
    print(f"  num_hard_overloaded_branches: {state.metrics['num_hard_overloaded_branches']}")
    print(f"  outaged_branch_ids:           {state.outaged_branch_ids}")

    print("\nNeural value:")
    print(f"  value={value:+.4f}")

    print("\nTop neural policy actions:")
    top_indices = policy.argsort()[::-1][: args.top_actions]
    all_actions = action_space.build_all_actions(state)

    for action_id in top_indices:
        action = all_actions[int(action_id)]
        print(
            f"  action={int(action_id):>3} | "
            f"type={action.action_type:<17} | "
            f"branch={str(action.branch_id):>4} | "
            f"p={float(policy[action_id]):.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
