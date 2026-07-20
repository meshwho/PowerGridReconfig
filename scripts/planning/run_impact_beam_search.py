from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMAdapter,
    GridFMState,
)
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.impact_beam_search import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchNode,
    ImpactBeamSearchPlanner,
)
from grid_topology_ai.search.impact_beam_search import safety_score
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

def _column_index(name: str) -> int | None:
    if name not in BRANCH_FEATURE_COLUMNS:
        return None

    return BRANCH_FEATURE_COLUMNS.index(name)


def print_state_snapshot(
    title: str,
    state: GridFMState,
    top_n: int = 15,
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    metrics = state.metrics

    print(f"  max_loading_percent:          {metrics.get('max_loading_percent', float('nan')):.4f}")
    print(f"  mean_loading_percent:         {metrics.get('mean_loading_percent', float('nan')):.4f}")
    print(f"  num_overloaded_branches:      {metrics.get('num_overloaded_branches', 'n/a')}")
    print(f"  num_hard_overloaded_branches: {metrics.get('num_hard_overloaded_branches', 'n/a')}")
    print(f"  min_vm_pu:                    {metrics.get('min_vm_pu', float('nan')):.4f}")
    print(f"  max_vm_pu:                    {metrics.get('max_vm_pu', float('nan')):.4f}")
    print(f"  num_low_voltage_buses:        {metrics.get('num_low_voltage_buses', 'n/a')}")
    print(f"  num_high_voltage_buses:       {metrics.get('num_high_voltage_buses', 'n/a')}")
    print(f"  total_voltage_violation:      {metrics.get('total_voltage_violation', float('nan')):.6f}")
    print(f"  num_outaged_branches:         {metrics.get('num_outaged_branches', 'n/a')}")
    print(f"  outaged_branch_ids:           {state.outaged_branch_ids}")
    print(f"  safety_score:                 {safety_score(state):.4f}")
    branch_features = state.branch_features

    idx_col = _column_index("idx")
    from_col = _column_index("from_bus")
    to_col = _column_index("to_bus")
    status_col = _column_index("br_status")
    loading_col = _column_index("loading_percent")
    rate_col = _column_index("rate_a")

    if status_col is None or loading_col is None:
        print("\nCannot print top loaded branches: required columns are missing.")
        return

    status = branch_features[:, status_col]
    loading = branch_features[:, loading_col]

    active_positions = np.flatnonzero(status > 0.0)

    if active_positions.size == 0:
        print("\nNo active branches.")
        return

    order = active_positions[np.argsort(loading[active_positions])[::-1]]
    order = order[: max(int(top_n), 0)]

    print(f"\nTop {len(order)} active branches by loading:")
    print(
        " rank | branch_pos | branch_id | loading % | rate_a"
    )
    print("-" * 58)

    for rank, branch_pos in enumerate(order, start=1):
        row = branch_features[int(branch_pos)]

        branch_id = int(row[idx_col]) if idx_col is not None else int(branch_pos)
        from_bus = int(row[from_col]) if from_col is not None else -1
        to_bus = int(row[to_col]) if to_col is not None else -1
        rate_a = float(row[rate_col]) if rate_col is not None else float("nan")

        print(
            f"{rank:>5} | "
            f"{int(branch_pos):>10} | "
            f"{branch_id:>9} | "
            f"{float(row[loading_col]):>9.3f} | "
            f"{rate_a:>7.2f}"
        )

def print_node(title: str, node: ImpactBeamSearchNode) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(f"Sequence:              {node.short_sequence()}")
    print(f"Action IDs:            {node.action_ids}")
    print(f"Branch IDs:            {node.branch_ids}")
    print(f"Rewards:               {[round(x, 4) for x in node.rewards]}")
    print(f"Impact scores:         {[round(x, 4) for x in node.impact_scores]}")
    print(f"Cumulative score:      {node.cumulative_score:.4f}")
    print(f"Discounted score:      {node.discounted_score:.4f}")
    print(f"Safety score:          {node.safety_score:.4f}")
    print(f"Total hard overload:   {node.total_hard_overload:.4f}")
    print(f"Squared hard overload: {node.squared_hard_overload:.4f}")
    print(f"Total overload:        {node.total_overload:.4f}")
    print(f"Depth:                 {node.depth}")
    print(f"Done:                  {node.done}")
    print(f"Solved:                {node.solved}")
    print(f"Termination reason:    {node.termination_reason}")

    state = node.env.current_state

    if state is not None:
        print("\nFinal state metrics:")
        print(f"  max_loading_percent:          {state.metrics['max_loading_percent']:.4f}")
        print(f"  mean_loading_percent:         {state.metrics['mean_loading_percent']:.4f}")
        print(f"  num_overloaded_branches:      {state.metrics['num_overloaded_branches']}")
        print(f"  num_hard_overloaded_branches: {state.metrics['num_hard_overloaded_branches']}")
        print(f"  num_outaged_branches:         {state.metrics['num_outaged_branches']}")
        print(f"  outaged_branch_ids:           {state.outaged_branch_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run impact-aware beam search for topology switching."
    )

    parser.add_argument("raw_dir", type=str)

    parser.add_argument("--scenario", type=int, required=True)

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=20)

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=120,
        help="Number of loading-prefiltered actions to impact-test. 0 means all valid actions.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of impact-tested actions kept per node.",
    )

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--pf-alg", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)

    parser.add_argument(
        "--disable-cache",
        action="store_true",
    )

    parser.add_argument(
        "--show-initial-top-n",
        type=int,
        default=15,
        help="Number of initially most loaded active branches to print.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable dynamic progress bar.",
    )

    parser.add_argument(
        "--allow-hard-count-increase",
        action="store_true",
        help=(
            "Allow actions that increase the number of hard-overloaded branches. "
            "By default this is disabled for safer teacher search."
        ),
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print("=" * 100)
    print("Running impact-aware beam search")
    print("=" * 100)
    print(f"Raw directory:     {raw_dir.resolve()}")
    print(f"Scenario:          {args.scenario}")
    print(f"Depth:             {args.depth}")
    print(f"Beam width:        {args.beam_width}")
    print(f"Candidate pool:    {args.candidate_pool}")
    print(f"Top-K actions:     {args.top_k}")
    print(f"Gamma:             {args.gamma}")
    print(f"PF algorithm:      {args.pf_alg}")
    print(f"Cache enabled:     {not args.disable_cache}")
    print(f"Progress bar:      {not args.no_progress}")

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=__import__("dataclasses").replace(__import__("grid_topology_ai.config.physics", fromlist=["DEFAULT_PHYSICS_CONFIG"]).DEFAULT_PHYSICS_CONFIG, pf_alg=args.pf_alg),
        enable_cache=not args.disable_cache,
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not args.disable_cache,
    )

    reward_fn = GridFMReward()

    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=args.max_steps,
    )

    initial_state = env.reset(args.scenario)

    print_state_snapshot(
        title="Initial state before impact beam search",
        state=initial_state,
        top_n=args.show_initial_top_n,
    )

    config = ImpactBeamSearchConfig(
        max_depth=args.depth,
        beam_width=args.beam_width,
        candidate_pool_size=args.candidate_pool,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        include_stop_action=True,
        allow_hard_count_increase=args.allow_hard_count_increase,
        show_progress=not args.no_progress,
        progress_update_every=1,
    )

    planner = ImpactBeamSearchPlanner(config)

    result = planner.search(
        env=env,
        scenario_id=args.scenario,
    )

    print_node("Best sequence found", result.best_node)

    print("\n" + "=" * 100)
    print("Final beam")
    print("=" * 100)

    for i, node in enumerate(result.final_beam, start=1):
        print(
            f"{i:2d}. "
            f"seq={node.short_sequence():40s} | "
            f"safety={node.safety_score:10.4f} | "
            f"score={node.discounted_score:10.4f} | "
            f"hard={node.num_hard_overloaded:3d} | "
            f"hard_sum={node.total_hard_overload:8.3f} | "
            f"hard_sq={node.squared_hard_overload:10.3f} | "
            f"max={node.max_loading_percent:8.3f}% | "
            f"over={node.num_overloaded:3d} | "
            f"over_sum={node.total_overload:8.3f} | "
            f"solved={str(node.solved):5s} | "
            f"done={str(node.done):5s}"
        )

    print("\nCaches:")
    print("  Power flow:", backend.cache_info())
    print("  Action space:", action_space.cache_info())

    print(f"\nEvaluated actions: {result.evaluated_actions}")
    print("\nDone.")


if __name__ == "__main__":
    main()