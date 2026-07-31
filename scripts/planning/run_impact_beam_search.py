from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMAdapter, GridFMState
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.impact_beam_search import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchNode,
    ImpactBeamSearchPlanner,
    safety_score,
)
from scripts.topology_action_cli import (
    action_space_kwargs,
    add_topology_action_arguments,
    print_topology_action_config,
    topology_action_config_from_args,
)


def _column_index(name: str) -> int | None:
    return None if name not in BRANCH_FEATURE_COLUMNS else BRANCH_FEATURE_COLUMNS.index(name)


def print_state_snapshot(
    title: str,
    state: GridFMState,
    top_n: int = 15,
    physics_config: PhysicsConfig | None = None,
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for name in (
        "max_loading_percent", "mean_loading_percent",
        "num_overloaded_branches", "num_hard_overloaded_branches",
        "min_vm_pu", "max_vm_pu", "num_low_voltage_buses",
        "num_high_voltage_buses", "total_voltage_violation",
        "num_outaged_branches",
    ):
        print(f"  {name}: {state.metrics.get(name, 'n/a')}")
    print(f"  outaged_branch_ids: {state.outaged_branch_ids}")
    print(f"  safety_score: {safety_score(state, physics_config=physics_config):.4f}")

    status_col = _column_index("br_status")
    loading_col = _column_index("loading_percent")
    idx_col = _column_index("idx")
    if status_col is None or loading_col is None:
        return
    positions = np.flatnonzero(state.branch_features[:, status_col] > 0.0)
    loading = state.branch_features[:, loading_col]
    positions = positions[np.argsort(loading[positions])[::-1]][:max(int(top_n), 0)]
    print(f"\nTop {len(positions)} active branches by loading:")
    for rank, branch_pos in enumerate(positions, start=1):
        row = state.branch_features[int(branch_pos)]
        branch_id = int(row[idx_col]) if idx_col is not None else int(branch_pos)
        print(
            f"{rank:>3}. branch_pos={int(branch_pos):>4} | "
            f"branch_id={branch_id:>4} | loading={float(row[loading_col]):>9.3f}%"
        )


def print_node(title: str, node: ImpactBeamSearchNode) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Sequence:              {node.short_sequence()}")
    print(f"Action IDs:            {node.action_ids}")
    print(f"Branch IDs:            {node.branch_ids}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run impact-aware beam search for topology switching."
    )
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--pf-alg", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--show-initial-top-n", type=int, default=15)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--allow-hard-count-increase", action="store_true")
    add_topology_action_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    cache_enabled = not args.disable_cache
    topology_config = topology_action_config_from_args(args)

    print("=" * 100)
    print("Running impact-aware beam search")
    print("=" * 100)
    print(f"Raw directory:  {raw_dir.resolve()}")
    print(f"Scenario:       {args.scenario}")
    print(f"Depth:          {args.depth}")
    print(f"Beam width:     {args.beam_width}")
    print(f"Candidate pool: {args.candidate_pool}")
    print(f"Top-K actions:  {args.top_k}")
    print(f"Gamma:          {args.gamma}")
    print(f"PF algorithm:   {args.pf_alg}")
    print(f"Cache enabled:  {cache_enabled}")
    print_topology_action_config(topology_config)

    physics_config = replace(DEFAULT_PHYSICS_CONFIG, pf_alg=args.pf_alg)
    adapter = GridFMAdapter(raw_dir, physics_config=physics_config)
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics_config,
        enable_cache=cache_enabled,
    )
    action_space = GridFMActionSpace(
        **action_space_kwargs(topology_config, enable_cache=cache_enabled)
    )
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=GridFMReward(physics_config=physics_config),
        max_steps=args.max_steps,
    )
    initial_state = env.reset(args.scenario)
    print_state_snapshot(
        "Initial state before impact beam search",
        initial_state,
        top_n=args.show_initial_top_n,
        physics_config=physics_config,
    )
    planner = ImpactBeamSearchPlanner(
        ImpactBeamSearchConfig(
            max_depth=args.depth,
            beam_width=args.beam_width,
            candidate_pool_size=args.candidate_pool,
            top_k_actions=args.top_k,
            gamma=args.gamma,
            include_stop_action=True,
            allow_hard_count_increase=args.allow_hard_count_increase,
            show_progress=not args.no_progress,
            progress_update_every=1,
        ),
        physics_config=physics_config,
    )
    result = planner.search(env=env, scenario_id=args.scenario)
    print_node("Best sequence found", result.best_node)
    print("\nFinal beam")
    for index, node in enumerate(result.final_beam, start=1):
        print(
            f"{index:2d}. seq={node.short_sequence():40s} | "
            f"safety={node.safety_score:10.4f} | score={node.discounted_score:10.4f} | "
            f"hard={node.num_hard_overloaded:3d} | max={node.max_loading_percent:8.3f}% | "
            f"solved={str(node.solved):5s} | done={str(node.done):5s}"
        )
    print("\nCaches:")
    print("  Power flow:", backend.cache_info())
    print("  Action space:", action_space.cache_info())
    print(f"\nEvaluated actions: {result.evaluated_actions}")


if __name__ == "__main__":
    main()
