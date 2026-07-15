from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.impact_beam_search import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
    ImpactBeamSearchResult,
    safety_score,
)
from grid_topology_ai.self_play.examples import ExampleWriter


def make_one_hot_policy(action_id: int) -> dict[int, float]:
    return {int(action_id): 1.0}


def make_policy_from_final_beam(
    result: ImpactBeamSearchResult,
    temperature: float,
) -> tuple[dict[int, float], dict[int, int]]:
    """
    Convert final beam into a policy target over FIRST actions.

    temperature <= 0:
        deterministic one-hot policy for the best sequence first action.

    temperature > 0:
        soft policy. Sequences close to the best safety score get higher weight.

    This gives the neural network a richer target than a single hard label,
    while still being based on physics-based impact search.
    """

    best_node = result.best_node

    if not best_node.action_ids:
        return {}, {}

    best_action_id = int(best_node.action_ids[0])

    if temperature <= 1e-12:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    best_safety = float(best_node.safety_score)

    weights_by_action: dict[int, float] = {}
    counts_by_action: dict[int, int] = {}

    for node in result.final_beam:
        if not node.action_ids:
            continue

        action_id = int(node.action_ids[0])

        safety_gap = max(float(node.safety_score) - best_safety, 0.0)
        weight = float(np.exp(-safety_gap / float(temperature)))

        weights_by_action[action_id] = weights_by_action.get(action_id, 0.0) + weight
        counts_by_action[action_id] = counts_by_action.get(action_id, 0) + 1

    total = float(sum(weights_by_action.values()))

    if total <= 0.0:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    policy = {
        int(action_id): float(weight / total)
        for action_id, weight in weights_by_action.items()
    }

    return policy, counts_by_action


def iter_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    transitions = pd.read_csv(transitions_path)

    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions file must contain scenario_id column: {transitions_path}"
        )

    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]

    return scenario_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate teacher examples using safety-aware impact beam search."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw directory.",
    )

    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV with scenario_id column.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for teacher examples.",
    )

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=70)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=5)

    parser.add_argument(
        "--soft-policy-temperature",
        type=float,
        default=500.0,
        help=(
            "Temperature for converting final beam into soft policy. "
            "Use 0 for deterministic one-hot target."
        ),
    )

    parser.add_argument(
        "--min-safety-improvement",
        type=float,
        default=0.0,
        help="Skip scenario if best final safety improvement is below this value.",
    )

    parser.add_argument(
        "--allow-hard-count-increase",
        action="store_true",
        help="Allow beam search to increase hard-overload count.",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable power flow and action-space caches.",
    )

    parser.add_argument(
        "--clear-cache-between-scenarios",
        action="store_true",
        help="Clear backend/action-space caches before every scenario.",
    )

    parser.add_argument(
        "--show-search-progress",
        action="store_true",
        help="Show internal progress bar for every impact beam search.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    scenario_ids = iter_scenario_ids(
        transitions_path=transitions_path,
        limit=args.limit,
    )

    print("=" * 100)
    print("Generating impact-beam teacher examples")
    print("=" * 100)
    print(f"Raw directory:        {raw_dir.resolve()}")
    print(f"Transitions:          {transitions_path.resolve()}")
    print(f"Output dir:           {args.output_dir}")
    print(f"Scenarios:            {len(scenario_ids)}")
    print(f"Depth:                {args.depth}")
    print(f"Beam width:           {args.beam_width}")
    print(f"Candidate pool:       {args.candidate_pool}")
    print(f"Top-K actions:        {args.top_k}")
    print(f"Gamma:                {args.gamma}")
    print(f"PF algorithm:         {args.pf_alg}")
    print(f"Soft policy temp:     {args.soft_policy_temperature}")
    print(f"Min safety improve:   {args.min_safety_improvement}")
    print(f"Allow hard increase:  {args.allow_hard_count_increase}")
    print(f"Cache enabled:        {not args.disable_cache}")
    print(f"Clear cache/scenario: {args.clear_cache_between_scenarios}")

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=args.pf_alg,
        enable_cache=not args.disable_cache,
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not args.disable_cache,
    )

    reward_fn = GridFMReward()

    planner_config = ImpactBeamSearchConfig(
        max_depth=args.depth,
        beam_width=args.beam_width,
        candidate_pool_size=args.candidate_pool,
        top_k_actions=args.top_k,
        gamma=args.gamma,
        include_stop_action=True,
        allow_hard_count_increase=args.allow_hard_count_increase,
        show_progress=args.show_search_progress,
        progress_update_every=1,
    )

    planner = ImpactBeamSearchPlanner(planner_config)
    example_writer = ExampleWriter(args.output_dir)

    total_saved = 0
    total_skipped = 0

    scenario_iter = scenario_ids

    if tqdm is not None:
        scenario_iter = tqdm(
            scenario_ids,
            desc="Teacher scenarios",
            unit="scenario",
            dynamic_ncols=True,
        )

    for scenario_id in scenario_iter:
        if args.clear_cache_between_scenarios:
            backend.clear_cache()
            action_space.clear_cache()

        env = TopologySwitchingEnv(
            adapter=adapter,
            backend=backend,
            action_space=action_space,
            reward_fn=reward_fn,
            max_steps=args.max_steps,
        )

        state = env.reset(int(scenario_id))
        action_mask = env.valid_action_mask()

        initial_safety = safety_score(state)

        result = planner.search(
            env=env,
            scenario_id=int(scenario_id),
        )

        best = result.best_node

        if not best.action_ids:
            total_skipped += 1
            print(
                f"Scenario {scenario_id}: skipped, no teacher action found."
            )
            continue

        selected_action_id = int(best.action_ids[0])
        selected_branch_id = best.branch_ids[0]

        final_safety = float(best.safety_score)
        safety_improvement = float(initial_safety - final_safety)

        if safety_improvement < float(args.min_safety_improvement):
            total_skipped += 1
            print(
                f"Scenario {scenario_id}: skipped, "
                f"safety improvement {safety_improvement:.4f} "
                f"< {args.min_safety_improvement:.4f}."
            )
            continue

        policy_target, visit_counts = make_policy_from_final_beam(
            result=result,
            temperature=args.soft_policy_temperature,
        )

        if not policy_target:
            total_skipped += 1
            print(
                f"Scenario {scenario_id}: skipped, empty policy target."
            )
            continue

        first_step_reward = (
            float(best.impact_scores[0])
            if best.impact_scores
            else float(safety_improvement)
        )

        state_id = f"impact_teacher_scenario_{scenario_id:06d}_step_000"

        example_writer.add_example(
            state=state,
            state_id=state_id,
            action_mask=action_mask,
            scenario_id=int(scenario_id),
            step=0,
            selected_action_id=selected_action_id,
            selected_branch_id=selected_branch_id,
            step_reward=first_step_reward,
            final_return=float(safety_improvement),
            discounted_return_from_step=float(safety_improvement),
            solved=bool(best.solved),
            done=bool(best.done),
            termination_reason=best.termination_reason or "teacher_depth_limit",
            visit_counts=visit_counts,
            mcts_policy=policy_target,
            extra_metadata={
                "source": "impact_beam_teacher",
                "scenario_id": int(scenario_id),
                "initial_safety": float(initial_safety),
                "final_safety": float(final_safety),
                "safety_improvement": float(safety_improvement),
                "best_sequence_action_ids": [int(x) for x in best.action_ids],
                "best_sequence_branch_ids": [
                    None if x is None else int(x)
                    for x in best.branch_ids
                ],
                "best_max_loading_percent": float(best.max_loading_percent),
                "best_num_hard_overloaded": int(best.num_hard_overloaded),
                "best_num_overloaded": int(best.num_overloaded),
                "beam_depth": int(args.depth),
                "beam_width": int(args.beam_width),
                "candidate_pool": int(args.candidate_pool),
                "top_k": int(args.top_k),
                "soft_policy_temperature": float(args.soft_policy_temperature),
            },
        )

        total_saved += 1

        print(
            f"Scenario {scenario_id}: saved | "
            f"action={selected_action_id} | "
            f"branch={selected_branch_id} | "
            f"safety {initial_safety:.2f} -> {final_safety:.2f} | "
            f"improvement={safety_improvement:.2f} | "
            f"hard={best.num_hard_overloaded} | "
            f"max={best.max_loading_percent:.2f}% | "
            f"seq={best.short_sequence()}"
        )

    examples_path = example_writer.save()

    print("\nPower flow cache:")
    print(backend.cache_info())

    print("\nAction space cache:")
    print(action_space.cache_info())

    print("\n" + "=" * 100)
    print("Impact teacher generation summary")
    print("=" * 100)
    print(f"Saved examples:  {total_saved}")
    print(f"Skipped:         {total_skipped}")
    print(f"Examples CSV:    {examples_path}")
    print(f"States dir:      {example_writer.states_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()