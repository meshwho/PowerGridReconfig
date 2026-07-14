from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    generate_self_play_examples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AlphaZero-like self-play data using MCTS."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw output directory.",
    )
    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV used only to select scenario IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/self_play/mcts_v0",
        help="Output directory for self-play examples.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=300,
        help="Number of MCTS simulations per decision.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="MCTS depth per decision.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum real episode steps.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K actions considered by MCTS.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="PUCT exploration constant.",
    )
    parser.add_argument(
        "--prior-exponent",
        type=float,
        default=0.5,
        help="Exponent for heuristic prior smoothing.",
    )
    parser.add_argument(
        "--terminal-unsolved-penalty",
        type=float,
        default=500.0,
        help="Terminal penalty added when an episode ends without solving the grid.",
    )
    parser.add_argument(
        "--terminal-penalty-weight",
        type=float,
        default=0.10,
        help="Additional penalty weight for remaining final-state violations.",
    )
    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
        help="When MCTS is allowed to use the stop/handoff action.",
    )
    parser.add_argument(
        "--terminal-handoff-penalty",
        type=float,
        default=150.0,
        help="Terminal penalty for handoff_to_redispatch episodes.",
    )
    parser.add_argument(
        "--terminal-failure-penalty",
        type=float,
        default=1000.0,
        help="Terminal penalty for power flow failure.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional neural policy-value checkpoint for neural-guided MCTS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for neural evaluator: cpu or cuda.",
    )
    root_noise = parser.add_mutually_exclusive_group()
    root_noise.add_argument(
        "--use-root-noise",
        dest="use_root_noise",
        action="store_true",
        help="Use AlphaZero-style Dirichlet noise at MCTS root during self-play.",
    )
    root_noise.add_argument(
        "--no-use-root-noise",
        dest="use_root_noise",
        action="store_false",
        help="Disable AlphaZero-style Dirichlet noise at MCTS root.",
    )
    parser.set_defaults(use_root_noise=False)
    parser.add_argument(
        "--root-dirichlet-alpha",
        type=float,
        default=0.30,
        help="Dirichlet alpha for root exploration noise.",
    )
    parser.add_argument(
        "--root-exploration-fraction",
        type=float,
        default=0.25,
        help="Fraction of root prior replaced by Dirichlet noise.",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=0.0,
        help=(
            "Temperature for selecting actions from MCTS policy. "
            "0.0 = deterministic argmax, >0 = sampling for exploration."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for self-play action sampling and root noise.",
    )
    parser.add_argument(
        "--pf-alg",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="PYPOWER power flow algorithm: 1=NR, 2=FDXB, 3=FDBX, 4=GS.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable power flow/action/evaluator caches.",
    )
    continuation_gate = parser.add_mutually_exclusive_group()
    continuation_gate.add_argument(
        "--use-continuation-gate",
        dest="use_continuation_gate",
        action="store_true",
        help="Use lookahead continuation gate to select executed self-play actions.",
    )
    continuation_gate.add_argument(
        "--no-use-continuation-gate",
        dest="use_continuation_gate",
        action="store_false",
        help="Disable lookahead continuation gate.",
    )
    parser.set_defaults(use_continuation_gate=False)
    parser.add_argument(
        "--min-hard-improvement",
        type=float,
        default=50.0,
        help="Minimum penalty improvement required while hard overloads exist.",
    )
    parser.add_argument(
        "--min-soft-improvement",
        type=float,
        default=15.0,
        help="Minimum penalty improvement required after hard overloads are cleared.",
    )
    parser.add_argument(
        "--min-gate-visits",
        type=int,
        default=5,
        help="Minimum visits required for a branch to be trusted by continuation gate.",
    )
    parser.add_argument(
        "--min-gate-visit-fraction",
        type=float,
        default=0.01,
        help="Minimum root policy fraction required for a branch to be trusted.",
    )
    parser.add_argument(
        "--clear-cache-between-scenarios",
        action="store_true",
        help=(
            "Clear power flow/action/evaluator caches before each scenario. "
            "Useful for large self-play generation to avoid unbounded memory growth."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(
        simulations=args.simulations,
        depth=args.depth,
        max_steps=args.max_steps,
        top_k=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        prior_exponent=args.prior_exponent,
        selection_temperature=args.selection_temperature,
        use_root_noise=args.use_root_noise,
        use_continuation_gate=args.use_continuation_gate,
        pf_alg=args.pf_alg,
        stop_policy=args.stop_policy,
        terminal_unsolved_penalty=args.terminal_unsolved_penalty,
        terminal_handoff_penalty=args.terminal_handoff_penalty,
        terminal_failure_penalty=args.terminal_failure_penalty,
        terminal_penalty_weight=args.terminal_penalty_weight,
    )
    request = GenerationRequest(
        raw_dir=Path(args.raw_dir),
        transitions_csv=Path(args.transitions),
        output_dir=Path(args.output_dir),
        checkpoint=(None if args.checkpoint is None else Path(args.checkpoint)),
        config=config,
        seed=args.seed,
        clear_cache_between_scenarios=args.clear_cache_between_scenarios,
        device=args.device,
        enable_cache=not args.disable_cache,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        min_hard_improvement=args.min_hard_improvement,
        min_soft_improvement=args.min_soft_improvement,
        min_gate_visits=args.min_gate_visits,
        min_gate_visit_fraction=args.min_gate_visit_fraction,
    )

    examples_csv = generate_self_play_examples(request)
    print(examples_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
