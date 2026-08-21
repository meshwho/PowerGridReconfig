from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
        default=1.0,
        help=(
            "Terminal-utility gamma. The current contract requires 1.0."
        ),
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
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
        help="When MCTS is allowed to use the stop/handoff action.",
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
            "Action-sampling temperature used during "
            "the configured early iterations and steps."
        ),
    )
    parser.add_argument(
        "--temperature-steps",
        type=int,
        default=0,
        help=(
            "Number of early episode steps that use "
            "the positive selection temperature."
        ),
    )
    parser.add_argument(
        "--temperature-iterations",
        type=int,
        default=0,
        help=(
            "Number of early self-play iterations that "
            "use temperature-based action sampling."
        ),
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help=(
            "One-based self-play iteration number used "
            "to resolve the temperature schedule."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base seed used to derive independent "
            "MCTS and action-sampling streams."
        ),
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
    parser.add_argument(
        "--closeable-branch-id",
        dest="closeable_branch_ids",
        action="append",
        type=int,
        default=None,
        help=(
            "Branch ID that may be closed when currently inactive. "
            "Repeat the option for multiple normally-open or tie branches."
        ),
    )
    parser.add_argument(
        "--min-loading-for-switch-percent",
        type=float,
        default=0.0,
        help=(
            "Minimum loading percentage for branch-opening candidates. "
            "This threshold does not filter branch-closing actions."
        ),
    )
    connectivity = parser.add_mutually_exclusive_group()
    connectivity.add_argument(
        "--require-connected-after-switch",
        dest="require_connected_after_switch",
        action="store_true",
        help="Mask branch openings that disconnect the active grid.",
    )
    connectivity.add_argument(
        "--no-require-connected-after-switch",
        dest="require_connected_after_switch",
        action="store_false",
        help="Allow branch openings even when the active grid disconnects.",
    )
    parser.set_defaults(require_connected_after_switch=True)

    continuation_gate = parser.add_mutually_exclusive_group()
    continuation_gate.add_argument(
        "--use-continuation-gate",
        dest="use_continuation_gate",
        action="store_true",
        help=(
            "Run continuation analysis for diagnostics; it does not override "
            "the executed self-play action or policy target."
        ),
    )
    continuation_gate.add_argument(
        "--no-use-continuation-gate",
        dest="use_continuation_gate",
        action="store_false",
        help="Disable continuation diagnostics.",
    )
    parser.set_defaults(use_continuation_gate=False)
    parser.add_argument(
        "--min-hard-improvement",
        type=float,
        default=50.0,
        help="Minimum diagnostic improvement while hard overloads exist.",
    )
    parser.add_argument(
        "--min-soft-improvement",
        type=float,
        default=15.0,
        help="Minimum diagnostic improvement after hard overloads are cleared.",
    )
    parser.add_argument(
        "--min-gate-visits",
        type=int,
        default=5,
        help="Minimum visits required for continuation diagnostics.",
    )
    parser.add_argument(
        "--min-gate-visit-fraction",
        type=float,
        default=0.01,
        help="Minimum root policy fraction for continuation diagnostics.",
    )
    parser.add_argument(
        "--clear-cache-between-scenarios",
        action="store_true",
        help=(
            "Clear power flow/action/evaluator caches before each scenario. "
            "Useful for large self-play generation to avoid unbounded memory growth."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of persistent self-play worker processes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the matching generation run from progress.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_sequence = np.random.SeedSequence(
        [
            int(args.seed),
            int(args.iteration),
        ]
    )

    # Keep the same child positions as the full self-play loop.
    (
        _,
        mcts_sequence,
        action_sequence,
    ) = root_sequence.spawn(3)

    mcts_seed = int(
        mcts_sequence.generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
    action_seed = int(
        action_sequence.generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
    config = GenerationConfig(
        simulations=args.simulations,
        depth=args.depth,
        max_steps=args.max_steps,
        top_k=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        prior_exponent=args.prior_exponent,
        selection_temperature=(
            args.selection_temperature
        ),
        temperature_steps=args.temperature_steps,
        temperature_iterations=(
            args.temperature_iterations
        ),
        use_root_noise=args.use_root_noise,
        use_continuation_gate=args.use_continuation_gate,
        pf_alg=args.pf_alg,
        stop_policy=args.stop_policy,
        require_connected_after_switch=(
            args.require_connected_after_switch
        ),
        min_loading_for_switch_percent=(
            args.min_loading_for_switch_percent
        ),
        closeable_branch_ids=tuple(
            args.closeable_branch_ids or ()
        ),
    )
    request = GenerationRequest(
        raw_dir=Path(args.raw_dir),
        transitions_csv=Path(args.transitions),
        output_dir=Path(args.output_dir),
        checkpoint=(None if args.checkpoint is None else Path(args.checkpoint)),
        config=config,
        mcts_seed=mcts_seed,
        action_seed=action_seed,
        iteration=args.iteration,
        clear_cache_between_scenarios=args.clear_cache_between_scenarios,
        device=args.device,
        enable_cache=not args.disable_cache,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        min_hard_improvement=args.min_hard_improvement,
        min_soft_improvement=args.min_soft_improvement,
        min_gate_visits=args.min_gate_visits,
        min_gate_visit_fraction=args.min_gate_visit_fraction,
        workers=args.workers,
        resume=args.resume,
    )

    examples_csv = generate_self_play_examples(request)
    print(examples_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
