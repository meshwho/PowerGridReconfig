from __future__ import annotations

import argparse
from pathlib import Path

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.evaluation.checkpoint import (
    EvaluationRequest,
    evaluate_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a neural policy-value checkpoint with deterministic MCTS, "
            "optionally comparing ungated and constrained root policies."
        )
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
        "--checkpoint",
        type=str,
        required=True,
        help="Policy-value checkpoint to evaluate.",
    )
    parser.add_argument(
        "--use-continuation-gate",
        action="store_true",
        help=(
            "Evaluate both ungated MCTS and a constrained MCTS policy formed by "
            "filtering root visits with continuation analysis."
        ),
    )
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
    parser.add_argument("--simulations", type=int, default=150)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--prior-exponent", type=float, default=0.5)
    parser.add_argument(
        "--stop-policy",
        type=str,
        default="no_hard_overloads",
        choices=["never", "solved_only", "no_hard_overloads", "always"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Neural evaluator device: cpu, cuda, or auto depending on evaluator support.",
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
        "--allow-handoff-with-hard-overloads",
        action="store_true",
        help="Treat action 0 as redispatch handoff even when hard overloads remain.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. Use 1 for old sequential behavior. "
            "On CUDA start with 2 workers to avoid GPU memory issues."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of scenarios per worker task.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N scenarios from transitions CSV.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print one line per scenario. Much faster on Windows PowerShell.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save per-scenario evaluation results.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save machine-readable evaluation summary metrics.",
    )
    parser.add_argument(
        "--clear-caches-every",
        type=int,
        default=100,
        help=(
            "Clear backend/action/evaluator caches after this many scenarios per worker. "
            "Use 0 to never clear caches."
        ),
    )
    parser.add_argument(
        "--use-dc-screening",
        action="store_true",
        help=(
            "Enable optional DC power-flow screening for MCTS switch candidates. "
            "Disabled by default, so old behavior is preserved."
        ),
    )
    parser.add_argument(
        "--dc-top-k",
        type=int,
        default=30,
        help="Number of DC-ranked switch actions to keep at each MCTS node.",
    )
    parser.add_argument(
        "--dc-candidate-pool",
        type=int,
        default=120,
        help=(
            "Number of neural-policy actions considered by DC screening. "
            "Use <=0 to screen all valid switch actions."
        ),
    )
    parser.add_argument(
        "--dc-keep-policy-actions",
        type=int,
        default=5,
        help="Always keep this many pure neural-policy actions as backup.",
    )
    parser.add_argument(
        "--dc-keep-loading-actions",
        type=int,
        default=5,
        help="Always keep this many high-loading actions as backup.",
    )
    parser.add_argument(
        "--dc-policy-weight",
        type=float,
        default=0.0,
        help=(
            "Optional neural-prior tie-breaker inside DC ranking. "
            "0 means pure DC physical ranking."
        ),
    )
    parser.add_argument(
        "--dc-failure-penalty",
        type=float,
        default=1_000_000_000.0,
        help="Penalty assigned to DC PF failures.",
    )
    parser.add_argument(
        "--dc-max-depth",
        type=int,
        default=0,
        help=(
            "Maximum MCTS node depth where DC screening is used. "
            "0 means root only, 1 means root and depth-1 nodes, -1 means all depths."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EvaluationConfig(
        simulations=args.simulations,
        depth=args.depth,
        max_steps=args.max_steps,
        top_k=args.top_k,
        gamma=args.gamma,
        c_puct=args.c_puct,
        prior_exponent=args.prior_exponent,
        use_continuation_gate=args.use_continuation_gate,
        allow_handoff_with_hard_overloads=args.allow_handoff_with_hard_overloads,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        device=args.device,
        pf_alg=args.pf_alg,
    )
    request = EvaluationRequest(
        raw_dir=Path(args.raw_dir),
        transitions_csv=Path(args.transitions),
        checkpoint=Path(args.checkpoint),
        config=config,
        output_csv=None if args.output_csv is None else Path(args.output_csv),
        output_json=None if args.output_json is None else Path(args.output_json),
        limit=args.limit,
        quiet=args.quiet,
        disable_cache=args.disable_cache,
        stop_policy=args.stop_policy,
        min_hard_improvement=args.min_hard_improvement,
        min_soft_improvement=args.min_soft_improvement,
        min_gate_visits=args.min_gate_visits,
        min_gate_visit_fraction=args.min_gate_visit_fraction,
        clear_caches_every=args.clear_caches_every,
        use_dc_screening=args.use_dc_screening,
        dc_top_k=args.dc_top_k,
        dc_candidate_pool=args.dc_candidate_pool,
        dc_keep_policy_actions=args.dc_keep_policy_actions,
        dc_keep_loading_actions=args.dc_keep_loading_actions,
        dc_policy_weight=args.dc_policy_weight,
        dc_failure_penalty=args.dc_failure_penalty,
        dc_max_depth=args.dc_max_depth,
    )
    evaluate_checkpoint(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
