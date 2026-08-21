from __future__ import annotations

import argparse
from pathlib import Path


def _search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--simulations", type=int, default=150)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--widening-coefficient", type=float, default=2.0)
    parser.add_argument("--widening-exponent", type=float, default=0.5)
    parser.add_argument("--exploration-quota", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--prior-exponent", type=float, default=0.5)
    parser.add_argument("--stop-policy", choices=["never", "solved_only", "no_hard_overloads", "always"], default="no_hard_overloads")
    parser.add_argument("--pf-alg", type=int, choices=[1, 2, 3, 4], default=3)


def _input_arguments(parser: argparse.ArgumentParser, *, checkpoint_required: bool) -> None:
    parser.add_argument("raw_dir", help="GridFM raw data directory.")
    parser.add_argument("--transitions", required=True, help="Transitions CSV used to select scenarios.")
    parser.add_argument("--checkpoint", required=checkpoint_required, help="Policy-value checkpoint.")
    parser.add_argument("--scenario-id", action="append", type=int, dest="scenario_ids", help="Select a scenario ID; repeat to select several.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")


def _add_evaluate(parser: argparse.ArgumentParser) -> None:
    _input_arguments(parser, checkpoint_required=True)
    _search_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-continuation-gate", action="store_true", help="Evaluate the single constrained policy mode.")
    parser.add_argument("--allow-handoff-with-hard-overloads", action="store_true")
    parser.add_argument("--min-hard-improvement", type=float, default=50.0)
    parser.add_argument("--min-soft-improvement", type=float, default=15.0)
    parser.add_argument("--min-gate-visits", type=int, default=5)
    parser.add_argument("--min-gate-visit-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--clear-caches-every", type=int, default=100)
    parser.add_argument("--use-dc-screening", action="store_true")
    parser.add_argument("--dc-top-k", type=int, default=30)
    parser.add_argument("--dc-candidate-pool", type=int, default=120)
    parser.add_argument("--dc-keep-policy-actions", type=int, default=5)
    parser.add_argument("--dc-keep-loading-actions", type=int, default=5)
    parser.add_argument("--dc-policy-weight", type=float, default=0.0)
    parser.add_argument("--dc-failure-penalty", type=float, default=1e9)
    parser.add_argument("--dc-max-depth", type=int, default=0)
    parser.set_defaults(handler=_evaluate)


def _evaluate(args: argparse.Namespace) -> int:
    from grid_topology_ai.config import EvaluationConfig
    from grid_topology_ai.evaluation import EvaluationRequest, evaluate_checkpoint

    config = EvaluationConfig(simulations=args.simulations, depth=args.depth, max_steps=args.max_steps, top_k=args.top_k, widening_coefficient=args.widening_coefficient, widening_exponent=args.widening_exponent, exploration_quota=args.exploration_quota, random_seed=args.seed, gamma=args.gamma, c_puct=args.c_puct, prior_exponent=args.prior_exponent, primary_policy_mode="constrained" if args.use_continuation_gate else "ungated", use_continuation_gate=args.use_continuation_gate, allow_handoff_with_hard_overloads=args.allow_handoff_with_hard_overloads, num_workers=args.workers, batch_size=args.batch_size, device=args.device, pf_alg=args.pf_alg)
    request = EvaluationRequest(raw_dir=Path(args.raw_dir), transitions_csv=Path(args.transitions), checkpoint=Path(args.checkpoint), config=config, output_csv=Path(args.output_csv) if args.output_csv else None, output_json=Path(args.output_json) if args.output_json else None, limit=args.limit, scenario_ids=tuple(args.scenario_ids) if args.scenario_ids else None, quiet=args.quiet, disable_cache=args.disable_cache, stop_policy=args.stop_policy, min_hard_improvement=args.min_hard_improvement, min_soft_improvement=args.min_soft_improvement, min_gate_visits=args.min_gate_visits, min_gate_visit_fraction=args.min_gate_visit_fraction, clear_caches_every=args.clear_caches_every, use_dc_screening=args.use_dc_screening, dc_top_k=args.dc_top_k, dc_candidate_pool=args.dc_candidate_pool, dc_keep_policy_actions=args.dc_keep_policy_actions, dc_keep_loading_actions=args.dc_keep_loading_actions, dc_policy_weight=args.dc_policy_weight, dc_failure_penalty=args.dc_failure_penalty, dc_max_depth=args.dc_max_depth)
    evaluate_checkpoint(request)
    return 0


def _add_self_play(parser: argparse.ArgumentParser) -> None:
    _input_arguments(parser, checkpoint_required=False)
    _search_arguments(parser)
    parser.set_defaults(simulations=300, top_k=40)
    parser.add_argument("--output", default="data/self_play/mcts_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcts-seed", type=int)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--selection-temperature", type=float, default=0.0)
    parser.add_argument("--temperature-steps", type=int, default=0)
    parser.add_argument("--temperature-iterations", type=int, default=0)
    parser.add_argument("--use-root-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.30)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--use-continuation-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-hard-improvement", type=float, default=50.0)
    parser.add_argument("--min-soft-improvement", type=float, default=15.0)
    parser.add_argument("--min-gate-visits", type=int, default=5)
    parser.add_argument("--min-gate-visit-fraction", type=float, default=0.01)
    parser.add_argument("--require-connected-after-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-loading-for-switch-percent", type=float, default=0.0)
    parser.add_argument("--closeable-branch-id", action="append", type=int, default=[])
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--clear-cache-between-scenarios", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(handler=_self_play)


def _self_play(args: argparse.Namespace) -> int:
    import numpy as np
    from grid_topology_ai.config import GenerationConfig
    from grid_topology_ai.self_play.generation import GenerationRequest, generate_self_play_examples

    root = np.random.SeedSequence([args.seed, args.iteration])
    _, mcts, action = root.spawn(3)
    mcts_seed = args.mcts_seed if args.mcts_seed is not None else int(mcts.generate_state(1, dtype=np.uint64)[0])
    action_seed = args.action_seed if args.action_seed is not None else int(action.generate_state(1, dtype=np.uint64)[0])
    config = GenerationConfig(simulations=args.simulations, depth=args.depth, max_steps=args.max_steps, top_k=args.top_k, widening_coefficient=args.widening_coefficient, widening_exponent=args.widening_exponent, exploration_quota=args.exploration_quota, gamma=args.gamma, c_puct=args.c_puct, prior_exponent=args.prior_exponent, selection_temperature=args.selection_temperature, temperature_steps=args.temperature_steps, temperature_iterations=args.temperature_iterations, use_root_noise=args.use_root_noise, use_continuation_gate=args.use_continuation_gate, device=args.device, pf_alg=args.pf_alg, stop_policy=args.stop_policy, require_connected_after_switch=args.require_connected_after_switch, min_loading_for_switch_percent=args.min_loading_for_switch_percent, closeable_branch_ids=tuple(args.closeable_branch_id))
    request = GenerationRequest(raw_dir=Path(args.raw_dir), transitions_csv=Path(args.transitions), output_dir=Path(args.output), checkpoint=Path(args.checkpoint) if args.checkpoint else None, config=config, mcts_seed=mcts_seed, action_seed=action_seed, clear_cache_between_scenarios=args.clear_cache_between_scenarios, iteration=args.iteration, scenario_ids=tuple(args.scenario_ids) if args.scenario_ids else None, device=args.device, enable_cache=not args.disable_cache, root_dirichlet_alpha=args.root_dirichlet_alpha, root_exploration_fraction=args.root_exploration_fraction, min_hard_improvement=args.min_hard_improvement, min_soft_improvement=args.min_soft_improvement, min_gate_visits=args.min_gate_visits, min_gate_visit_fraction=args.min_gate_visit_fraction, workers=args.workers, resume=args.resume)
    print(generate_self_play_examples(request))
    return 0


def _add_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("examples", help="Training examples CSV.")
    parser.add_argument("--validation", help="Optional validation examples CSV.")
    parser.add_argument("--output", default="data/self_play/graph_v2/graph_policy_value_net.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-huber-delta", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--save-multiple-best", action="store_true")
    parser.add_argument("--tensorboard-log-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.set_defaults(handler=_train)


def _train(args: argparse.Namespace) -> int:
    from grid_topology_ai.config import TrainingConfig
    from grid_topology_ai.training import TrainingRequest, train_graph_policy_value_model

    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, value_loss_weight=args.value_loss_weight, value_huber_delta=args.value_huber_delta, num_workers=args.workers, device=args.device, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, save_multiple_best=args.save_multiple_best, no_tensorboard=args.no_tensorboard)
    request = TrainingRequest(project_root=Path.cwd().resolve(), examples_csv=Path(args.examples), output_path=Path(args.output), config=config, init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None, resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None, validation_examples_csv=Path(args.validation) if args.validation else None, use_amp=args.amp, normalize_features=not args.no_normalize_features, save_best=args.save_best, tensorboard_log_dir=Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else None, run_name=args.run_name, metrics_csv=Path(args.metrics_csv) if args.metrics_csv else None, seed=args.seed)
    print(train_graph_policy_value_model(request))
    return 0


def _add_teacher(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("raw_dir")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--pf-alg", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--pf-max-iter", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-teacher-steps", type=int, default=4)
    parser.add_argument("--soft-policy-temperature", type=float, default=0.0)
    parser.add_argument("--use-soft-root-policy", action="store_true")
    parser.add_argument("--allow-hard-count-increase", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--power-flow-failure-penalty", type=float, default=1e6)
    parser.add_argument("--min-continue-improvement-with-hard", type=float, default=100.0)
    parser.add_argument("--min-continue-improvement-without-hard", type=float, default=150.0)
    parser.add_argument("--max-loading-increase-limit", type=float, default=5.0)
    parser.add_argument("--add-handoff-example", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use-lodf-screening", action="store_true")
    parser.add_argument("--lodf-screen-top-k", type=int, default=0)
    parser.add_argument("--lodf-min-candidate-count", type=int, default=8)
    parser.add_argument("--value-target-mode", choices=["legacy_discounted_return", "tanh_step_reward_discounted_average"], default="tanh_step_reward_discounted_average")
    parser.add_argument("--value-reward-scale", default="auto")
    parser.add_argument("--value-reward-scale-quantile", type=float, default=0.95)
    parser.set_defaults(handler=_teacher)


def _teacher(args: argparse.Namespace) -> int:
    from grid_topology_ai.teacher_runtime import main as teacher_main

    argv = [args.raw_dir, "--transitions", args.transitions, "--output-dir", args.output]
    values = {"depth": args.depth, "beam-width": args.beam_width, "candidate-pool": args.candidate_pool, "top-k": args.top_k, "gamma": args.gamma, "pf-alg": args.pf_alg, "pf-max-iter": args.pf_max_iter, "max-steps": args.max_steps, "max-teacher-steps": args.max_teacher_steps, "soft-policy-temperature": args.soft_policy_temperature, "power-flow-failure-penalty": args.power_flow_failure_penalty, "min-continue-improvement-with-hard": args.min_continue_improvement_with_hard, "min-continue-improvement-without-hard": args.min_continue_improvement_without_hard, "max-loading-increase-limit": args.max_loading_increase_limit, "num-workers": args.workers, "batch-size": args.batch_size, "lodf-screen-top-k": args.lodf_screen_top_k, "lodf-min-candidate-count": args.lodf_min_candidate_count, "value-target-mode": args.value_target_mode, "value-reward-scale": args.value_reward_scale, "value-reward-scale-quantile": args.value_reward_scale_quantile}
    if args.limit is not None:
        values["limit"] = args.limit
    for name, value in values.items():
        argv.extend([f"--{name}", str(value)])
    for option, enabled in (("use-soft-root-policy", args.use_soft_root_policy), ("allow-hard-count-increase", args.allow_hard_count_increase), ("disable-cache", args.disable_cache), ("add-handoff-example", args.add_handoff_example), ("quiet-success", args.quiet), ("use-lodf-screening", args.use_lodf_screening)):
        if enabled:
            argv.append(f"--{option}")
    return int(teacher_main(argv) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="power-grid-reconfig", description="Light power-grid topology control workflows.")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_teacher(commands.add_parser("teacher", help="Generate deterministic teacher examples."))
    _add_train(commands.add_parser("train", help="Train the policy-value model."))
    _add_self_play(commands.add_parser("self-play", help="Generate MCTS self-play examples."))
    _add_evaluate(commands.add_parser("evaluate", help="Evaluate one checkpoint policy."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
