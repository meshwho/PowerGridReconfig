from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# This is the behavior oracle for the Light migration. Keep the list focused on
# scientific/runtime behavior that must survive the structural simplification;
# do not add tests whose only purpose is preserving legacy module layout,
# provenance schemas, reporting formats or persistent-L2 implementation details.
ORACLE_TESTS: dict[str, tuple[str, ...]] = {
    "power_flow": (
        "tests/test_ac_numerical_fixtures.py",
        "tests/test_pypower_q_limit_compat.py",
        "tests/test_q_limit_solver_reuse_contract.py",
        "tests/test_pypower_network_workspace.py",
        "tests/test_pypower_newton_workspace.py",
        "tests/test_power_flow_cache_boundaries.py",
        "tests/test_exact_power_flow_cache.py",
        "tests/test_power_flow_contracts.py",
    ),
    "actions_and_state": (
        "tests/test_action_masking.py",
        "tests/test_action_space_mask_layers.py",
        "tests/test_physical_edge_activity_propagation.py",
        "tests/test_multistep_generator_state.py",
    ),
    "physics_and_targets": (
        "tests/test_physical_constraints.py",
        "tests/test_physical_objective_boundaries.py",
        "tests/test_final_topology_quality_targets.py",
        "tests/test_reward_logic.py",
        "tests/test_unified_value_return_contract.py",
    ),
    "teacher": (
        "tests/test_teacher_trajectory_replay_contract.py",
        "tests/test_teacher_checkpoint_resume.py",
        "tests/test_numpy_runtime_scenario.py",
        "tests/test_teacher_config_runtime.py",
        "tests/test_teacher_staged_worker_start.py",
    ),
    "search": (
        "tests/test_mcts_action_ranking.py",
        "tests/test_mcts_value_contract.py",
        "tests/test_mcts_off_prior_exploration.py",
        "tests/test_physical_stop_policy.py",
    ),
    "model_and_dataset": (
        "tests/test_graph_policy_value_net_v2.py",
        "tests/test_variable_graph_batches.py",
        "tests/test_neural_evaluator_normalization.py",
    ),
    "training": (
        "tests/self_play/test_training_api.py",
        "tests/self_play/test_training_cli.py",
    ),
    "self_play": (
        "tests/self_play/test_generation_api.py",
        "tests/self_play/test_generation_policy_target.py",
        "tests/self_play/test_no_legal_action.py",
        "tests/self_play/test_stages.py",
    ),
    "evaluation": (
        "tests/evaluation/test_canonical_topology_quality.py",
        "tests/evaluation/test_checkpoint_policy_modes.py",
    ),
}

REQUIRED_DOMAINS = frozenset(
    {
        "power_flow",
        "actions_and_state",
        "physics_and_targets",
        "teacher",
        "search",
        "model_and_dataset",
        "training",
        "self_play",
        "evaluation",
    }
)

TEACHER_MODULE = "scripts.self_play.generate_impact_teacher_redispatch_runtime"

# Fixed workload for Light timing. L1 remains enabled because --disable-cache
# is not used; removed cache layers need no compatibility configuration.
TEACHER_BENCHMARK_ARGS = (
    "--depth",
    "2",
    "--beam-width",
    "4",
    "--candidate-pool",
    "20",
    "--top-k",
    "10",
    "--gamma",
    "1",
    "--pf-alg",
    "1",
    "--pf-max-iter",
    "30",
    "--max-steps",
    "3",
    "--max-teacher-steps",
    "3",
    "--clear-caches-every",
    "1",
    "--batch-size",
    "1",
    "--max-tasks-per-child",
    "0",
    "--value-target-mode",
    "tanh_step_reward_discounted_average",
    "--value-reward-scale",
    "500",
    "--add-handoff-example",
    "--use-lodf-screening",
    "--lodf-screen-top-k",
    "10",
    "--lodf-min-candidate-count",
    "1",
    "--quiet-success",
)


def _flatten_behavior_tests() -> tuple[str, ...]:
    missing = REQUIRED_DOMAINS - set(ORACLE_TESTS)
    if missing:
        raise RuntimeError(
            "Light oracle is missing required behavior domains: "
            + ", ".join(sorted(missing))
        )

    tests: list[str] = []
    seen: set[str] = set()
    for domain in ORACLE_TESTS.values():
        for test_path in domain:
            if test_path not in seen:
                tests.append(test_path)
                seen.add(test_path)
    return tuple(tests)


def _run_behavior_oracle(extra_pytest_args: list[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *_flatten_behavior_tests(),
        *extra_pytest_args,
    ]
    print("Behavior oracle:")
    print(" ".join(command))
    return subprocess.call(command, cwd=ROOT)


def _benchmark_environment() -> dict[str, str]:
    env = os.environ.copy()

    # Keep native numerical libraries deterministic and avoid nested
    # oversubscription while the teacher itself controls process parallelism.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"

    return env


def _teacher_benchmark_command(
    *,
    raw_dir: Path,
    transitions: Path,
    checkpoint: Path,
    workers: int,
    limit: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        TEACHER_MODULE,
        str(raw_dir),
        "--transitions",
        str(transitions),
        "--checkpoint",
        str(checkpoint),
        *TEACHER_BENCHMARK_ARGS,
        "--num-workers",
        str(workers),
        "--limit",
        str(limit),
    ]


def _run_teacher_once(
    *,
    raw_dir: Path,
    transitions: Path,
    workers: int,
    limit: int,
    env: dict[str, str],
) -> float:
    with tempfile.TemporaryDirectory(prefix="light-oracle-") as tmp:
        output_root = Path(tmp)
        checkpoint = output_root / "teacher_checkpoint.jsonl"
        command = _teacher_benchmark_command(
            raw_dir=raw_dir,
            transitions=transitions,
            checkpoint=checkpoint,
            workers=workers,
            limit=limit,
        )

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                "Teacher performance oracle failed:\n"
                + completed.stderr
            )
        return elapsed


def _run_teacher_benchmark(
    *,
    raw_dir: Path,
    transitions: Path,
    workers: int,
    limit: int,
    repeat: int,
    output: Path | None,
) -> int:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    raw_dir = raw_dir.resolve()
    transitions = transitions.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    if not transitions.is_file():
        raise FileNotFoundError(transitions)

    env = _benchmark_environment()
    timings = [
        _run_teacher_once(
            raw_dir=raw_dir,
            transitions=transitions,
            workers=workers,
            limit=limit,
            env=env,
        )
        for _ in range(repeat)
    ]

    payload = {
        "raw_dir": str(raw_dir),
        "transitions": str(transitions),
        "workers": workers,
        "limit": limit,
        "repeat": repeat,
        "l1_cache": True,
        "runtime": "numpy-mmap",
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "timings_seconds": timings,
    }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior and performance oracle for the Light migration."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    behavior = subparsers.add_parser(
        "behavior",
        help="Run the deterministic behavior oracle.",
    )
    behavior.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to pytest.",
    )

    performance = subparsers.add_parser(
        "performance",
        help="Benchmark the production teacher on a fixed workload.",
    )
    performance.add_argument("--raw-dir", type=Path, required=True)
    performance.add_argument("--transitions", type=Path, required=True)
    performance.add_argument("--workers", type=int, default=1)
    performance.add_argument("--limit", type=int, default=3)
    performance.add_argument("--repeat", type=int, default=3)
    performance.add_argument("--output", type=Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "behavior":
        return _run_behavior_oracle(list(args.pytest_args))
    if args.mode == "performance":
        return _run_teacher_benchmark(
            raw_dir=args.raw_dir,
            transitions=args.transitions,
            workers=args.workers,
            limit=args.limit,
            repeat=args.repeat,
            output=args.output,
        )
    raise RuntimeError(f"Unsupported oracle mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
