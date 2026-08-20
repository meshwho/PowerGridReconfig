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
    "1.0",
    "--pf-alg",
    "1",
    "--pf-max-iter",
    "30",
    "--max-steps",
    "3",
    "--max-teacher-steps",
    "3",
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


def _oracle_paths() -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for group in ORACLE_TESTS.values():
        for path in group:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def test_light_oracle_covers_required_domains() -> None:
    assert set(ORACLE_TESTS) == REQUIRED_DOMAINS
    assert all(ORACLE_TESTS[name] for name in REQUIRED_DOMAINS)


def test_light_oracle_paths_exist() -> None:
    missing = [path for path in _oracle_paths() if not (ROOT / path).is_file()]
    assert missing == []


def _run_behavior_oracle(extra_pytest_args: list[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *_oracle_paths(),
        *extra_pytest_args,
    ]
    print("Behavior oracle:")
    print(" ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


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

    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def _run_teacher_benchmark(
    *,
    raw_dir: Path,
    transitions: Path,
    workers: int,
    limit: int,
    repeat: int,
    report: Path | None,
) -> int:
    raw_dir = raw_dir.resolve()
    transitions = transitions.resolve()

    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")
    if not transitions.is_file():
        raise SystemExit(f"transitions file does not exist: {transitions}")
    if workers < 1:
        raise SystemExit("--workers must be >= 1")
    if limit < 1:
        raise SystemExit("--limit must be >= 1")
    if repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    env = _benchmark_environment()
    timings: list[float] = []
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="pgr_light_oracle_") as tmp:
        tmp_root = Path(tmp)

        for index in range(repeat):
            output_dir = tmp_root / f"teacher_{index:02d}"
            command = [
                sys.executable,
                "-u",
                "-m",
                TEACHER_MODULE,
                str(raw_dir),
                "--transitions",
                str(transitions),
                "--output-dir",
                str(output_dir),
                *TEACHER_BENCHMARK_ARGS,
                "--num-workers",
                str(workers),
                "--limit",
                str(limit),
            ]

            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.perf_counter() - started

            print(completed.stdout, end="")
            if completed.returncode != 0:
                print(f"teacher benchmark failed with exit code {completed.returncode}")
                return completed.returncode

            examples = output_dir / "examples.csv"
            checkpoint = output_dir / "teacher_checkpoint.jsonl"
            states = output_dir / "states"
            if not examples.is_file() or not checkpoint.is_file() or not states.is_dir():
                raise RuntimeError("teacher benchmark completed without canonical output artifacts")

            timings.append(elapsed)
            records.append(
                {
                    "run": index,
                    "elapsed_seconds": elapsed,
                    "examples_bytes": examples.stat().st_size,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "state_files": len(list(states.glob("*.npz"))),
                }
            )
            print(f"benchmark run {index + 1}/{repeat}: {elapsed:.3f}s")

    summary = {
        "oracle": "light-teacher-performance-v1",
        "python": sys.version.split()[0],
        "workers": workers,
        "limit": limit,
        "repeat": repeat,
        "l1_cache": True,
        "runtime": "numpy-mmap",
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "runs": records,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if report is not None:
        report = report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote benchmark report: {report}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical behavior/performance oracle used while reducing "
            "PowerGridReconfig to the Light implementation."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    behavior = subparsers.add_parser("behavior", help="run the focused pytest oracle")
    behavior.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to pytest after the oracle paths",
    )

    subparsers.add_parser("list", help="print the tests in the behavior oracle")

    benchmark = subparsers.add_parser(
        "teacher-benchmark",
        help="time a deterministic teacher workload with L1 enabled and L2 disabled",
    )
    benchmark.add_argument("raw_dir", type=Path)
    benchmark.add_argument("transitions", type=Path)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument("--limit", type=int, default=3)
    benchmark.add_argument("--repeat", type=int, default=3)
    benchmark.add_argument("--report", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "behavior"):
        extra = [] if args.command is None else list(args.pytest_args)
        return _run_behavior_oracle(extra)

    if args.command == "list":
        for domain, paths in ORACLE_TESTS.items():
            print(f"[{domain}]")
            for path in paths:
                print(path)
        return 0

    if args.command == "teacher-benchmark":
        return _run_teacher_benchmark(
            raw_dir=args.raw_dir,
            transitions=args.transitions,
            workers=args.workers,
            limit=args.limit,
            repeat=args.repeat,
            report=args.report,
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
