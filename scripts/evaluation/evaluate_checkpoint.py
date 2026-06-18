from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import json
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.data_adapter import GridFMAdapter
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.reward import GridFMReward
from grid_topology_ai.search.continuation_gate import (
    analyze_root_branches,
    make_do_nothing_action,
)
from grid_topology_ai.search.mcts import MCTSConfig, MCTSPlanner


# ======================================================================================
# Worker-global context
# ======================================================================================

_WORKER_CONTEXT: dict[str, Any] | None = None


def _require_worker_context() -> dict[str, Any]:
    global _WORKER_CONTEXT

    if _WORKER_CONTEXT is None:
        raise RuntimeError(
            "Worker context is not initialized. "
            "This should not happen when ProcessPoolExecutor initializer is used."
        )

    return _WORKER_CONTEXT


def init_worker_context(
    raw_dir_str: str,
    checkpoint_path_str: str,
    task_config: dict[str, Any],
) -> None:
    """
    Initialize heavy objects once per worker process.

    This is the main speed-up:
    - GridFMAdapter is loaded once per worker;
    - PYPOWER backend is created once per worker;
    - action space cache is local to the worker;
    - neural evaluator is loaded once per worker;
    - MCTS planner is created once per worker.
    """

    global _WORKER_CONTEXT

    raw_dir = Path(raw_dir_str)
    checkpoint_path = Path(checkpoint_path_str)

    adapter = GridFMAdapter(raw_dir)

    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        pf_alg=int(task_config["pf_alg"]),
        enable_cache=not bool(task_config["disable_cache"]),
    )

    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=not bool(task_config["disable_cache"]),
    )

    reward_fn = GridFMReward()

    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=checkpoint_path,
        device=str(task_config["device"]),
        enable_cache=not bool(task_config["disable_cache"]),
    )

    mcts_config = MCTSConfig(
        num_simulations=int(task_config["simulations"]),
        max_depth=int(task_config["depth"]),
        top_k_actions=int(task_config["top_k"]),
        gamma=float(task_config["gamma"]),
        c_puct=float(task_config["c_puct"]),
        leaf_penalty_weight=float(task_config["leaf_penalty_weight"]),
        include_stop_action=True,
        prior_exponent=float(task_config["prior_exponent"]),
        stop_policy=str(task_config["stop_policy"]),

        # Evaluation must be deterministic.
        use_root_dirichlet_noise=False,
        use_dc_screening=bool(task_config["use_dc_screening"]),
        dc_top_k_actions=int(task_config["dc_top_k"]),
        dc_candidate_pool=int(task_config["dc_candidate_pool"]),
        dc_keep_policy_actions=int(task_config["dc_keep_policy_actions"]),
        dc_keep_loading_actions=int(task_config["dc_keep_loading_actions"]),
        dc_policy_weight=float(task_config["dc_policy_weight"]),
        dc_failure_penalty=float(task_config["dc_failure_penalty"]),
        dc_max_depth=int(task_config["dc_max_depth"]),
    )

    planner = MCTSPlanner(
        config=mcts_config,
        evaluator=evaluator,
    )

    _WORKER_CONTEXT = {
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": reward_fn,
        "evaluator": evaluator,
        "planner": planner,
        "task_config": task_config,
        "processed_in_worker": 0,
    }


def clear_worker_caches_if_needed() -> None:
    """
    Prevent long-running workers from accumulating too much cache memory.
    """

    ctx = _require_worker_context()
    task = ctx["task_config"]

    every = int(task["clear_caches_every"])

    if every <= 0:
        return

    ctx["processed_in_worker"] = int(ctx.get("processed_in_worker", 0)) + 1

    if ctx["processed_in_worker"] % every != 0:
        return

    backend = ctx["backend"]
    action_space = ctx["action_space"]
    evaluator = ctx["evaluator"]

    if hasattr(backend, "clear_cache"):
        backend.clear_cache()

    if hasattr(action_space, "clear_cache"):
        action_space.clear_cache()

    if hasattr(evaluator, "clear_cache"):
        evaluator.clear_cache()


# ======================================================================================
# Scoring and episode execution
# ======================================================================================


def compute_safety_score(row: dict) -> float:
    """
    Operational checkpoint score.

    Higher is better.

    Priority:
        1. solved is best;
        2. handoff_to_redispatch is acceptable;
        3. max_steps_reached is bad;
        4. remaining hard overload is heavily penalized;
        5. remaining overload and high final loading are also penalized.
    """

    score = 0.0

    reason = row.get("termination_reason")
    solved = bool(row.get("solved", False))

    final_loading = float(row.get("final_max_loading_percent", 999.0))
    overloaded = int(row.get("final_num_overloaded_branches", 99))
    hard = int(row.get("final_num_hard_overloaded_branches", 99))
    discounted_return = float(row.get("discounted_return", 0.0))

    if solved:
        score += 1000.0
    elif reason in {
        "handoff_to_redispatch",
        "handoff_to_redispatch_with_hard_overload",
    }:
        score += 500.0
    elif reason == "max_steps_reached":
        score -= 300.0
    elif reason == "power_flow_failed":
        score -= 1000.0
    else:
        score -= 100.0

    score -= 300.0 * hard
    score -= 50.0 * overloaded

    if final_loading > 100.0:
        score -= 5.0 * (final_loading - 100.0)

    # Small contribution from reward, but do not let reward dominate safety.
    score += 0.05 * discounted_return

    return float(score)


def run_episode(
    scenario_id: int,
    adapter: GridFMAdapter,
    backend: GridFMPowerFlowBackend,
    action_space: GridFMActionSpace,
    reward_fn: GridFMReward,
    planner: MCTSPlanner,
    max_steps: int,
    gamma: float,
    use_continuation_gate: bool,
    min_hard_improvement: float,
    min_soft_improvement: float,
    min_gate_visits: int,
    min_gate_visit_fraction: float,
    allow_handoff_with_hard_overloads: bool = False,
) -> dict:
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=max_steps,
        allow_handoff_with_hard_overloads=allow_handoff_with_hard_overloads,
    )

    env.reset(scenario_id)

    total_reward = 0.0
    discounted_return = 0.0
    discount = 1.0

    actions: list[int] = []
    branches: list[int | None] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if env.done:
            break

        result = planner.search_from_env(env)

        if result.best_action_id is None:
            break

        raw_action_id = int(result.best_action_id)
        raw_branch_id = result.best_branch_id

        if use_continuation_gate:
            gate_decision = analyze_root_branches(
                result=result,
                min_hard_improvement=min_hard_improvement,
                min_soft_improvement=min_soft_improvement,
                min_visits=min_gate_visits,
                min_visit_fraction=min_gate_visit_fraction,
            )

            action_id = int(gate_decision.selected_action_id)
            branch_id = gate_decision.selected_branch_id
        else:
            action_id = raw_action_id
            branch_id = raw_branch_id

        if action_id == 0:
            action_to_execute = make_do_nothing_action()
        else:
            action_to_execute = result.root.actions_by_id.get(action_id)

            if action_to_execute is None:
                action_to_execute = env.action_by_id(action_id)

        step_result = env.step(action_to_execute)

        reward = float(step_result.reward)

        actions.append(action_id)
        branches.append(branch_id)
        rewards.append(reward)

        total_reward += reward
        discounted_return += discount * reward
        discount *= gamma

        if step_result.done:
            break

    final_state = env.current_state

    if final_state is None:
        final_max_loading = float("nan")
        final_overloaded = -1
        final_hard = -1
        final_outaged = -1
    else:
        final_max_loading = float(final_state.metrics["max_loading_percent"])
        final_overloaded = int(final_state.metrics["num_overloaded_branches"])
        final_hard = int(final_state.metrics["num_hard_overloaded_branches"])
        final_outaged = int(final_state.metrics["num_outaged_branches"])

    row = {
        "scenario_id": int(scenario_id),
        "steps": len(actions),
        "use_continuation_gate": bool(use_continuation_gate),
        "actions": str(actions),
        "branches": str(branches),
        "rewards": str([round(x, 4) for x in rewards]),
        "total_reward": float(total_reward),
        "discounted_return": float(discounted_return),
        "done": bool(env.done),
        "solved": bool(env.solved),
        "termination_reason": env.termination_reason,
        "final_max_loading_percent": final_max_loading,
        "final_num_overloaded_branches": final_overloaded,
        "final_num_hard_overloaded_branches": final_hard,
        "final_num_outaged_branches": final_outaged,
    }

    row["safety_score"] = compute_safety_score(row)

    return row


def run_episode_from_worker_context(scenario_id: int) -> dict[str, Any]:
    """
    Run one scenario using objects initialized in the worker process.
    """

    ctx = _require_worker_context()

    task = ctx["task_config"]

    try:
        row = run_episode(
            scenario_id=int(scenario_id),
            adapter=ctx["adapter"],
            backend=ctx["backend"],
            action_space=ctx["action_space"],
            reward_fn=ctx["reward_fn"],
            planner=ctx["planner"],
            max_steps=int(task["max_steps"]),
            gamma=float(task["gamma"]),
            use_continuation_gate=bool(task["use_continuation_gate"]),
            min_hard_improvement=float(task["min_hard_improvement"]),
            min_soft_improvement=float(task["min_soft_improvement"]),
            min_gate_visits=int(task["min_gate_visits"]),
            min_gate_visit_fraction=float(task["min_gate_visit_fraction"]),
            allow_handoff_with_hard_overloads=bool(
                task["allow_handoff_with_hard_overloads"]
            ),
        )

        clear_worker_caches_if_needed()

        return {
            "ok": True,
            "scenario_id": int(scenario_id),
            "row": row,
            "traceback": None,
        }

    except Exception:
        clear_worker_caches_if_needed()

        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "row": None,
            "traceback": traceback.format_exc(),
        }


def run_scenario_batch(scenario_ids: list[int]) -> list[dict[str, Any]]:
    """
    Run a batch of scenarios inside one worker.

    Batching reduces ProcessPool overhead.
    """

    results: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        results.append(run_episode_from_worker_context(int(scenario_id)))

    return results


# ======================================================================================
# CLI helpers
# ======================================================================================

def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def attach_difficulty_metadata(
    df: pd.DataFrame,
    transitions_path: Path,
) -> pd.DataFrame:
    """
    Attach difficulty_class from transitions CSV when available.

    This keeps evaluation JSON useful for curriculum and fixed eval splits.
    If difficulty_class is absent, the evaluation still works normally.
    """

    transitions = pd.read_csv(transitions_path)

    if "difficulty_class" not in transitions.columns:
        return df

    if "scenario_id" not in transitions.columns:
        return df

    difficulty = (
        transitions[["scenario_id", "difficulty_class"]]
        .drop_duplicates(subset=["scenario_id"])
        .copy()
    )

    difficulty["scenario_id"] = difficulty["scenario_id"].astype(int)

    result = df.merge(
        difficulty,
        on="scenario_id",
        how="left",
    )

    return result

def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    transitions = pd.read_csv(transitions_path)

    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: {transitions_path}"
        )

    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]

    return scenario_ids


def chunk_list(
    values: list[int],
    batch_size: int,
) -> list[list[int]]:
    batch_size = max(int(batch_size), 1)

    return [
        values[i : i + batch_size]
        for i in range(0, len(values), batch_size)
    ]


def make_task_config(args: argparse.Namespace) -> dict[str, Any]:
    reward_config = GridFMReward().config_dict()
    return {
        "simulations": int(args.simulations),
        "depth": int(args.depth),
        "max_steps": int(args.max_steps),
        "top_k": int(args.top_k),
        "gamma": float(args.gamma),
        "c_puct": float(args.c_puct),
        "prior_exponent": float(args.prior_exponent),
        "leaf_penalty_weight": float(args.leaf_penalty_weight),
        "stop_policy": str(args.stop_policy),
        "device": str(args.device),
        "pf_alg": int(args.pf_alg),
        "disable_cache": bool(args.disable_cache),
        "use_continuation_gate": bool(args.use_continuation_gate),
        "min_hard_improvement": float(args.min_hard_improvement),
        "min_soft_improvement": float(args.min_soft_improvement),
        "min_gate_visits": int(args.min_gate_visits),
        "min_gate_visit_fraction": float(args.min_gate_visit_fraction),
        "allow_handoff_with_hard_overloads": bool(args.allow_handoff_with_hard_overloads),
        "clear_caches_every": int(args.clear_caches_every),
        "use_dc_screening": bool(args.use_dc_screening),
        "dc_top_k": int(args.dc_top_k),
        "dc_candidate_pool": int(args.dc_candidate_pool),
        "dc_keep_policy_actions": int(args.dc_keep_policy_actions),
        "dc_keep_loading_actions": int(args.dc_keep_loading_actions),
        "dc_policy_weight": float(args.dc_policy_weight),
        "dc_failure_penalty": float(args.dc_failure_penalty),
        "dc_max_depth": int(args.dc_max_depth),
        "reward_config": reward_config,
    }


def print_row(row: dict[str, Any]) -> None:
    print(
        f"Scenario {int(row['scenario_id']):>5} | "
        f"reason={row['termination_reason']} | "
        f"solved={row['solved']} | "
        f"steps={row['steps']} | "
        f"branches={row['branches']} | "
        f"final_loading={float(row['final_max_loading_percent']):.2f}% | "
        f"overloaded={row['final_num_overloaded_branches']} | "
        f"hard={row['final_num_hard_overloaded_branches']} | "
        f"R={float(row['discounted_return']):.2f} | "
        f"score={float(row['safety_score']):.2f}"
    )

def _safe_mean(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None

    value = series.mean()

    if pd.isna(value):
        return None

    return float(value)


def build_evaluation_metrics(
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
    requested_scenarios: int,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build machine-readable evaluation metrics for self-play loop.

    The self-play loop should read this JSON instead of parsing stdout.
    """

    solved = df["solved"].astype(bool)

    termination_counts = {
        str(key): int(value)
        for key, value in df["termination_reason"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }

    metrics: dict[str, Any] = {
        "requested_scenarios": int(requested_scenarios),
        "evaluated_scenarios": int(len(df)),
        "failed_scenarios": int(len(failed_results)),
        "solve_count": int(solved.sum()),
        "solve_rate": float(solved.mean()) if len(df) > 0 else 0.0,
        "avg_steps": _safe_mean(df["steps"]),
        "avg_steps_to_solve": _safe_mean(df.loc[solved, "steps"]),
        "avg_discounted_return": _safe_mean(df["discounted_return"]),
        "avg_final_loading_percent": _safe_mean(df["final_max_loading_percent"]),
        "avg_final_num_overloaded_branches": _safe_mean(
            df["final_num_overloaded_branches"]
        ),
        "avg_final_num_hard_overloaded_branches": _safe_mean(
            df["final_num_hard_overloaded_branches"]
        ),
        "avg_safety_score": _safe_mean(df["safety_score"]),
        "total_safety_score": float(df["safety_score"].sum()),
        "termination_reason_counts": termination_counts,
        "task_config": dict(task_config),
    }

    if "difficulty_class" in df.columns:
        difficulty_metrics: dict[str, Any] = {}

        for difficulty in ["simple", "medium", "hard"]:
            subset = df[df["difficulty_class"] == difficulty]
            subset_solved = subset["solved"].astype(bool)

            if len(subset) == 0:
                solve_rate = None
                avg_steps_to_solve = None
            else:
                solve_rate = float(subset_solved.mean())
                avg_steps_to_solve = _safe_mean(
                    subset.loc[subset_solved, "steps"]
                )

            metrics[f"count_{difficulty}"] = int(len(subset))
            metrics[f"solve_rate_{difficulty}"] = solve_rate
            metrics[f"avg_steps_to_solve_{difficulty}"] = avg_steps_to_solve

            difficulty_metrics[difficulty] = {
                "count": int(len(subset)),
                "solve_count": int(subset_solved.sum()) if len(subset) else 0,
                "solve_rate": solve_rate,
                "avg_steps": _safe_mean(subset["steps"]) if len(subset) else None,
                "avg_steps_to_solve": avg_steps_to_solve,
                "avg_safety_score": (
                    _safe_mean(subset["safety_score"]) if len(subset) else None
                ),
            }

        metrics["difficulty_metrics"] = difficulty_metrics

    return metrics

def print_summary(
    df: pd.DataFrame,
    failed_results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print(f"\nEvaluated scenarios: {len(df)}")
    print(f"Failed scenarios:    {len(failed_results)}")

    if failed_results:
        print("\nFailures:")
        for item in failed_results[:20]:
            print(f"  Scenario {item['scenario_id']}: failed")
        if len(failed_results) > 20:
            print(f"  ... {len(failed_results) - 20} more failures")

    print("\nTermination reasons:")
    print(df["termination_reason"].value_counts(dropna=False).to_string())

    print("\nSolved:")
    print(df["solved"].value_counts(dropna=False).to_string())

    print("\nAverage metrics:")
    print(f"  Avg discounted return: {df['discounted_return'].mean():.4f}")
    print(f"  Avg final loading:     {df['final_max_loading_percent'].mean():.4f}%")
    print(f"  Avg overloaded:        {df['final_num_overloaded_branches'].mean():.4f}")
    print(f"  Avg hard overloaded:   {df['final_num_hard_overloaded_branches'].mean():.4f}")
    print(f"  Avg safety score:     {df['safety_score'].mean():.4f}")
    print(f"  Total safety score:   {df['safety_score'].sum():.4f}")


def run_sequential(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Sequential execution with the same worker-context logic.

    This keeps --num-workers 1 behavior close to the old script.
    """

    init_worker_context(
        raw_dir_str=str(raw_dir),
        checkpoint_path_str=str(checkpoint_path),
        task_config=task_config,
    )

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    iterator = scenario_batches

    if tqdm is not None:
        iterator = tqdm(
            scenario_batches,
            desc="Evaluating batches",
            unit="batch",
            dynamic_ncols=True,
        )

    for batch in iterator:
        batch_results = run_scenario_batch(batch)

        for result in batch_results:
            if result["ok"]:
                row = result["row"]
                rows.append(row)

                if not quiet:
                    print_row(row)
            else:
                failed.append(result)
                print(f"Scenario {result['scenario_id']}: failed")
                print(result["traceback"])

    return rows, failed


def run_parallel(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    num_workers: int,
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parallel evaluation.

    Each worker process loads its own GridFMAdapter, backend, action space,
    neural evaluator and MCTS planner once.
    """

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        initializer=init_worker_context,
        initargs=(str(raw_dir), str(checkpoint_path), task_config),
    ) as executor:
        futures = [
            executor.submit(run_scenario_batch, batch)
            for batch in scenario_batches
        ]

        iterator = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Evaluating batches",
                unit="batch",
                dynamic_ncols=True,
            )

        for future in iterator:
            batch_results = future.result()

            for result in batch_results:
                if result["ok"]:
                    row = result["row"]
                    rows.append(row)

                    if not quiet:
                        print_row(row)
                else:
                    failed.append(result)
                    print(f"Scenario {result['scenario_id']}: failed")
                    print(result["traceback"])

    return rows, failed


# ======================================================================================
# Main
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a neural policy-value checkpoint with deterministic MCTS."
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
        help="Use lookahead continuation gate for evaluation action selection.",
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
        "--leaf-penalty-weight",
        type=float,
        default=0.10,
        help="Weight of safety/penalty term inside MCTS leaf evaluation.",
    )

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
        default=1,
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
        help=(
            "Treat action 0 as redispatch handoff even when hard overloads remain."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. "
            "Use 1 for old sequential behavior. "
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
            "0 means root only, 1 means root and depth-1 nodes, "
            "-1 means all depths."
        ),
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)
    checkpoint_path = Path(args.checkpoint)

    scenario_ids = load_scenario_ids(
        transitions_path=transitions_path,
        limit=args.limit,
    )

    scenario_batches = chunk_list(
        values=scenario_ids,
        batch_size=int(args.batch_size),
    )

    task_config = make_task_config(args)

    print("=" * 100)
    print("Evaluating checkpoint")
    print("=" * 100)

    print(f"Raw directory:       {raw_dir.resolve()}")
    print(f"Transitions:         {transitions_path.resolve()}")
    print(f"Checkpoint:          {checkpoint_path.resolve()}")
    print(f"Use continuation gate: {args.use_continuation_gate}")
    print(f"Allow hard handoff:  {args.allow_handoff_with_hard_overloads}")
    print(f"Scenarios:           {len(scenario_ids)}")
    print(f"Batches:             {len(scenario_batches)}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Num workers:         {args.num_workers}")
    print(f"Device:              {args.device}")
    print(f"Quiet:               {args.quiet}")
    print(f"Use DC screening:   {args.use_dc_screening}")


    if args.use_dc_screening:
        print(f"  dc max depth:      {args.dc_max_depth}")
        print(f"  dc top k:          {args.dc_top_k}")
        print(f"  dc candidate pool: {args.dc_candidate_pool}")
        print(f"  dc keep policy:    {args.dc_keep_policy_actions}")
        print(f"  dc keep loading:   {args.dc_keep_loading_actions}")
        print(f"  dc policy weight:  {args.dc_policy_weight}")

    if args.use_continuation_gate:
        print(f"  min hard improvement: {args.min_hard_improvement}")
        print(f"  min soft improvement: {args.min_soft_improvement}")
        print(f"  min gate visits:      {args.min_gate_visits}")
        print(f"  min gate visit frac:  {args.min_gate_visit_fraction}")

    if str(args.device).lower().startswith("cuda") and int(args.num_workers) > 1:
        print(
            "\nWARNING: CUDA + multiple worker processes means each worker loads "
            "its own model copy on GPU. Start with --num-workers 2. "
            "If CUDA memory grows too much, use --num-workers 1 or --device cpu.\n"
        )

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions CSV not found: {transitions_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if int(args.num_workers) <= 1:
        rows, failed_results = run_sequential(
            scenario_batches=scenario_batches,
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
            task_config=task_config,
            quiet=bool(args.quiet),
        )
    else:
        rows, failed_results = run_parallel(
            scenario_batches=scenario_batches,
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
            task_config=task_config,
            num_workers=int(args.num_workers),
            quiet=bool(args.quiet),
        )

    if not rows:
        raise RuntimeError("No scenarios were successfully evaluated.")

        df = pd.DataFrame(rows)
    df = df.sort_values("scenario_id", ascending=True).reset_index(drop=True)

    df = attach_difficulty_metadata(
        df=df,
        transitions_path=transitions_path,
    )

    metrics = build_evaluation_metrics(
        df=df,
        failed_results=failed_results,
        requested_scenarios=len(scenario_ids),
        task_config=task_config,
    )

    if args.output_csv is not None:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"\nSaved evaluation CSV: {output_csv_path}")

    if args.output_json is not None:
        output_json_path = Path(args.output_json)
        save_json(
            path=output_json_path,
            payload=metrics,
        )
        print(f"\nSaved evaluation JSON: {output_json_path}")

    print_summary(
        df=df,
        failed_results=failed_results,
    )

    if int(args.num_workers) <= 1:
        ctx = _require_worker_context()

        print("\nPower flow cache:")
        print(ctx["backend"].cache_info())

        print("\nAction space cache:")
        print(ctx["action_space"].cache_info())

        print("\nNeural evaluator cache:")
        print(ctx["evaluator"].cache_info())
    else:
        print("\nCache info:")
        print(
            "Parallel mode uses separate per-process caches. "
            "Global cache statistics are not aggregated."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()