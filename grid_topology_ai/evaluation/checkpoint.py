from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.evaluation.metrics import (
    attach_difficulty_metadata,
    build_evaluation_metrics,
    compute_safety_score,
    print_row,
    print_summary,
)
from grid_topology_ai.self_play.artifacts import save_json

GridFMActionSpace = None
GridFMAdapter = None
GridFMPowerFlowBackend = None
GridFMReward = None
MCTSConfig = None
MCTSPlanner = None
NeuralPolicyValueEvaluator = None
TopologySwitchingEnv = None
analyze_root_branches = None
make_do_nothing_action = None
_RUNTIME_DEPENDENCIES_LOADED = False

_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    raw_dir: Path
    transitions_csv: Path
    checkpoint: Path
    config: EvaluationConfig
    output_csv: Path | None = None
    output_json: Path | None = None
    limit: int | None = None
    quiet: bool = False
    pf_alg: int = 1
    disable_cache: bool = False
    leaf_penalty_weight: float = 0.10
    stop_policy: str = "no_hard_overloads"
    min_hard_improvement: float = 50.0
    min_soft_improvement: float = 15.0
    min_gate_visits: int = 5
    min_gate_visit_fraction: float = 0.01
    clear_caches_every: int = 100
    use_dc_screening: bool = False
    dc_top_k: int = 30
    dc_candidate_pool: int = 120
    dc_keep_policy_actions: int = 5
    dc_keep_loading_actions: int = 5
    dc_policy_weight: float = 0.0
    dc_failure_penalty: float = 1_000_000_000.0
    dc_max_depth: int = 0

    def __post_init__(self) -> None:
        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("limit must be None or > 0")
        if int(self.pf_alg) not in {1, 2, 3, 4}:
            raise ValueError("pf_alg must be one of 1, 2, 3, or 4")
        if float(self.leaf_penalty_weight) < 0.0:
            raise ValueError("leaf_penalty_weight must be >= 0")
        if self.stop_policy not in {
            "never",
            "solved_only",
            "no_hard_overloads",
            "always",
        }:
            raise ValueError("Unsupported stop_policy")
        if float(self.min_hard_improvement) < 0.0:
            raise ValueError("min_hard_improvement must be >= 0")
        if float(self.min_soft_improvement) < 0.0:
            raise ValueError("min_soft_improvement must be >= 0")
        if int(self.min_gate_visits) < 0:
            raise ValueError("min_gate_visits must be >= 0")
        if not 0.0 <= float(self.min_gate_visit_fraction) <= 1.0:
            raise ValueError("min_gate_visit_fraction must be in [0, 1]")
        if int(self.clear_caches_every) < 0:
            raise ValueError("clear_caches_every must be >= 0")
        if int(self.dc_top_k) <= 0:
            raise ValueError("dc_top_k must be > 0")
        if int(self.dc_keep_policy_actions) < 0:
            raise ValueError("dc_keep_policy_actions must be >= 0")
        if int(self.dc_keep_loading_actions) < 0:
            raise ValueError("dc_keep_loading_actions must be >= 0")
        if float(self.dc_policy_weight) < 0.0:
            raise ValueError("dc_policy_weight must be >= 0")
        if float(self.dc_failure_penalty) < 0.0:
            raise ValueError("dc_failure_penalty must be >= 0")
        if int(self.dc_max_depth) < -1:
            raise ValueError("dc_max_depth must be >= -1")


def _ensure_runtime_dependencies() -> None:
    global GridFMActionSpace
    global GridFMAdapter
    global GridFMPowerFlowBackend
    global GridFMReward
    global MCTSConfig
    global MCTSPlanner
    global NeuralPolicyValueEvaluator
    global TopologySwitchingEnv
    global analyze_root_branches
    global make_do_nothing_action
    global _RUNTIME_DEPENDENCIES_LOADED

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    from grid_topology_ai.action_space import GridFMActionSpace as _ActionSpace
    from grid_topology_ai.data_adapter import GridFMAdapter as _Adapter
    from grid_topology_ai.environment import TopologySwitchingEnv as _Env
    from grid_topology_ai.models.neural_evaluator import (
        NeuralPolicyValueEvaluator as _Evaluator,
    )
    from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend as _Backend
    from grid_topology_ai.reward import GridFMReward as _Reward
    from grid_topology_ai.search.continuation_gate import (
        analyze_root_branches as _analyze_root_branches,
        make_do_nothing_action as _make_do_nothing_action,
    )
    from grid_topology_ai.search.mcts import MCTSConfig as _MCTSConfig
    from grid_topology_ai.search.mcts import MCTSPlanner as _MCTSPlanner

    GridFMActionSpace = _ActionSpace
    GridFMAdapter = _Adapter
    GridFMPowerFlowBackend = _Backend
    GridFMReward = _Reward
    MCTSConfig = _MCTSConfig
    MCTSPlanner = _MCTSPlanner
    NeuralPolicyValueEvaluator = _Evaluator
    TopologySwitchingEnv = _Env
    analyze_root_branches = _analyze_root_branches
    make_do_nothing_action = _make_do_nothing_action
    _RUNTIME_DEPENDENCIES_LOADED = True


def _release_worker_context() -> None:
    global _WORKER_CONTEXT

    context = _WORKER_CONTEXT
    _WORKER_CONTEXT = None

    if context is None:
        return

    for name in ("backend", "action_space", "evaluator"):
        cached_object = context.get(name)
        clear_cache = getattr(cached_object, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

def _require_worker_context() -> dict[str, Any]:
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
    global _WORKER_CONTEXT

    _ensure_runtime_dependencies()
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
    planner = MCTSPlanner(config=mcts_config, evaluator=evaluator)

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


def run_episode(
    scenario_id: int,
    adapter: Any,
    backend: Any,
    action_space: Any,
    reward_fn: Any,
    planner: Any,
    max_steps: int,
    gamma: float,
    use_continuation_gate: bool,
    min_hard_improvement: float,
    min_soft_improvement: float,
    min_gate_visits: int,
    min_gate_visit_fraction: float,
    allow_handoff_with_hard_overloads: bool = False,
) -> dict[str, Any]:
    _ensure_runtime_dependencies()
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
    # Intentional process-worker boundary:
    # serialize one scenario failure with its traceback instead of
    # terminating the entire evaluation pool.
    except Exception:
        clear_worker_caches_if_needed()
        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "row": None,
            "traceback": traceback.format_exc(),
        }


def run_scenario_batch(scenario_ids: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        results.append(run_episode_from_worker_context(int(scenario_id)))
    return results


def load_scenario_ids(transitions_path: Path, limit: int | None) -> list[int]:
    transitions = pd.read_csv(transitions_path)

    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: {transitions_path}"
        )

    scenario_ids = sorted(int(x) for x in transitions["scenario_id"].unique())

    if limit is not None:
        scenario_ids = scenario_ids[: int(limit)]

    return scenario_ids


def chunk_list(values: list[int], batch_size: int) -> list[list[int]]:
    batch_size = max(int(batch_size), 1)
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def _make_task_config(request: EvaluationRequest) -> dict[str, Any]:
    if GridFMReward is None:
        _ensure_runtime_dependencies()

    reward_config = GridFMReward().config_dict()
    config = request.config
    return {
        "simulations": int(config.simulations),
        "depth": int(config.depth),
        "max_steps": int(config.max_steps),
        "top_k": int(config.top_k),
        "gamma": float(config.gamma),
        "c_puct": float(config.c_puct),
        "prior_exponent": float(config.prior_exponent),
        "leaf_penalty_weight": float(request.leaf_penalty_weight),
        "stop_policy": str(request.stop_policy),
        "device": str(config.device),
        "pf_alg": int(request.pf_alg),
        "disable_cache": bool(request.disable_cache),
        "use_continuation_gate": bool(config.use_continuation_gate),
        "min_hard_improvement": float(request.min_hard_improvement),
        "min_soft_improvement": float(request.min_soft_improvement),
        "min_gate_visits": int(request.min_gate_visits),
        "min_gate_visit_fraction": float(request.min_gate_visit_fraction),
        "allow_handoff_with_hard_overloads": bool(
            config.allow_handoff_with_hard_overloads
        ),
        "clear_caches_every": int(request.clear_caches_every),
        "use_dc_screening": bool(request.use_dc_screening),
        "dc_top_k": int(request.dc_top_k),
        "dc_candidate_pool": int(request.dc_candidate_pool),
        "dc_keep_policy_actions": int(request.dc_keep_policy_actions),
        "dc_keep_loading_actions": int(request.dc_keep_loading_actions),
        "dc_policy_weight": float(request.dc_policy_weight),
        "dc_failure_penalty": float(request.dc_failure_penalty),
        "dc_max_depth": int(request.dc_max_depth),
        "reward_config": reward_config,
    }


def run_sequential(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        initializer=init_worker_context,
        initargs=(str(raw_dir), str(checkpoint_path), task_config),
    ) as executor:
        futures = [executor.submit(run_scenario_batch, batch) for batch in scenario_batches]
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


def evaluate_checkpoint(request: EvaluationRequest) -> dict[str, Any]:
    sequential_mode = int(request.config.num_workers) <= 1

    try:
        raw_dir = request.raw_dir
        transitions_path = request.transitions_csv
        checkpoint_path = request.checkpoint

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        if not transitions_path.exists():
            raise FileNotFoundError(f"Transitions CSV not found: {transitions_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        scenario_ids = load_scenario_ids(
            transitions_path=transitions_path,
            limit=request.limit,
        )
        scenario_batches = chunk_list(
            values=scenario_ids,
            batch_size=int(request.config.batch_size),
        )
        task_config = _make_task_config(request)

        print("=" * 100)
        print("Evaluating checkpoint")
        print("=" * 100)
        print(f"Raw directory:       {raw_dir.resolve()}")
        print(f"Transitions:         {transitions_path.resolve()}")
        print(f"Checkpoint:          {checkpoint_path.resolve()}")
        print(f"Use continuation gate: {request.config.use_continuation_gate}")
        print(
            "Allow hard handoff:  "
            f"{request.config.allow_handoff_with_hard_overloads}"
        )
        print(f"Scenarios:           {len(scenario_ids)}")
        print(f"Batches:             {len(scenario_batches)}")
        print(f"Batch size:          {request.config.batch_size}")
        print(f"Num workers:         {request.config.num_workers}")
        print(f"Device:              {request.config.device}")
        print(f"Quiet:               {request.quiet}")
        print(f"Use DC screening:   {request.use_dc_screening}")

        if request.use_dc_screening:
            print(f"  dc max depth:      {request.dc_max_depth}")
            print(f"  dc top k:          {request.dc_top_k}")
            print(f"  dc candidate pool: {request.dc_candidate_pool}")
            print(f"  dc keep policy:    {request.dc_keep_policy_actions}")
            print(f"  dc keep loading:   {request.dc_keep_loading_actions}")
            print(f"  dc policy weight:  {request.dc_policy_weight}")

        if request.config.use_continuation_gate:
            print(f"  min hard improvement: {request.min_hard_improvement}")
            print(f"  min soft improvement: {request.min_soft_improvement}")
            print(f"  min gate visits:      {request.min_gate_visits}")
            print(f"  min gate visit frac:  {request.min_gate_visit_fraction}")

        if (
            str(request.config.device).lower().startswith("cuda")
            and int(request.config.num_workers) > 1
        ):
            print(
                "\nWARNING: CUDA + multiple worker processes means each worker loads "
                "its own model copy on GPU. Start with --num-workers 2. "
                "If CUDA memory grows too much, use --num-workers 1 or --device cpu.\n"
            )

        if sequential_mode:
            rows, failed_results = run_sequential(
                scenario_batches=scenario_batches,
                raw_dir=raw_dir,
                checkpoint_path=checkpoint_path,
                task_config=task_config,
                quiet=bool(request.quiet),
            )
        else:
            rows, failed_results = run_parallel(
                scenario_batches=scenario_batches,
                raw_dir=raw_dir,
                checkpoint_path=checkpoint_path,
                task_config=task_config,
                num_workers=int(request.config.num_workers),
                quiet=bool(request.quiet),
            )

        if not rows:
            raise RuntimeError("No scenarios were successfully evaluated.")

        df = pd.DataFrame(rows)
        df = df.sort_values("scenario_id", ascending=True).reset_index(drop=True)
        df = attach_difficulty_metadata(df=df, transitions_path=transitions_path)
        metrics = build_evaluation_metrics(
            df=df,
            failed_results=failed_results,
            requested_scenarios=len(scenario_ids),
            task_config=task_config,
        )

        if request.output_csv is not None:
            request.output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(request.output_csv, index=False)
            print(f"\nSaved evaluation CSV: {request.output_csv}")

        if request.output_json is not None:
            save_json(payload=metrics, path=request.output_json)
            print(f"\nSaved evaluation JSON: {request.output_json}")

        print_summary(df=df, failed_results=failed_results)

        if sequential_mode:
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
        return metrics
    finally:
        if sequential_mode:
            _release_worker_context()
