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
from grid_topology_ai.config.physics import PhysicsConfig, resolve_physics_config
from grid_topology_ai.contracts import physics_provenance, require_physics_provenance
from grid_topology_ai.evaluation.episode_result import (
    EvaluationEpisodeTrace,
    build_evaluation_episode_row,
)
from grid_topology_ai.evaluation.metrics import (
    attach_difficulty_metadata,
    print_row,
    print_summary,
)
from grid_topology_ai.evaluation.policy_comparison import (
    PolicyMode,
    build_policy_comparison_metrics,
    evaluation_policy_modes,
    print_policy_comparison_summary,
    select_evaluation_root_policy,
)
from grid_topology_ai.physical_objective import STOP_POLICIES
from grid_topology_ai.self_play.artifacts import save_json

GridFMActionSpace = GridFMAdapter = GridFMPowerFlowBackend = None
GridFMReward = MCTSConfig = MCTSPlanner = None
NeuralPolicyValueEvaluator = TopologySwitchingEnv = None
analyze_root_branches = make_do_nothing_action = None
_RUNTIME_DEPENDENCIES_LOADED = False
_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    raw_dir: Path
    transitions_csv: Path
    checkpoint: Path
    config: EvaluationConfig
    physics_config: PhysicsConfig | None = None
    output_csv: Path | None = None
    output_json: Path | None = None
    limit: int | None = None
    quiet: bool = False
    pf_alg: int | None = None
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

    @property
    def resolved_pf_alg(self) -> int:
        return self.resolved_physics_config.pf_alg

    @property
    def resolved_physics_config(self) -> PhysicsConfig:
        legacy = self.config.pf_alg if self.pf_alg is None else self.pf_alg
        return resolve_physics_config(self.physics_config, legacy)

    def __post_init__(self) -> None:
        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("limit must be None or > 0")
        if self.resolved_pf_alg not in {1, 2, 3, 4}:
            raise ValueError("resolved pf_alg must be one of 1, 2, 3, or 4")
        if float(self.leaf_penalty_weight) < 0:
            raise ValueError("leaf_penalty_weight must be >= 0")
        if self.stop_policy not in STOP_POLICIES:
            raise ValueError("Unsupported stop_policy")
        if min(
            float(self.min_hard_improvement),
            float(self.min_soft_improvement),
        ) < 0:
            raise ValueError("continuation improvement thresholds must be >= 0")
        if int(self.min_gate_visits) < 0:
            raise ValueError("min_gate_visits must be >= 0")
        if not 0 <= float(self.min_gate_visit_fraction) <= 1:
            raise ValueError("min_gate_visit_fraction must be in [0, 1]")
        if int(self.clear_caches_every) < 0:
            raise ValueError("clear_caches_every must be >= 0")
        if int(self.dc_top_k) <= 0:
            raise ValueError("dc_top_k must be > 0")
        if min(
            int(self.dc_keep_policy_actions),
            int(self.dc_keep_loading_actions),
        ) < 0:
            raise ValueError("DC keep counts must be >= 0")
        if min(
            float(self.dc_policy_weight),
            float(self.dc_failure_penalty),
        ) < 0:
            raise ValueError("DC weights and penalties must be >= 0")
        if int(self.dc_max_depth) < -1:
            raise ValueError("dc_max_depth must be >= -1")


def _ensure_runtime_dependencies() -> None:
    global GridFMActionSpace, GridFMAdapter, GridFMPowerFlowBackend, GridFMReward
    global MCTSConfig, MCTSPlanner, NeuralPolicyValueEvaluator, TopologySwitchingEnv
    global analyze_root_branches, make_do_nothing_action
    global _RUNTIME_DEPENDENCIES_LOADED

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    from grid_topology_ai.action_space import GridFMActionSpace as ActionSpace
    from grid_topology_ai.data_adapter import GridFMAdapter as Adapter
    from grid_topology_ai.environment import TopologySwitchingEnv as Env
    from grid_topology_ai.models.neural_evaluator import (
        NeuralPolicyValueEvaluator as Evaluator,
    )
    from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend as Backend
    from grid_topology_ai.reward import GridFMReward as Reward
    from grid_topology_ai.search.continuation_gate import (
        analyze_root_branches as analyze,
        make_do_nothing_action as stop_action,
    )
    from grid_topology_ai.search.mcts import (
        MCTSConfig as SearchConfig,
        MCTSPlanner as Planner,
    )

    GridFMActionSpace, GridFMAdapter = ActionSpace, Adapter
    GridFMPowerFlowBackend, GridFMReward = Backend, Reward
    MCTSConfig, MCTSPlanner = SearchConfig, Planner
    NeuralPolicyValueEvaluator, TopologySwitchingEnv = Evaluator, Env
    analyze_root_branches, make_do_nothing_action = analyze, stop_action
    _RUNTIME_DEPENDENCIES_LOADED = True


def _clear_context_caches(context: dict[str, Any] | None) -> None:
    if context is None:
        return
    for name in ("backend", "action_space", "evaluator"):
        clear = getattr(context.get(name), "clear_cache", None)
        if clear is not None:
            clear()


def _release_worker_context() -> None:
    global _WORKER_CONTEXT
    context, _WORKER_CONTEXT = _WORKER_CONTEXT, None
    _clear_context_caches(context)


def _require_worker_context() -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context is not initialized.")
    return _WORKER_CONTEXT


def init_worker_context(
    raw_dir_str: str,
    checkpoint_path_str: str,
    task_config: dict[str, Any],
) -> None:
    global _WORKER_CONTEXT
    _ensure_runtime_dependencies()

    physics = require_physics_provenance(
        task_config,
        source="evaluation task",
    )
    adapter = GridFMAdapter(Path(raw_dir_str), physics_config=physics)
    cache = not bool(task_config["disable_cache"])
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=physics,
        enable_cache=cache,
    )
    action_space = GridFMActionSpace(
        require_connected_after_switch=True,
        enable_cache=cache,
    )
    reward_fn = GridFMReward(physics_config=physics)
    evaluator = NeuralPolicyValueEvaluator(
        checkpoint_path=Path(checkpoint_path_str),
        device=str(task_config["device"]),
        enable_cache=cache,
        physics_config=physics,
    )
    search_config = MCTSConfig(
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
    _WORKER_CONTEXT = {
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": reward_fn,
        "evaluator": evaluator,
        "planner": MCTSPlanner(
            config=search_config,
            evaluator=evaluator,
            physics_config=physics,
        ),
        "physics_config": physics,
        "task_config": task_config,
        "processed_in_worker": 0,
    }


def clear_worker_caches_if_needed() -> None:
    context = _require_worker_context()
    every = int(context["task_config"]["clear_caches_every"])
    if every <= 0:
        return
    context["processed_in_worker"] = int(context["processed_in_worker"]) + 1
    if context["processed_in_worker"] % every == 0:
        _clear_context_caches(context)


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
    physics_config: PhysicsConfig | None = None,
    policy_mode: PolicyMode | str | None = None,
) -> dict[str, Any]:
    _ensure_runtime_dependencies()
    mode = PolicyMode(policy_mode) if policy_mode is not None else (
        PolicyMode.CONSTRAINED
        if use_continuation_gate
        else PolicyMode.UNGATED
    )
    env = TopologySwitchingEnv(
        adapter=adapter,
        backend=backend,
        action_space=action_space,
        reward_fn=reward_fn,
        max_steps=max_steps,
        allow_handoff_with_hard_overloads=(
            allow_handoff_with_hard_overloads
        ),
    )
    env.reset(scenario_id)
    trace = EvaluationEpisodeTrace()
    discount = 1.0

    for _ in range(max_steps):
        if env.done:
            break
        result = planner.search_from_env(env)
        if result.best_action_id is None:
            break

        analysis = None
        if mode is PolicyMode.CONSTRAINED:
            analysis = analyze_root_branches(
                result=result,
                min_hard_improvement=min_hard_improvement,
                min_soft_improvement=min_soft_improvement,
                min_visits=min_gate_visits,
                min_visit_fraction=min_gate_visit_fraction,
                physics_config=physics_config,
            )
        decision = select_evaluation_root_policy(
            search_result=result,
            mode=mode,
            continuation_analysis=analysis,
        )
        trace.raw_policies.append(decision.raw_policy)
        trace.executed_policies.append(decision.policy)
        trace.allowed_action_ids.append(list(decision.allowed_action_ids))
        trace.constraint_changed_policy_steps += int(
            decision.constraint_changed_policy
        )
        if decision.empty_constrained_support:
            trace.empty_constrained_support_count += 1
            break

        assert decision.action_id is not None
        action_id = int(decision.action_id)
        action = (
            make_do_nothing_action()
            if action_id == 0
            else result.root.actions_by_id[action_id]
        )
        step_result = env.step(action)
        reward = float(step_result.reward)
        trace.actions.append(action_id)
        trace.branches.append(decision.branch_id)
        trace.rewards.append(reward)
        trace.total_reward += reward
        trace.discounted_return += discount * reward
        discount *= gamma
        if step_result.done:
            break

    return build_evaluation_episode_row(
        scenario_id=scenario_id,
        policy_mode=mode.value,
        env=env,
        trace=trace,
        physics_config=physics_config,
    )


def run_episode_from_worker_context(
    scenario_id: int,
    policy_mode: PolicyMode | str | None = None,
) -> dict[str, Any]:
    context = _require_worker_context()
    task = context["task_config"]
    mode = PolicyMode(
        policy_mode
        or task.get("primary_policy_mode", PolicyMode.UNGATED.value)
    )
    try:
        row = run_episode(
            scenario_id=int(scenario_id),
            adapter=context["adapter"],
            backend=context["backend"],
            action_space=context["action_space"],
            reward_fn=context["reward_fn"],
            planner=context["planner"],
            max_steps=int(task["max_steps"]),
            gamma=float(task["gamma"]),
            use_continuation_gate=mode is PolicyMode.CONSTRAINED,
            min_hard_improvement=float(task["min_hard_improvement"]),
            min_soft_improvement=float(task["min_soft_improvement"]),
            min_gate_visits=int(task["min_gate_visits"]),
            min_gate_visit_fraction=float(
                task["min_gate_visit_fraction"]
            ),
            allow_handoff_with_hard_overloads=bool(
                task["allow_handoff_with_hard_overloads"]
            ),
            physics_config=context["physics_config"],
            policy_mode=mode,
        )
        return {
            "ok": True,
            "scenario_id": int(scenario_id),
            "policy_mode": mode.value,
            "row": row,
            "traceback": None,
        }
    # Intentional process-worker boundary: serialize one evaluation-mode
    # failure with its traceback without terminating the worker pool.
    except Exception:
        return {
            "ok": False,
            "scenario_id": int(scenario_id),
            "policy_mode": mode.value,
            "row": None,
            "traceback": traceback.format_exc(),
        }


def run_scenario_batch(scenario_ids: list[int]) -> list[dict[str, Any]]:
    task = _require_worker_context()["task_config"]
    modes = tuple(PolicyMode(mode) for mode in task["evaluation_modes"])
    results: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        results.extend(
            run_episode_from_worker_context(int(scenario_id), mode)
            for mode in modes
        )
        clear_worker_caches_if_needed()
    return results


def load_scenario_ids(
    transitions_path: Path,
    limit: int | None,
) -> list[int]:
    transitions = pd.read_csv(transitions_path)
    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions CSV must contain scenario_id column: "
            f"{transitions_path}"
        )
    scenario_ids = sorted(
        int(value) for value in transitions["scenario_id"].unique()
    )
    return scenario_ids if limit is None else scenario_ids[: int(limit)]


def chunk_list(values: list[int], batch_size: int) -> list[list[int]]:
    size = max(int(batch_size), 1)
    return [
        values[index : index + size]
        for index in range(0, len(values), size)
    ]


def _make_task_config(request: EvaluationRequest) -> dict[str, Any]:
    if GridFMReward is None:
        _ensure_runtime_dependencies()
    config = request.config
    modes = evaluation_policy_modes(config.use_continuation_gate)
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
        "pf_alg": request.resolved_physics_config.pf_alg,
        **physics_provenance(request.resolved_physics_config),
        "disable_cache": bool(request.disable_cache),
        "use_continuation_gate": bool(config.use_continuation_gate),
        "evaluation_modes": [mode.value for mode in modes],
        "primary_policy_mode": modes[-1].value,
        "min_hard_improvement": float(request.min_hard_improvement),
        "min_soft_improvement": float(request.min_soft_improvement),
        "min_gate_visits": int(request.min_gate_visits),
        "min_gate_visit_fraction": float(
            request.min_gate_visit_fraction
        ),
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
        "reward_config": GridFMReward(
            physics_config=request.resolved_physics_config
        ).config_dict(),
    }


def _record_batch_results(
    batch_results: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    quiet: bool,
) -> None:
    for result in batch_results:
        if result["ok"]:
            rows.append(result["row"])
            if not quiet:
                print_row(result["row"])
        else:
            failed.append(result)
            print(
                f"Scenario {result['scenario_id']} "
                f"[{result['policy_mode']}]: failed"
            )
            print(result["traceback"])


def run_sequential(
    scenario_batches: list[list[int]],
    raw_dir: Path,
    checkpoint_path: Path,
    task_config: dict[str, Any],
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    init_worker_context(str(raw_dir), str(checkpoint_path), task_config)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    iterator = (
        tqdm(
            scenario_batches,
            desc="Evaluating batches",
            unit="batch",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else scenario_batches
    )
    for batch in iterator:
        _record_batch_results(
            run_scenario_batch(batch),
            rows=rows,
            failed=failed,
            quiet=quiet,
        )
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
            _record_batch_results(
                future.result(),
                rows=rows,
                failed=failed,
                quiet=quiet,
            )
    return rows, failed


def _print_mode_summaries(
    df: pd.DataFrame,
    failures: list[dict[str, Any]],
    modes: list[str],
) -> None:
    for mode in modes:
        print(f"\nPolicy mode: {mode}")
        subset = df[df["policy_mode"] == mode]
        mode_failures = [
            item
            for item in failures
            if item.get("policy_mode", mode) == mode
        ]
        if subset.empty:
            print(f"No successful rows. Failed rows: {len(mode_failures)}")
        else:
            print_summary(df=subset, failed_results=mode_failures)


def _prepare_results_frame(
    rows: list[dict[str, Any]],
    transitions_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    defaults = {
        "policy_mode": PolicyMode.UNGATED.value,
        "constraint_changed_policy": False,
        "constraint_changed_policy_steps": 0,
        "constraint_exhausted": False,
        "empty_constrained_support_count": 0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    df = df.sort_values(
        ["scenario_id", "policy_mode"]
    ).reset_index(drop=True)
    return attach_difficulty_metadata(
        df=df,
        transitions_path=transitions_path,
    )


def evaluate_checkpoint(request: EvaluationRequest) -> dict[str, Any]:
    sequential = int(request.config.num_workers) <= 1
    try:
        for label, path in (
            ("Raw directory", request.raw_dir),
            ("Transitions CSV", request.transitions_csv),
            ("Checkpoint", request.checkpoint),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        scenario_ids = load_scenario_ids(
            request.transitions_csv,
            request.limit,
        )
        batches = chunk_list(
            scenario_ids,
            request.config.batch_size,
        )
        task = _make_task_config(request)
        print("=" * 100)
        print("Evaluating checkpoint")
        print("=" * 100)
        print(f"Policy modes: {', '.join(task['evaluation_modes'])}")
        print(
            f"Scenarios: {len(scenario_ids)} | "
            f"workers: {request.config.num_workers}"
        )

        runner = run_sequential if sequential else run_parallel
        kwargs: dict[str, Any] = {
            "scenario_batches": batches,
            "raw_dir": request.raw_dir,
            "checkpoint_path": request.checkpoint,
            "task_config": task,
            "quiet": bool(request.quiet),
        }
        if not sequential:
            kwargs["num_workers"] = int(request.config.num_workers)
        rows, failures = runner(**kwargs)
        if not rows:
            raise RuntimeError("No scenarios were successfully evaluated.")

        df = _prepare_results_frame(rows, request.transitions_csv)
        metrics = build_policy_comparison_metrics(
            df=df,
            failed_results=failures,
            requested_scenarios=len(scenario_ids),
            task_config=task,
        )
        if request.output_csv is not None:
            request.output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            df.to_csv(request.output_csv, index=False)
            print(f"\nSaved evaluation CSV: {request.output_csv}")
        if request.output_json is not None:
            save_json(
                payload=metrics,
                path=request.output_json,
            )
            print(f"\nSaved evaluation JSON: {request.output_json}")

        _print_mode_summaries(
            df,
            failures,
            list(task["evaluation_modes"]),
        )
        print_policy_comparison_summary(metrics)
        if sequential:
            context = _require_worker_context()
            for label, name in (
                ("Power flow", "backend"),
                ("Action space", "action_space"),
                ("Neural evaluator", "evaluator"),
            ):
                print(f"\n{label} cache:")
                print(context[name].cache_info())
        else:
            print("\nParallel mode uses separate per-process caches.")
        print("\nDone.")
        return metrics
    finally:
        if sequential:
            _release_worker_context()
