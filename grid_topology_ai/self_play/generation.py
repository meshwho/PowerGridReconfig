from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config import (
    GenerationConfig,
    PhysicsConfig,
    resolve_physics_config,
)
from grid_topology_ai.search.mcts import (
    require_action_in_policy_support,
    select_policy_action,
)

_RUNTIME_DEPENDENCIES_LOADED = False

GridFMActionSpace = None
TopologySwitchingEnv = None
GridFMPowerFlowBackend = None
GridFMReward = None
GridFMAdapter = None
NeuralPolicyValueEvaluator = None
MCTSConfig = None
MCTSPlanner = None
ExampleWriter = None
make_do_nothing_action = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    raw_dir: Path
    transitions_csv: Path
    output_dir: Path
    checkpoint: Path | None
    config: GenerationConfig
    mcts_seed: int
    action_seed: int
    clear_cache_between_scenarios: bool
    iteration: int = 1
    physics_config: PhysicsConfig | None = None
    scenario_ids: tuple[int, ...] | None = None
    device: str = "cpu"
    enable_cache: bool = True
    root_dirichlet_alpha: float = 0.30
    root_exploration_fraction: float = 0.25
    min_hard_improvement: float = 50.0
    min_soft_improvement: float = 15.0
    min_gate_visits: int = 5
    min_gate_visit_fraction: float = 0.01
    workers: int = 1
    resume: bool = False

    def __post_init__(self) -> None:
        for field_name in ("mcts_seed", "action_seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field_name} must be a non-negative integer.")
            seed = int(value)
            if seed < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
            object.__setattr__(self, field_name, seed)

        if isinstance(self.iteration, bool) or not isinstance(
            self.iteration, (int, np.integer)
        ):
            raise ValueError("iteration must be a positive integer.")
        iteration = int(self.iteration)
        if iteration <= 0:
            raise ValueError("iteration must be a positive integer.")
        object.__setattr__(self, "iteration", iteration)

        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, (int, np.integer))
            or int(self.workers) < 1
        ):
            raise ValueError("workers must be a positive integer.")
        object.__setattr__(self, "workers", int(self.workers))

    @property
    def resolved_physics_config(self) -> PhysicsConfig:
        return resolve_physics_config(
            self.physics_config,
            self.config.pf_alg,
        )


@dataclass(frozen=True, slots=True)
class _GenerationActionDecision:
    selected_action_id: int
    selected_branch_id: int | None
    policy_target: dict[int, float]


def _ensure_runtime_dependencies() -> None:
    global _RUNTIME_DEPENDENCIES_LOADED
    global GridFMActionSpace
    global TopologySwitchingEnv
    global GridFMPowerFlowBackend
    global GridFMReward
    global GridFMAdapter
    global NeuralPolicyValueEvaluator
    global MCTSConfig
    global MCTSPlanner
    global ExampleWriter
    global make_do_nothing_action

    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    from grid_topology_ai.actions import GridFMActionSpace as _ActionSpace
    from grid_topology_ai.actions import (
        make_do_nothing_action as _make_do_nothing_action,
    )
    from grid_topology_ai.data import GridFMAdapter as _Adapter
    from grid_topology_ai.environment import TopologySwitchingEnv as _Env
    from grid_topology_ai.evaluator import (
        NeuralPolicyValueEvaluator as _Evaluator,
    )
    from grid_topology_ai.physics.utility import GridFMReward as _Reward
    from grid_topology_ai.power_flow.backend import (
        GridFMPowerFlowBackend as _Backend,
    )
    from grid_topology_ai.search.mcts import MCTSConfig as _MCTSConfig
    from grid_topology_ai.search.mcts import MCTSPlanner as _MCTSPlanner
    from grid_topology_ai.self_play.examples import (
        ExampleWriter as _ExampleWriter,
    )

    GridFMActionSpace = _ActionSpace
    TopologySwitchingEnv = _Env
    GridFMPowerFlowBackend = _Backend
    GridFMReward = _Reward
    GridFMAdapter = _Adapter
    NeuralPolicyValueEvaluator = _Evaluator
    MCTSConfig = _MCTSConfig
    MCTSPlanner = _MCTSPlanner
    ExampleWriter = _ExampleWriter
    make_do_nothing_action = _make_do_nothing_action
    _RUNTIME_DEPENDENCIES_LOADED = True


def discounted_returns(
    rewards: list[float],
    gamma: float,
) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _scenario_seed(stream_seed: int, scenario_id: int) -> int:
    sequence = np.random.SeedSequence([int(stream_seed), int(scenario_id)])
    state = sequence.generate_state(1, dtype=np.uint64)
    return int(state[0])


def selection_temperature_for_step(
    config: GenerationConfig,
    *,
    iteration: int,
    step: int,
) -> float:
    if iteration <= 0:
        raise ValueError("iteration must be positive.")
    if step < 0:
        raise ValueError("step must be non-negative.")
    if config.selection_temperature <= 0.0:
        return 0.0
    if config.temperature_iterations <= 0 or config.temperature_steps <= 0:
        return 0.0
    if iteration > config.temperature_iterations:
        return 0.0
    if step >= config.temperature_steps:
        return 0.0
    return float(config.selection_temperature)


def _select_generation_action(
    *,
    search_result: Any,
    temperature: float,
    rng: np.random.Generator,
    scenario_id: int | None = None,
    step: int | None = None,
) -> _GenerationActionDecision:
    context = f"self-play behavior policy (scenario_id={scenario_id}, step={step})"
    selection = select_policy_action(
        search_result.policy,
        temperature,
        rng,
        context=context,
    )
    selected_action_id = int(selection.action_id)
    policy_target = dict(selection.policy)
    require_action_in_policy_support(
        selected_action_id,
        policy_target,
        context=context,
    )

    if selected_action_id == 0:
        selected_branch_id = None
    else:
        selected_action = search_result.root.actions_by_id.get(selected_action_id)
        if selected_action is None:
            raise RuntimeError(
                f"Action {selected_action_id} is present in {context} but "
                "missing from root.actions_by_id."
            )
        selected_branch_id = selected_action.branch_id

    return _GenerationActionDecision(
        selected_action_id=selected_action_id,
        selected_branch_id=selected_branch_id,
        policy_target=policy_target,
    )


def _scenario_ids_from_request(
    request: GenerationRequest,
) -> list[int]:
    if not request.transitions_csv.is_file():
        raise FileNotFoundError(
            f"Transitions file not found: {request.transitions_csv}"
        )
    transitions = pd.read_csv(request.transitions_csv)
    if "scenario_id" not in transitions.columns:
        raise ValueError(
            f"Transitions CSV is missing scenario_id: {request.transitions_csv}"
        )

    available = {int(value) for value in transitions["scenario_id"].unique()}
    if request.scenario_ids is None:
        scenario_ids = sorted(available)
    else:
        scenario_ids = [int(value) for value in request.scenario_ids]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must not contain duplicates.")
        missing = [value for value in scenario_ids if value not in available]
        if missing:
            raise ValueError(
                "scenario_ids contains values missing from transitions: "
                f"{missing[:20]}."
            )

    if not scenario_ids:
        raise ValueError("No self-play scenarios were selected.")
    return scenario_ids


def _step_metadata(
    *,
    request: GenerationRequest,
    scenario_id: int,
    step: int,
    scenario_mcts_seed: int,
    scenario_action_seed: int,
    selection_temperature: float,
    selection_mode: str,
    search_result: Any,
) -> dict[str, Any]:
    return {
        "source": "mcts_self_play",
        "scenario_id": int(scenario_id),
        "step": int(step),
        "mcts_stream_seed": int(request.mcts_seed),
        "action_sampling_stream_seed": int(request.action_seed),
        "scenario_mcts_seed": int(scenario_mcts_seed),
        "scenario_action_sampling_seed": int(scenario_action_seed),
        "self_play_iteration": int(request.iteration),
        "selection_temperature": float(selection_temperature),
        "selection_mode": selection_mode,
        "temperature_steps": int(request.config.temperature_steps),
        "temperature_iterations": int(request.config.temperature_iterations),
        "mcts_simulations": int(request.config.simulations),
        "mcts_depth": int(request.config.depth),
        "mcts_top_k": int(request.config.top_k),
        "mcts_widening_coefficient": float(request.config.widening_coefficient),
        "mcts_widening_exponent": float(request.config.widening_exponent),
        "mcts_exploration_quota": int(request.config.exploration_quota),
        "pf_alg": request.resolved_physics_config.pf_alg,
        "policy_target_source": ("temperature_adjusted_mcts_visit_distribution"),
        "execution_action_source": "policy_target_sampling",
        "mcts_best_action_id": (
            None
            if search_result.best_action_id is None
            else int(search_result.best_action_id)
        ),
        "mcts_best_branch_id": (
            None
            if getattr(search_result, "best_branch_id", None) is None
            else int(search_result.best_branch_id)
        ),
    }


_WORKER_RUNTIME: dict[str, Any] | None = None


@dataclass(slots=True)
class _ScenarioResult:
    scenario_id: int
    pending_examples: list[dict[str, Any]]
    rewards: list[float]
    solved: bool
    done: bool
    termination_reason: Any
    terminal_outcome_evidence: Any


def _build_runtime(request: GenerationRequest) -> dict[str, Any]:
    _ensure_runtime_dependencies()
    adapter = GridFMAdapter(
        request.raw_dir,
        physics_config=request.resolved_physics_config,
    )
    backend = GridFMPowerFlowBackend(
        adapter=adapter,
        physics_config=request.resolved_physics_config,
        enable_cache=request.enable_cache,
    )
    action_config = request.config.action_space_config
    action_space = GridFMActionSpace(
        require_connected_after_switch=(action_config.require_connected_after_switch),
        min_loading_for_switch_percent=(action_config.min_loading_for_switch_percent),
        closeable_branch_ids=action_config.closeable_branch_ids,
        enable_cache=request.enable_cache,
    )
    reward_fn = GridFMReward(
        physics_config=request.resolved_physics_config,
        discount_factor=request.config.gamma,
    )
    mcts_config = MCTSConfig(
        num_simulations=request.config.simulations,
        max_depth=request.config.depth,
        top_k_actions=request.config.top_k,
        widening_coefficient=request.config.widening_coefficient,
        widening_exponent=request.config.widening_exponent,
        exploration_quota=request.config.exploration_quota,
        gamma=request.config.gamma,
        c_puct=request.config.c_puct,
        include_stop_action=True,
        prior_exponent=request.config.prior_exponent,
        stop_policy=request.config.stop_policy,
        use_root_dirichlet_noise=request.config.use_root_noise,
        root_dirichlet_alpha=request.root_dirichlet_alpha,
        root_exploration_fraction=request.root_exploration_fraction,
        random_seed=request.mcts_seed,
    )
    evaluator = None
    if request.checkpoint is not None:
        evaluator = NeuralPolicyValueEvaluator(
            checkpoint_path=request.checkpoint,
            device=request.device,
            enable_cache=request.enable_cache,
            physics_config=request.resolved_physics_config,
        )
        if (
            evaluator.topology_action_config.contract_fingerprint()
            != action_space.config.contract_fingerprint()
        ):
            raise ValueError(
                "Configured self-play topology action space does not match "
                f"checkpoint {request.checkpoint}."
            )
    planner = MCTSPlanner(
        config=mcts_config,
        evaluator=evaluator,
        physics_config=request.resolved_physics_config,
    )
    return {
        "request": request,
        "adapter": adapter,
        "backend": backend,
        "action_space": action_space,
        "reward_fn": reward_fn,
        "evaluator": evaluator,
        "planner": planner,
    }


def _initialize_generation_worker(request: GenerationRequest) -> None:
    """Create one reusable runtime in each spawned worker process."""
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = _build_runtime(request)


def _release_generation_worker_runtime() -> None:
    """Drop a parent-side preflight runtime before spawning real workers."""
    global _WORKER_RUNTIME
    runtime = _WORKER_RUNTIME
    if runtime is None:
        return
    evaluator = runtime.get("evaluator")
    uses_cuda = (
        evaluator is not None
        and getattr(evaluator, "device", None) is not None
        and evaluator.device.type == "cuda"
    )
    _WORKER_RUNTIME = None
    del evaluator
    del runtime
    if uses_cuda:
        import torch

        torch.cuda.empty_cache()


def _generate_scenario(scenario_id: int) -> _ScenarioResult:
    """Generate one episode using only scenario-derived random streams."""
    if _WORKER_RUNTIME is None:
        raise RuntimeError("Self-play worker runtime was not initialized.")
    runtime = _WORKER_RUNTIME
    request: GenerationRequest = runtime["request"]
    backend = runtime["backend"]
    action_space = runtime["action_space"]
    evaluator = runtime["evaluator"]
    planner = runtime["planner"]

    if request.clear_cache_between_scenarios:
        backend.clear_cache()
        action_space.clear_cache()
        if evaluator is not None:
            evaluator.clear_cache()

    scenario_mcts_seed = _scenario_seed(request.mcts_seed, scenario_id)
    scenario_action_seed = _scenario_seed(request.action_seed, scenario_id)
    planner.reset_rng(scenario_mcts_seed)
    action_rng = np.random.default_rng(scenario_action_seed)
    env = TopologySwitchingEnv(
        adapter=runtime["adapter"],
        backend=backend,
        action_space=action_space,
        reward_fn=runtime["reward_fn"],
        max_steps=request.config.max_steps,
    )
    env.reset(scenario_id)
    pending_examples: list[dict[str, Any]] = []
    rewards: list[float] = []

    for step in range(request.config.max_steps):
        if env.done:
            break
        state_before = env.current_state
        if state_before is None:
            raise RuntimeError("Active self-play environment has no current state.")
        action_mask = env.operational_action_mask()
        search_result = planner.search_from_env(env)
        if search_result.best_action_id is None:
            env.terminate_no_legal_action()
            break
        temperature = selection_temperature_for_step(
            request.config, iteration=request.iteration, step=step
        )
        selection_mode = "sample" if temperature > 1e-8 else "argmax"
        decision = _select_generation_action(
            search_result=search_result,
            temperature=temperature,
            rng=action_rng,
            scenario_id=scenario_id,
            step=step,
        )
        require_action_in_policy_support(
            decision.selected_action_id,
            decision.policy_target,
            context=(
                f"self-play policy target (scenario_id={scenario_id}, step={step})"
            ),
        )
        selected_action = (
            make_do_nothing_action()
            if decision.selected_action_id == 0
            else search_result.root.actions_by_id[decision.selected_action_id]
        )
        step_result = env.step(selected_action)
        rewards.append(float(step_result.reward))
        pending_examples.append(
            {
                "state": state_before,
                "action_mask": action_mask,
                "scenario_id": scenario_id,
                "step": step,
                "selected_action_id": decision.selected_action_id,
                "selected_branch_id": decision.selected_branch_id,
                "step_reward": float(step_result.reward),
                "visit_counts": search_result.visit_counts,
                "mcts_policy": decision.policy_target,
                "selection_temperature": float(temperature),
                "selection_mode": selection_mode,
                "extra_metadata": _step_metadata(
                    request=request,
                    scenario_id=scenario_id,
                    step=step,
                    scenario_mcts_seed=scenario_mcts_seed,
                    scenario_action_seed=scenario_action_seed,
                    selection_temperature=temperature,
                    selection_mode=selection_mode,
                    search_result=search_result,
                ),
            }
        )
        if step_result.done:
            break

    if (
        not env.done
        or env.termination_reason is None
        or env.terminal_outcome_evidence is None
    ):
        raise RuntimeError(
            "Self-play episode ended without validated terminal outcome "
            f"evidence for scenario {scenario_id}."
        )
    return _ScenarioResult(
        scenario_id=int(scenario_id),
        pending_examples=pending_examples,
        rewards=rewards,
        solved=bool(env.solved),
        done=bool(env.done),
        termination_reason=env.termination_reason,
        terminal_outcome_evidence=env.terminal_outcome_evidence,
    )


def _source_identity(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "size": None, "mtime_ns": None}
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


_RAW_SOURCE_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
)


def _preflight_generation_inputs(request: GenerationRequest) -> None:
    if not request.raw_dir.is_dir():
        raise FileNotFoundError(f"GridFM raw directory not found: {request.raw_dir}")
    for name in _RAW_SOURCE_FILES:
        path = request.raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required GridFM parquet not found: {path}")
    if not request.transitions_csv.is_file():
        raise FileNotFoundError(
            f"Transitions file not found: {request.transitions_csv}"
        )
    if request.checkpoint is not None and not request.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {request.checkpoint}")


def _raw_source_identity(raw_dir: Path) -> dict[str, object]:
    return {name: _source_identity(raw_dir / name) for name in _RAW_SOURCE_FILES}


def _resume_identity(
    request: GenerationRequest, scenario_ids: list[int]
) -> dict[str, object]:
    config = asdict(request.config)
    config["closeable_branch_ids"] = list(config["closeable_branch_ids"])
    config.pop("use_continuation_gate", None)
    return {
        "raw_source": _raw_source_identity(request.raw_dir),
        "transitions": _source_identity(request.transitions_csv),
        "checkpoint": _source_identity(request.checkpoint),
        "scenario_ids": list(scenario_ids),
        "iteration": request.iteration,
        "mcts_seed": request.mcts_seed,
        "action_seed": request.action_seed,
        "device": str(request.device),
        "physics_config": request.resolved_physics_config.to_dict(),
        "action_space_config": request.config.action_space_config.to_contract_dict(),
        "generation_config": config,
        "root_dirichlet_alpha": request.root_dirichlet_alpha,
        "root_exploration_fraction": request.root_exploration_fraction,
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_generation_progress(
    request: GenerationRequest, identity: dict[str, object]
) -> tuple[str, set[int]]:
    progress_path = request.output_dir / "progress.json"
    examples_path = request.output_dir / "examples.csv"
    if request.resume:
        if not progress_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without progress file: {progress_path}"
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Self-play resume identity does not match this request.")
        raw_run_id = progress.get("run_id")
        if not isinstance(raw_run_id, str) or not raw_run_id.strip():
            raise ValueError("progress.json run_id must be a non-empty string.")
        run_id = raw_run_id
        requested = set(identity["scenario_ids"])
        completed = {int(value) for value in progress.get("completed_scenario_ids", [])}
        if not completed <= requested:
            raise ValueError(
                "progress.json contains scenario IDs outside this request."
            )
        if examples_path.exists():
            frame = pd.read_csv(examples_path)
            required = {
                "run_id",
                "iteration",
                "episode_id",
                "scenario_id",
                "step",
                "state_id",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    "examples.csv is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            if frame.empty:
                return run_id, completed
            if frame[list(required)].isna().any().any():
                raise ValueError("examples.csv required columns contain null values.")
            run_ids = frame["run_id"].astype(str).unique()
            if len(run_ids) != 1:
                raise ValueError("examples.csv must contain exactly one run ID.")
            if str(run_ids[0]) != run_id:
                raise ValueError("examples.csv run ID does not match progress.json.")
            try:
                numeric_columns = {
                    name: pd.to_numeric(frame[name], errors="raise")
                    for name in ("iteration", "scenario_id", "step")
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "examples.csv iteration, scenario_id, and step must be integers."
                ) from exc
            if any(
                not np.isfinite(values).all()
                or not (values == values.astype(np.int64)).all()
                for values in numeric_columns.values()
            ):
                raise ValueError(
                    "examples.csv iteration, scenario_id, and step must be integers."
                )
            iterations = numeric_columns["iteration"].astype(np.int64)
            scenarios = numeric_columns["scenario_id"].astype(np.int64)
            steps = numeric_columns["step"].astype(np.int64)
            if not (iterations == request.iteration).all():
                raise ValueError("examples.csv iteration does not match request.")
            if not set(scenarios) <= requested:
                raise ValueError(
                    "examples.csv contains scenario IDs outside this request."
                )
            validated = frame.assign(
                iteration=iterations, scenario_id=scenarios, step=steps
            )
            if validated["state_id"].astype(str).duplicated().any():
                raise ValueError("examples.csv contains duplicate state_id values.")
            if validated.duplicated(["episode_id", "step"]).any():
                raise ValueError(
                    "examples.csv contains duplicate (episode_id, step) values."
                )
            if (validated.groupby("scenario_id")["episode_id"].nunique() != 1).any():
                raise ValueError(
                    "Each scenario_id must correspond to exactly one episode_id."
                )
            if (validated.groupby("episode_id")["scenario_id"].nunique() != 1).any():
                raise ValueError(
                    "Each episode_id must correspond to exactly one scenario_id."
                )
            for episode_id, episode in validated.groupby("episode_id"):
                episode_steps = sorted(int(value) for value in episode["step"])
                if episode_steps != list(range(len(episode_steps))):
                    raise ValueError(
                        f"examples.csv episode {episode_id!r} has non-contiguous steps."
                    )
            completed.update(int(value) for value in scenarios.unique())
        return run_id, completed

    if progress_path.exists() or examples_path.exists():
        raise FileExistsError(
            "Self-play output already exists; choose a new output directory or use resume."
        )
    run_id = uuid.uuid4().hex
    _atomic_json(
        progress_path,
        {
            "identity": identity,
            "completed_scenario_ids": [],
            "run_id": run_id,
        },
    )
    return run_id, set()


def generate_self_play_examples(request: GenerationRequest) -> Path:
    _preflight_generation_inputs(request)
    scenario_ids = _scenario_ids_from_request(request)
    identity = _resume_identity(request, scenario_ids)

    # A fresh run must prove that the complete runtime is constructible before
    # committing progress.  In particular, an existing but invalid checkpoint
    # or an incompatible topology contract must not poison an otherwise reusable
    # output directory with a progress.json file.
    runtime_prepared = False
    if not request.resume:
        _initialize_generation_worker(request)
        runtime_prepared = request.workers == 1
        if not runtime_prepared:
            _release_generation_worker_runtime()

    request.output_dir.mkdir(parents=True, exist_ok=True)
    run_id, completed = _load_generation_progress(request, identity)
    remaining = [sid for sid in scenario_ids if sid not in completed]

    _ensure_runtime_dependencies()
    writer_kwargs: dict[str, object] = {
        "physics_config": request.resolved_physics_config,
        "action_space_config": request.config.action_space_config,
    }
    if "run_id" in inspect.signature(ExampleWriter.__init__).parameters:
        writer_kwargs["run_id"] = run_id
    writer = ExampleWriter(request.output_dir, **writer_kwargs)

    if request.workers == 1:
        if not runtime_prepared:
            _initialize_generation_worker(request)
        results = map(_generate_scenario, remaining)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=request.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_generation_worker,
            initargs=(request,),
        )
        results = executor.map(_generate_scenario, remaining)

    try:
        for result in results:
            returns = discounted_returns(result.rewards, request.config.gamma)
            if result.pending_examples:
                writer.add_episode(
                    result.pending_examples,
                    final_return=returns[0] if returns else 0.0,
                    returns_from_step=returns,
                    solved=result.solved,
                    done=result.done,
                    termination_reason=result.termination_reason,
                    terminal_outcome_evidence=result.terminal_outcome_evidence,
                    iteration=request.iteration,
                )
                writer.save()
            completed.add(result.scenario_id)
            ordered_completed = [sid for sid in scenario_ids if sid in completed]
            _atomic_json(
                request.output_dir / "progress.json",
                {
                    "identity": identity,
                    "completed_scenario_ids": ordered_completed,
                    "run_id": run_id,
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    examples_path = request.output_dir / "examples.csv"
    if not examples_path.exists():
        writer.save()
    return examples_path
