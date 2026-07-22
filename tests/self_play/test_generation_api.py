from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    generate_self_play_examples,
    state_security_penalty,
    terminal_outcome_reward,
)
from grid_topology_ai.termination import TerminationReason


class _FakeAction:
    branch_id = 10


class _FakeStepResult:
    reward = 1.0
    done = True
    solved = True
    info = {"termination_reason": TerminationReason.SOLVED}


class _FakeEnv:
    seen_scenarios: list[int] = []
    seen_seeds: list[int] = []

    def __init__(self, **kwargs: Any) -> None:
        self.done = False
        self.solved = False
        self.termination_reason = None
        self.current_state = object()
        self.kwargs = kwargs

    def reset(self, scenario_id: int) -> None:
        self.seen_scenarios.append(int(scenario_id))

    def valid_action_mask(self) -> list[int]:
        return [1]

    def step(self, action: object) -> _FakeStepResult:
        self.done = True
        self.solved = True
        self.termination_reason = TerminationReason.SOLVED
        return _FakeStepResult()

    def action_by_id(self, action_id: int) -> _FakeAction:
        return _FakeAction()


class _FakeCache:
    clear_calls = 0
    instances: list[Any] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.clear_calls = 0
        self.instances.append(self)

    def clear_cache(self) -> None:
        self.clear_calls += 1
        type(self).clear_calls += 1

    def cache_info(self) -> dict[str, int]:
        return {"size": 0}


class _FakeMCTSConfig:
    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs


class _FakePlanner:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def search_from_env(self, env: _FakeEnv) -> SimpleNamespace:
        action = _FakeAction()
        return SimpleNamespace(
            best_action_id=1,
            policy={1: 1.0},
            visit_counts={1: 3},
            root=SimpleNamespace(actions_by_id={1: action}),
        )


class _FakeExampleWriter:
    COLUMNS = [
        "state_id",
        "state_path",
        "scenario_id",
        "step",
        "selected_action_id",
        "selected_branch_id",
        "step_reward",
        "final_return",
        "discounted_return_from_step",
        "solved",
        "done",
        "termination_reason",
        "physical_objective_schema_version",
        "outcome_value_target_contract_version",
        "physics_config_contract_version",
        "physics_config",
        "physics_config_fingerprint",
        "visit_counts_json",
        "mcts_policy_json",
    ]

    def __init__(
        self,
        output_dir: Path,
        *,
        physics_config: PhysicsConfig,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.physics_config = physics_config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.output_dir / "states"
        self.rows: list[dict[str, object]] = []

    def add_example(self, **kwargs: object) -> None:
        provenance = physics_provenance(self.physics_config)
        self.rows.append(
            {
                "state_id": kwargs["state_id"],
                "state_path": "states/fake.npz",
                "scenario_id": kwargs["scenario_id"],
                "step": kwargs["step"],
                "selected_action_id": kwargs["selected_action_id"],
                "selected_branch_id": kwargs["selected_branch_id"],
                "step_reward": kwargs["step_reward"],
                "final_return": kwargs["final_return"],
                "discounted_return_from_step": kwargs[
                    "discounted_return_from_step"
                ],
                "solved": kwargs["solved"],
                "done": kwargs["done"],
                "termination_reason": kwargs["termination_reason"],
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
                "outcome_value_target_contract_version": (
                    OUTCOME_VALUE_TARGET_CONTRACT_VERSION
                ),
                "physics_config_contract_version": provenance[
                    "physics_config_contract_version"
                ],
                "physics_config": json.dumps(
                    provenance["physics_config"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "physics_config_fingerprint": provenance[
                    "physics_config_fingerprint"
                ],
                "visit_counts_json": '{"1": 3}',
                "mcts_policy_json": '{"1": 1.0}',
            }
        )

    def save(self) -> Path:
        path = self.output_dir / "examples.csv"
        pd.DataFrame(self.rows, columns=self.COLUMNS).to_csv(path, index=False)
        return path


@pytest.fixture
def fake_generation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEnv.seen_scenarios = []
    _FakeCache.clear_calls = 0
    _FakeCache.instances = []
    _FakeMCTSConfig.last_kwargs = None

    monkeypatch.setattr(generation, "GridFMAdapter", _FakeCache)
    monkeypatch.setattr(generation, "GridFMPowerFlowBackend", _FakeCache)
    monkeypatch.setattr(generation, "GridFMActionSpace", _FakeCache)
    monkeypatch.setattr(generation, "GridFMReward", _FakeCache)
    monkeypatch.setattr(generation, "TopologySwitchingEnv", _FakeEnv)
    monkeypatch.setattr(generation, "NeuralPolicyValueEvaluator", _FakeCache)
    monkeypatch.setattr(generation, "MCTSConfig", _FakeMCTSConfig)
    monkeypatch.setattr(generation, "MCTSPlanner", _FakePlanner)
    monkeypatch.setattr(generation, "ExampleWriter", _FakeExampleWriter)
    monkeypatch.setattr(generation, "make_do_nothing_action", lambda: object())
    monkeypatch.setattr(
        generation,
        "analyze_root_branches",
        lambda **kwargs: SimpleNamespace(
            selected_action_id=1,
            selected_branch_id=10,
            selected_reason="fake",
        ),
    )
    monkeypatch.setattr(
        generation,
        "_ensure_runtime_dependencies",
        lambda: None,
    )


def _request(tmp_path: Path, **kwargs: object) -> GenerationRequest:
    transitions_csv = tmp_path / "transitions.csv"
    if not transitions_csv.exists():
        transitions_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    values = {
        "raw_dir": tmp_path / "raw",
        "transitions_csv": transitions_csv,
        "output_dir": tmp_path / "out",
        "checkpoint": None,
        "config": GenerationConfig(max_steps=1),
        "seed": 42,
        "clear_cache_between_scenarios": False,
    }
    values.update(kwargs)
    return GenerationRequest(**values)  # type: ignore[arg-type]


def _minimal_state() -> GridFMState:
    branch_features = np.zeros((3, len(BRANCH_FEATURE_COLUMNS)), dtype=float)
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    branch_features[:, loading_idx] = [130.0, 105.0, 200.0]
    branch_features[:, status_idx] = [1.0, 1.0, 0.0]

    return GridFMState(
        scenario_id=1,
        load_scenario_idx=1.0,
        bus_features=np.zeros((1, 1), dtype=float),
        branch_features=branch_features,
        edge_index=np.zeros((2, 3), dtype=int),
        branch_ids=np.array([1, 2, 3], dtype=int),
        branch_status=np.array([1, 1, 0], dtype=int),
        metrics={
            "num_overloaded_branches": 2,
            "num_hard_overloaded_branches": 1,
            "total_voltage_violation": 0.2,
        },
        outaged_branch_ids=[],
    )


def test_state_security_penalty_works_without_generation_initialization() -> None:
    assert state_security_penalty(_minimal_state()) == 270.0


def test_terminal_outcome_reward_works_without_generation_initialization() -> None:
    assert terminal_outcome_reward(
        state=_minimal_state(),
        solved=False,
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
        terminal_unsolved_penalty=500.0,
        terminal_handoff_penalty=150.0,
        terminal_failure_penalty=1000.0,
        terminal_penalty_weight=0.1,
    ) == -527.0


def test_runtime_loader_uses_explicit_loaded_flag() -> None:
    source = Path("grid_topology_ai/self_play/generation.py").read_text(
        encoding="utf-8"
    )

    assert "_RUNTIME_DEPENDENCIES_LOADED" in source
    assert "if GridFMActionSpace is not None" not in source
    assert "ExampleWriter" in source
    assert ("SelfPlay" + "ReplayBuffer") not in source
    assert ("self_play." + "replay_buffer") not in source


def test_generation_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.seed = 1  # type: ignore[misc]

    assert not hasattr(request, "__dict__")


def test_generate_self_play_examples_creates_output(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    examples_csv = generate_self_play_examples(_request(tmp_path))

    assert examples_csv == tmp_path / "out" / "examples.csv"
    assert examples_csv.is_file()


def test_generation_preserves_scenario_order(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    request = _request(
        tmp_path,
        scenario_ids=(3, 1, 2),
        transitions_csv=tmp_path / "transitions.csv",
    )
    request.transitions_csv.write_text(
        "scenario_id\n3\n1\n2\n",
        encoding="utf-8",
    )

    generate_self_play_examples(request)

    assert _FakeEnv.seen_scenarios == [3, 1, 2]


def test_generation_uses_request_seed(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    generate_self_play_examples(_request(tmp_path, seed=123))

    assert _FakeMCTSConfig.last_kwargs is not None
    assert _FakeMCTSConfig.last_kwargs["random_seed"] == 123


def test_generation_uses_typed_config(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    config = GenerationConfig(
        simulations=17,
        depth=2,
        max_steps=3,
        top_k=11,
        gamma=0.91,
        c_puct=1.7,
        prior_exponent=0.6,
        selection_temperature=0.25,
        use_root_noise=False,
        use_continuation_gate=False,
        pf_alg=2,
        stop_policy="solved_only",
        terminal_unsolved_penalty=321.0,
        terminal_handoff_penalty=123.0,
        terminal_failure_penalty=777.0,
        terminal_penalty_weight=0.2,
    )

    generate_self_play_examples(_request(tmp_path, config=config))

    assert _FakeMCTSConfig.last_kwargs == {
        "num_simulations": 17,
        "max_depth": 2,
        "top_k_actions": 11,
        "gamma": 0.91,
        "c_puct": 1.7,
        "include_stop_action": True,
        "prior_exponent": 0.6,
        "stop_policy": "solved_only",
        "use_root_dirichlet_noise": False,
        "root_dirichlet_alpha": 0.30,
        "root_exploration_fraction": 0.25,
        "random_seed": 42,
    }


def test_clear_cache_between_scenarios(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    request = _request(
        tmp_path,
        scenario_ids=(1, 2),
        clear_cache_between_scenarios=True,
    )

    generate_self_play_examples(request)

    assert _FakeCache.clear_calls == 4

    _FakeCache.clear_calls = 0
    generate_self_play_examples(
        _request(tmp_path, output_dir=tmp_path / "out2", scenario_ids=(1, 2))
    )
    assert _FakeCache.clear_calls == 0


def test_missing_transitions_csv_raises(tmp_path: Path) -> None:
    request = GenerationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=tmp_path / "missing.csv",
        output_dir=tmp_path / "out",
        checkpoint=None,
        config=GenerationConfig(),
        seed=42,
        clear_cache_between_scenarios=False,
    )

    with pytest.raises(FileNotFoundError):
        generate_self_play_examples(request)


def test_output_schema_matches_existing_generation_schema(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    examples_csv = generate_self_play_examples(_request(tmp_path))

    assert list(pd.read_csv(examples_csv).columns) == _FakeExampleWriter.COLUMNS


def test_empty_scenario_selection_preserves_old_behavior(
    tmp_path: Path,
    fake_generation_runtime: None,
) -> None:
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text("scenario_id\n", encoding="utf-8")

    examples_csv = generate_self_play_examples(
        _request(tmp_path, transitions_csv=transitions_csv)
    )

    assert examples_csv.is_file()
    assert _FakeEnv.seen_scenarios == []
