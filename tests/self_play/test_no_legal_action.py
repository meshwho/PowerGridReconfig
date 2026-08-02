from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.outcome_contract import (
    RedispatchStatus,
    TerminalOutcomeEvidence,
)
from grid_topology_ai.pypower_backend import GridFMPowerFlowResult
from grid_topology_ai.return_contract import terminal_utility_from_outcome
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    generate_self_play_examples,
)
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


def _unsafe_state() -> GridFMState:
    import numpy as np

    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((1, 4), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([10], dtype=np.int64),
        branch_status=np.array([1], dtype=np.int64),
        metrics={
            "power_flow_converged": True,
            "all_values_finite": True,
            "topology_connected": True,
            "max_loading_percent": 110.0,
            "num_overloaded_branches": 1,
            "num_hard_overloaded_branches": 0,
            "total_thermal_overload_mva": 4.0,
            "num_low_voltage_buses": 0,
            "num_high_voltage_buses": 0,
            "total_voltage_violation": 0.0,
            "num_generator_p_violations": 0,
            "total_generator_p_violation_mw": 0.0,
            "num_generator_q_violations": 0,
            "total_generator_q_violation_mvar": 0.0,
            "num_angle_difference_violations": 0,
            "total_angle_difference_violation_degrees": 0.0,
        },
        outaged_branch_ids=[],
    )


class _InitialStateBackend:
    def __init__(self, state: GridFMState) -> None:
        self.state = state

    def run_power_flow(
        self,
        scenario_id: int,
        switched_off_branch_id: int | None = None,
    ) -> GridFMPowerFlowResult:
        return GridFMPowerFlowResult(
            success=True,
            scenario_id=int(scenario_id),
            switched_off_branch_id=switched_off_branch_id,
            next_state=self.state,
            raw_result=None,
            message="fake initial power flow",
        )


def test_environment_classifies_no_legal_action() -> None:
    env = TopologySwitchingEnv(
        adapter=object(),
        backend=_InitialStateBackend(_unsafe_state()),
        action_space=object(),
        reward_fn=object(),
    )
    env.reset(1)

    evidence = env.terminate_no_legal_action()

    assert env.done is True
    assert env.solved is False
    assert env.termination_reason is TerminationReason.NO_LEGAL_ACTION
    assert env.terminal_outcome_evidence is evidence
    assert evidence.redispatch_status is RedispatchStatus.NOT_REQUESTED
    assert evidence.assessment is not None
    assert evidence.assessment.physically_secure is False
    assert env.step_count == 0
    assert env.applied_actions == []

    with pytest.raises(RuntimeError, match="Episode is already done"):
        env.terminate_no_legal_action()


def test_no_legal_action_evidence_and_utility_contract() -> None:
    evidence = terminal_evidence(TerminationReason.NO_LEGAL_ACTION)

    assert evidence.termination_reason is TerminationReason.NO_LEGAL_ACTION
    assert evidence.redispatch_status is RedispatchStatus.NOT_REQUESTED
    assert terminal_utility_from_outcome(
        False,
        TerminationReason.NO_LEGAL_ACTION,
        evidence=evidence,
    ) == (-1.0, "no_legal_action")

    with pytest.raises(ValueError, match="physical assessment"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.NO_LEGAL_ACTION,
            assessment=None,
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
        )

    secure_assessment = terminal_evidence(
        TerminationReason.SOLVED
    ).assessment
    assert secure_assessment is not None
    with pytest.raises(ValueError, match="physically_secure"):
        TerminalOutcomeEvidence(
            solved=False,
            termination_reason=TerminationReason.NO_LEGAL_ACTION,
            assessment=secure_assessment,
            redispatch_status=RedispatchStatus.NOT_REQUESTED,
        )


class _Action:
    branch_id = 10


class _RuntimeObject:
    def __init__(self, *args, **kwargs) -> None:
        self.config = SimpleNamespace()

    def clear_cache(self) -> None:
        return None

    def cache_info(self) -> dict[str, int]:
        return {"size": 0}


class _Writer:
    instances: list["_Writer"] = []

    def __init__(
        self,
        output_dir: Path,
        *,
        physics_config,
        action_space_config,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.states_dir = self.output_dir / "states"
        self.rows: list[dict[str, object]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        type(self).instances.append(self)

    def add_example(self, **kwargs: object) -> None:
        evidence = kwargs["terminal_outcome_evidence"]
        assert isinstance(evidence, TerminalOutcomeEvidence)
        self.rows.append(
            {
                "done": kwargs["done"],
                "solved": kwargs["solved"],
                "termination_reason": kwargs["termination_reason"],
                "terminal_outcome_evidence_json": evidence.to_json(),
            }
        )

    def save(self) -> Path:
        path = self.output_dir / "examples.csv"
        pd.DataFrame(
            self.rows,
            columns=[
                "done",
                "solved",
                "termination_reason",
                "terminal_outcome_evidence_json",
            ],
        ).to_csv(path, index=False)
        return path


class _GenerationEnv:
    instances: list["_GenerationEnv"] = []

    def __init__(self, **kwargs) -> None:
        self.done = False
        self.solved = False
        self.termination_reason = None
        self.terminal_outcome_evidence = None
        self.current_state = object()
        self.step_count = 0
        self.applied_actions: list[object] = []
        type(self).instances.append(self)

    def reset(self, scenario_id: int) -> None:
        return None

    def valid_action_mask(self) -> list[bool]:
        return [True, True]

    def step(self, action: object) -> SimpleNamespace:
        self.step_count += 1
        self.applied_actions.append(action)
        return SimpleNamespace(
            reward=0.5,
            done=False,
            solved=False,
            info={"termination_reason": None},
        )

    def terminate_no_legal_action(self) -> TerminalOutcomeEvidence:
        evidence = terminal_evidence(TerminationReason.NO_LEGAL_ACTION)
        self.done = True
        self.solved = False
        self.termination_reason = evidence.termination_reason
        self.terminal_outcome_evidence = evidence
        return evidence


class _Planner:
    action_before_failure = False

    def __init__(self, **kwargs) -> None:
        self.calls = 0

    def reset_rng(self, random_seed: int) -> None:
        return None

    def search_from_env(self, env: _GenerationEnv) -> SimpleNamespace:
        self.calls += 1
        if not self.action_before_failure or self.calls > 1:
            return SimpleNamespace(best_action_id=None)
        return SimpleNamespace(
            best_action_id=1,
            best_branch_id=10,
            policy={1: 1.0},
            visit_counts={1: 3},
            root=SimpleNamespace(actions_by_id={1: _Action()}),
            root_legal_action_count=1,
            root_considered_action_count=1,
            root_visited_action_count=1,
            root_action_coverage=1.0,
            root_visited_action_coverage=1.0,
        )


def _install_generation_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Writer.instances = []
    _GenerationEnv.instances = []
    monkeypatch.setattr(
        generation,
        "_ensure_runtime_dependencies",
        lambda: None,
    )
    monkeypatch.setattr(generation, "GridFMAdapter", _RuntimeObject)
    monkeypatch.setattr(
        generation,
        "GridFMPowerFlowBackend",
        _RuntimeObject,
    )
    monkeypatch.setattr(
        generation,
        "GridFMActionSpace",
        _RuntimeObject,
    )
    monkeypatch.setattr(generation, "GridFMReward", _RuntimeObject)
    monkeypatch.setattr(generation, "MCTSConfig", _RuntimeObject)
    monkeypatch.setattr(generation, "MCTSPlanner", _Planner)
    monkeypatch.setattr(
        generation,
        "TopologySwitchingEnv",
        _GenerationEnv,
    )
    monkeypatch.setattr(generation, "ExampleWriter", _Writer)
    monkeypatch.setattr(
        generation,
        "make_do_nothing_action",
        lambda: object(),
    )


def _request(tmp_path: Path, *, max_steps: int) -> GenerationRequest:
    transitions = tmp_path / "transitions.csv"
    transitions.write_text("scenario_id\n1\n", encoding="utf-8")
    return GenerationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions,
        output_dir=tmp_path / "out",
        checkpoint=None,
        config=GenerationConfig(max_steps=max_steps),
        mcts_seed=7,
        action_seed=8,
        clear_cache_between_scenarios=False,
    )


def test_generation_terminates_without_synthetic_first_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_generation_runtime(monkeypatch)
    _Planner.action_before_failure = False

    examples_csv = generate_self_play_examples(
        _request(tmp_path, max_steps=1)
    )

    env = _GenerationEnv.instances[0]
    assert env.done is True
    assert env.termination_reason is TerminationReason.NO_LEGAL_ACTION
    assert env.applied_actions == []
    assert _Writer.instances[0].rows == []
    assert examples_csv.is_file()


def test_generation_labels_prior_decisions_as_no_legal_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_generation_runtime(monkeypatch)
    _Planner.action_before_failure = True

    examples_csv = generate_self_play_examples(
        _request(tmp_path, max_steps=2)
    )

    env = _GenerationEnv.instances[0]
    assert len(env.applied_actions) == 1
    row = pd.read_csv(examples_csv).iloc[0]
    assert bool(row["done"]) is True
    assert bool(row["solved"]) is False
    assert row["termination_reason"] == "no_legal_action"
    assert row["terminal_outcome_evidence_json"] == terminal_evidence(
        TerminationReason.NO_LEGAL_ACTION
    ).to_json()
