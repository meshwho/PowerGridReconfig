from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.value_targets import VALUE_TARGET_MODE
from grid_topology_ai.self_play.examples import ExampleWriter, SelfPlayExample
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence
from tests.topology_contract_helpers import TEST_ACTION_SPACE_CONFIG


def _writer(tmp_path: Path) -> ExampleWriter:
    return ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        branch_ids=np.asarray([3, 4], dtype=np.int64),
    )


def _pending_example(step: int) -> dict[str, object]:
    return {
        "state": _state(),
        "action_mask": np.asarray([True, True, True]),
        "scenario_id": 1,
        "step": step,
        "selected_action_id": 0,
        "selected_branch_id": None,
        "step_reward": 0.5,
        "visit_counts": {0: 4},
        "mcts_policy": {0: 1.0},
        "selection_temperature": 0.0,
        "selection_mode": "argmax",
        "policy_target_entropy": 0.0,
        "policy_target_normalized_entropy": 0.0,
        "mcts_legal_action_count": 1,
        "mcts_considered_action_count": 1,
        "mcts_visited_action_count": 1,
        "mcts_action_coverage": 1.0,
        "mcts_visited_action_coverage": 1.0,
        "extra_metadata": {"source": "test"},
    }


def _install_file_store(writer: ExampleWriter) -> None:
    def save_state(*, state_id: str, **kwargs: object) -> Path:
        path = writer.states_dir / f"{state_id}.npz"
        path.write_bytes(b"state")
        return path

    writer.state_store = SimpleNamespace(save_state=save_state)


def test_example_writer_class_name_is_explicit() -> None:
    assert ExampleWriter.__name__ == "ExampleWriter"


def test_example_writer_uses_expected_artifact_names(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    assert writer.states_dir == tmp_path / "states"
    assert writer.examples_path == tmp_path / "examples.csv"


def test_example_writers_use_distinct_run_ids(tmp_path: Path) -> None:
    first = _writer(tmp_path / "first")
    second = _writer(tmp_path / "second")

    assert first.run_id != second.run_id


def test_example_writer_rejects_off_policy_selected_action(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    pending = _pending_example(0)
    pending["selected_action_id"] = 1
    with pytest.raises(ValueError, match="outside the support"):
        writer.add_episode(
            [pending], final_return=1.0, returns_from_step=[1.0],
            solved=True, done=True, termination_reason=TerminationReason.SOLVED,
            terminal_outcome_evidence=terminal_evidence(TerminationReason.SOLVED),
            iteration=1,
        )


def test_example_writer_rejects_mismatched_terminal_evidence(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match="contradicts"):
        writer.add_episode(
            [_pending_example(0)], final_return=1.0, returns_from_step=[1.0],
            solved=True, done=True, termination_reason=TerminationReason.SOLVED,
            terminal_outcome_evidence=terminal_evidence(
                TerminationReason.MAX_STEPS_REACHED
            ), iteration=1,
        )


def test_example_writer_adds_complete_episode_targets(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    _install_file_store(writer)

    count = writer.add_episode(
        [_pending_example(0), _pending_example(1)],
        final_return=0.75,
        returns_from_step=[0.75, 0.5],
        solved=True,
        done=True,
        termination_reason=TerminationReason.SOLVED,
        terminal_outcome_evidence=terminal_evidence(
            TerminationReason.SOLVED
        ),
        iteration=2,
    )

    assert count == 2
    assert len(writer.examples) == 2
    assert {example.episode_id for example in writer.examples} == {
        writer.examples[0].episode_id
    }
    assert [example.step for example in writer.examples] == [0, 1]
    assert [
        example.outcome_steps_to_terminal
        for example in writer.examples
    ] == [2, 1]
    assert {
        example.outcome_value_target
        for example in writer.examples
    } == {1.0}
    assert {example.outcome_class for example in writer.examples} == {
        "solved"
    }
    assert {
        example.outcome_value_target_mode
        for example in writer.examples
    } == {VALUE_TARGET_MODE}
    assert {example.outcome_gamma for example in writer.examples} == {1.0}

    path = writer.save()
    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        field.name for field in fields(SelfPlayExample)
    ]
    assert frame["outcome_value_target"].tolist() == [1.0, 1.0]
    assert frame["outcome_steps_to_terminal"].tolist() == [2, 1]


def test_example_writer_rolls_back_partial_episode(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    calls = 0

    def save_state(*, state_id: str, **kwargs: object) -> Path:
        nonlocal calls
        calls += 1
        path = writer.states_dir / f"{state_id}.npz"
        if calls == 2:
            raise OSError("simulated state write failure")
        path.write_bytes(b"state")
        return path

    writer.state_store = SimpleNamespace(save_state=save_state)

    with pytest.raises(OSError, match="simulated state write failure"):
        writer.add_episode(
            [_pending_example(0), _pending_example(1)],
            final_return=-0.5,
            returns_from_step=[-0.5, -0.25],
            solved=False,
            done=True,
            termination_reason=TerminationReason.MAX_STEPS_REACHED,
            terminal_outcome_evidence=terminal_evidence(
                TerminationReason.MAX_STEPS_REACHED
            ),
            iteration=1,
        )

    assert writer.examples == []
    assert list(writer.states_dir.glob("*.npz")) == []
