"""Current Light generation determinism, multiprocessing, and resume contract."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.examples import ExampleWriter
from grid_topology_ai.self_play.generation import GenerationRequest
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


SCENARIOS = (31, 12, 27)
SEMANTIC_COLUMNS = [
    "scenario_id",
    "step",
    "episode_id",
    "state_id",
    "selected_action_id",
    "selected_branch_id",
    "mcts_policy_json",
    "termination_reason",
    "solved",
    "outcome_value_target",
]
STATE_KEYS = (
    "bus_features",
    "branch_features",
    "edge_index",
    "branch_ids",
    "branch_status",
    "action_mask",
)


def _spawn_initializer(_request: GenerationRequest) -> None:
    """Picklable test-only initializer used by the real spawn pool."""


def _state(scenario_id: int, step: int = 0) -> GridFMState:
    branch_features = np.zeros((2, len(BRANCH_FEATURE_COLUMNS)), dtype=np.float32)
    status = np.array([1.0, float(scenario_id % 2)], dtype=np.float32)
    branch_features[:, BRANCH_FEATURE_COLUMNS.index("br_status")] = status
    return GridFMState(
        scenario_id=scenario_id,
        load_scenario_idx=float(scenario_id),
        bus_features=np.pad(
            np.array([[scenario_id, step], [scenario_id + 1, step]], dtype=np.float32),
            ((0, 0), (0, len(BUS_FEATURE_COLUMNS) - 2)),
        ),
        branch_features=branch_features,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_ids=np.array([101, 202], dtype=np.int64),
        branch_status=status,
        metrics={
            "max_loading_percent": 90.0,
            "num_overloaded_branches": 0,
            "num_hard_overloaded_branches": 0,
            "total_voltage_violation": 0.0,
        },
        outaged_branch_ids=[] if status[1] else [202],
        bus_ids=np.array([10, 20], dtype=np.int64),
    )


def _spawn_scenario(scenario_id: int) -> generation._ScenarioResult:
    """Small deterministic trajectory; runs unchanged in spawned children."""
    rng = np.random.default_rng(np.random.SeedSequence([101, 202, scenario_id]))
    action = int(rng.integers(1, 3))
    branch = (101, 202)[action - 1]
    return generation._ScenarioResult(
        scenario_id=scenario_id,
        pending_examples=[
            {
                "scenario_id": scenario_id,
                "step": 0,
                "state": _state(scenario_id),
                "action_mask": [True, True, True],
                "selected_action_id": action,
                "selected_branch_id": branch,
                "step_reward": float(scenario_id) + 99.0,
                "visit_counts": {1: 3, 2: 1},
                "mcts_policy": {1: 0.75, 2: 0.25},
                "selection_temperature": 0.0,
                "selection_mode": "greedy",
            }
        ],
        rewards=[float(scenario_id) + 99.0],
        solved=True,
        done=True,
        termination_reason=TerminationReason.SOLVED,
        terminal_outcome_evidence=terminal_evidence("solved"),
    )


def _zero_scenario(scenario_id: int) -> generation._ScenarioResult:
    return generation._ScenarioResult(
        scenario_id=scenario_id,
        pending_examples=[],
        rewards=[],
        solved=True,
        done=True,
        termination_reason=TerminationReason.SOLVED,
        terminal_outcome_evidence=terminal_evidence("solved"),
    )


@pytest.fixture
def light_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "_ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(generation, "ExampleWriter", ExampleWriter)
    monkeypatch.setattr(generation, "_initialize_generation_worker", _spawn_initializer)
    monkeypatch.setattr(generation, "_generate_scenario", _spawn_scenario)


def _inputs(root: Path, ids: tuple[int, ...] = SCENARIOS) -> tuple[Path, Path, Path]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    for name in ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet"):
        (raw / name).write_bytes(name.encode())
    transitions = root / "transitions.csv"
    transitions.write_text("scenario_id\n" + "".join(f"{sid}\n" for sid in ids))
    checkpoint = root / "model.pt"
    checkpoint.write_bytes(b"deterministic checkpoint identity")
    return raw, transitions, checkpoint


def _request(inputs, output: Path, **changes) -> GenerationRequest:
    raw, transitions, checkpoint = inputs
    values = dict(
        raw_dir=raw,
        transitions_csv=transitions,
        output_dir=output,
        checkpoint=checkpoint,
        config=GenerationConfig(max_steps=1),
        mcts_seed=101,
        action_seed=202,
        clear_cache_between_scenarios=False,
        scenario_ids=SCENARIOS,
        workers=1,
    )
    values.update(changes)
    return GenerationRequest(**values)


def _canonical(path: Path) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
    frame = pd.read_csv(path)
    rows = frame[SEMANTIC_COLUMNS].to_dict("records")
    arrays = []
    for state_path in frame["state_path"]:
        with np.load(state_path, allow_pickle=False) as state:
            arrays.append({key: state[key].copy() for key in STATE_KEYS})
    return rows, arrays


def _assert_canonical_equal(left: Path, right: Path) -> None:
    left_rows, left_states = _canonical(left)
    right_rows, right_states = _canonical(right)
    assert left_rows == right_rows
    for a, b in zip(left_states, right_states, strict=True):
        for key in STATE_KEYS:
            np.testing.assert_array_equal(a[key], b[key])


def test_deterministic_trajectory_and_real_tensor_target_parity(
    tmp_path: Path,
    light_harness: None,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    first = generation.generate_self_play_examples(_request(inputs, tmp_path / "one"))
    second = generation.generate_self_play_examples(_request(inputs, tmp_path / "two"))
    _assert_canonical_equal(first, second)

    frame = pd.read_csv(first)
    dataset = GraphSelfPlayDataset(first, normalize_features=False)
    for index, row in frame.iterrows():
        sample = dataset[index]
        with np.load(row.state_path, allow_pickle=False) as state:
            for key in STATE_KEYS:
                assert key in state
            np.testing.assert_array_equal(
                sample["edge_active_mask"].numpy(),
                state["branch_status"] > 0.5,
            )
        policy = json.loads(row.mcts_policy_json)
        expected = np.zeros(3, dtype=np.float32)
        for action_id, probability in policy.items():
            expected[int(action_id)] = probability
        np.testing.assert_allclose(sample["target_policy"].numpy(), expected)
        assert sample["target_value"].item() == pytest.approx(row.outcome_value_target)
        assert sample["target_value"].item() != pytest.approx(row.step_reward)
        assert sample["target_value"].item() != pytest.approx(row.final_return)
        assert sample["target_value"].item() != pytest.approx(
            row.discounted_return_from_step
        )


def test_real_spawn_workers_and_worker_change_resume_parity(
    tmp_path: Path,
    light_harness: None,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    serial = generation.generate_self_play_examples(
        _request(inputs, tmp_path / "serial")
    )
    parallel = generation.generate_self_play_examples(
        _request(inputs, tmp_path / "parallel", workers=2)
    )
    _assert_canonical_equal(serial, parallel)

    resumed_dir = tmp_path / "worker-change"
    partial = _request(inputs, resumed_dir, workers=2, scenario_ids=SCENARIOS[:2])
    generation.generate_self_play_examples(partial)
    progress = json.loads((resumed_dir / "progress.json").read_text())
    # Preserve the original semantic identity, as a crash after A/B would.
    progress["identity"]["scenario_ids"] = list(SCENARIOS)
    (resumed_dir / "progress.json").write_text(json.dumps(progress))
    resumed = generation.generate_self_play_examples(
        _request(inputs, resumed_dir, workers=1, resume=True)
    )
    _assert_canonical_equal(serial, resumed)


def test_interruption_resume_and_csv_ahead_of_progress(
    tmp_path: Path,
    light_harness: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    clean = generation.generate_self_play_examples(_request(inputs, tmp_path / "clean"))
    calls = 0

    def interrupt(scenario_id: int):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return _spawn_scenario(scenario_id)

    monkeypatch.setattr(generation, "_generate_scenario", interrupt)
    interrupted = _request(inputs, tmp_path / "resume")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generation.generate_self_play_examples(interrupted)
    monkeypatch.setattr(generation, "_generate_scenario", _spawn_scenario)
    resumed = generation.generate_self_play_examples(replace(interrupted, resume=True))
    _assert_canonical_equal(clean, resumed)
    frame = pd.read_csv(resumed)
    assert frame.scenario_id.tolist() == list(SCENARIOS)
    assert not frame.state_id.duplicated().any()
    assert not frame.duplicated(["episode_id", "step"]).any()

    progress_path = interrupted.output_dir / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["completed_scenario_ids"].remove(SCENARIOS[1])
    progress_path.write_text(json.dumps(progress))
    before = resumed.read_bytes()
    generation.generate_self_play_examples(replace(interrupted, resume=True))
    assert resumed.read_bytes() == before


def test_zero_example_scenario_is_completed_and_not_repeated(
    tmp_path: Path,
    light_harness: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path / "inputs", (31,))
    request = _request(inputs, tmp_path / "out", scenario_ids=(31,))
    monkeypatch.setattr(generation, "_generate_scenario", _zero_scenario)
    generation.generate_self_play_examples(request)
    progress = json.loads((request.output_dir / "progress.json").read_text())
    assert progress["completed_scenario_ids"] == [31]
    monkeypatch.setattr(
        generation, "_generate_scenario", lambda _: pytest.fail("rerun")
    )
    generation.generate_self_play_examples(replace(request, resume=True))


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "transitions",
        "checkpoint",
        "scenario_order",
        "iteration",
        "mcts_seed",
        "action_seed",
        "device",
        "config",
    ],
)
def test_incompatible_resume_identity_is_rejected(
    tmp_path: Path,
    light_harness: None,
    mutation: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    request = _request(inputs, tmp_path / "out")
    generation.generate_self_play_examples(request)
    changes = {"resume": True}
    changed_target = None
    original_bytes = None
    if mutation in {"raw", "transitions", "checkpoint"}:
        changed_target = {
            "raw": inputs[0] / "bus_data.parquet",
            "transitions": inputs[1],
            "checkpoint": inputs[2],
        }[mutation]
        original_bytes = changed_target.read_bytes()
        changed_target.write_bytes(
            original_bytes + (b"\n" if mutation == "transitions" else b"changed")
        )
    elif mutation == "scenario_order":
        changes["scenario_ids"] = tuple(reversed(SCENARIOS))
    elif mutation == "config":
        changes["config"] = GenerationConfig(max_steps=2)
    else:
        changes[mutation] = {
            "iteration": 2,
            "mcts_seed": 9,
            "action_seed": 8,
            "device": "cuda",
        }[mutation]
    with pytest.raises(ValueError, match="identity does not match"):
        generation.generate_self_play_examples(replace(request, **changes))
    # Operational controls are deliberately absent from semantic identity.
    if changed_target is None:
        generation.generate_self_play_examples(replace(request, resume=True, workers=2))


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("run", "one run ID"),
        ("iteration", "iteration does not match"),
        ("state", "duplicate state_id"),
        ("episode_step", "duplicate .*episode_id, step"),
        ("episodes", "exactly one episode_id"),
        ("steps", "non-contiguous"),
    ],
)
def test_corrupted_committed_csv_is_rejected(
    tmp_path: Path,
    light_harness: None,
    corruption: str,
    message: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    request = _request(inputs, tmp_path / "out")
    path = generation.generate_self_play_examples(request)
    frame = pd.read_csv(path)
    if corruption == "run":
        frame.loc[1, "run_id"] = "other"
    elif corruption == "iteration":
        frame.loc[1, "iteration"] = 2
    elif corruption == "state":
        frame.loc[1, "state_id"] = frame.loc[0, "state_id"]
    elif corruption == "episode_step":
        frame.loc[1, ["episode_id", "step"]] = frame.loc[0, ["episode_id", "step"]]
    elif corruption == "episodes":
        frame.loc[1, "scenario_id"] = frame.loc[0, "scenario_id"]
    else:
        frame.loc[0, "step"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        generation.generate_self_play_examples(replace(request, resume=True))


@pytest.mark.parametrize(
    "missing",
    [
        "raw",
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
        "transitions",
        "checkpoint",
    ],
)
def test_preflight_fails_before_committed_examples(
    tmp_path: Path,
    light_harness: None,
    missing: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    request = _request(inputs, tmp_path / "out")
    target = {"raw": inputs[0], "transitions": inputs[1], "checkpoint": inputs[2]}.get(
        missing, inputs[0] / missing
    )
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    with pytest.raises(FileNotFoundError):
        generation.generate_self_play_examples(request)
    assert not (request.output_dir / "examples.csv").exists()


def test_invalid_existing_checkpoint_leaves_only_progress_for_follow_up(
    tmp_path: Path,
    light_harness: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    request = _request(inputs, tmp_path / "out")
    monkeypatch.setattr(
        generation,
        "_initialize_generation_worker",
        lambda _: (_ for _ in ()).throw(ValueError("invalid checkpoint")),
    )
    with pytest.raises(ValueError, match="invalid checkpoint"):
        generation.generate_self_play_examples(request)
    assert not (request.output_dir / "examples.csv").exists()
    assert (request.output_dir / "progress.json").exists()
