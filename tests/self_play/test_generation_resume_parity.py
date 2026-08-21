from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.generation import GenerationRequest
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


class _Writer:
    def __init__(self, output_dir: Path, *, physics_config, action_space_config, run_id: str):
        self.path = Path(output_dir) / "examples.csv"
        self.run_id = run_id
        self.rows = [] if not self.path.exists() else pd.read_csv(self.path).to_dict("records")

    def add_episode(self, pending_examples, *, iteration: int, **kwargs) -> int:
        for item in pending_examples:
            scenario_id, step = int(item["scenario_id"]), int(item["step"])
            episode = f"iteration_{iteration:06d}_scenario_{scenario_id}"
            self.rows.append({
                "run_id": self.run_id, "iteration": iteration, "episode_id": episode,
                "scenario_id": scenario_id, "step": step,
                "state_id": f"{episode}_step_{step:03d}",
                "selected_action_id": int(item["selected_action_id"]),
            })
        return len(pending_examples)

    def save(self) -> Path:
        pd.DataFrame(self.rows, columns=[
            "run_id", "iteration", "episode_id", "scenario_id", "step",
            "state_id", "selected_action_id",
        ]).to_csv(self.path, index=False)
        return self.path


class _InlineExecutor:
    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)
    def map(self, function, values):
        return map(function, values)
    def shutdown(self, **kwargs): pass


def _request(raw: Path, transitions: Path, output: Path, *, workers: int = 1, resume: bool = False):
    return GenerationRequest(
        raw_dir=raw, transitions_csv=transitions, output_dir=output,
        checkpoint=None, config=GenerationConfig(max_steps=1), mcts_seed=101,
        action_seed=202, clear_cache_between_scenarios=False,
        scenario_ids=(3, 1, 2), workers=workers, resume=resume,
    )


@pytest.fixture
def fake_generation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generation, "_ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(generation, "ExampleWriter", _Writer)
    monkeypatch.setattr(generation.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed-run"))
    monkeypatch.setattr(generation, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(generation, "_initialize_generation_worker", lambda request: None)

    def scenario(scenario_id: int):
        return generation._ScenarioResult(
            scenario_id=scenario_id,
            pending_examples=[{"scenario_id": scenario_id, "step": 0, "selected_action_id": scenario_id}],
            rewards=[float(scenario_id)], solved=True, done=True,
            termination_reason=TerminationReason.SOLVED,
            terminal_outcome_evidence=terminal_evidence("solved"),
        )
    monkeypatch.setattr(generation, "_generate_scenario", scenario)
    return scenario


def test_single_and_multi_worker_outputs_have_deterministic_parity(
    generation_inputs, tmp_path: Path, fake_generation
) -> None:
    raw, transitions = generation_inputs((1, 2, 3))
    serial = generation.generate_self_play_examples(_request(raw, transitions, tmp_path / "serial"))
    parallel = generation.generate_self_play_examples(
        _request(raw, transitions, tmp_path / "parallel", workers=3)
    )
    assert serial.read_bytes() == parallel.read_bytes()
    assert pd.read_csv(serial)["scenario_id"].tolist() == [3, 1, 2]


def test_interrupted_run_resumes_to_same_output_without_duplicates(
    generation_inputs, tmp_path: Path, fake_generation, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, transitions = generation_inputs((1, 2, 3))
    clean = generation.generate_self_play_examples(_request(raw, transitions, tmp_path / "clean"))
    calls = 0
    def interrupted(scenario_id: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return fake_generation(scenario_id)
    monkeypatch.setattr(generation, "_generate_scenario", interrupted)
    partial_request = _request(raw, transitions, tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generation.generate_self_play_examples(partial_request)
    monkeypatch.setattr(generation, "_generate_scenario", fake_generation)
    resumed = generation.generate_self_play_examples(replace(partial_request, resume=True))
    assert resumed.read_bytes() == clean.read_bytes()
    frame = pd.read_csv(resumed)
    assert frame["scenario_id"].tolist() == [3, 1, 2]
    assert not frame["state_id"].duplicated().any()
