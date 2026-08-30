from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.generation import GenerationRequest
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


def _request(raw: Path, transitions: Path, output: Path, **kwargs: object) -> GenerationRequest:
    values: dict[str, object] = dict(
        raw_dir=raw, transitions_csv=transitions, output_dir=output,
        checkpoint=None, config=GenerationConfig(max_steps=1), mcts_seed=10,
        action_seed=20, clear_cache_between_scenarios=False,
    )
    values.update(kwargs)
    return GenerationRequest(**values)  # type: ignore[arg-type]


def test_generation_request_is_frozen_and_has_light_controls(tmp_path: Path) -> None:
    request = _request(tmp_path / "raw", tmp_path / "t.csv", tmp_path / "out",
                       workers=3, resume=True)
    assert request.workers == 3
    assert request.resume is True
    with pytest.raises(FrozenInstanceError):
        request.workers = 1  # type: ignore[misc]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_generation_request_rejects_invalid_workers(tmp_path: Path, workers: object) -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _request(tmp_path / "raw", tmp_path / "t.csv", tmp_path / "out", workers=workers)


def test_empty_scenario_selection_is_rejected(generation_inputs, tmp_path: Path) -> None:
    raw, transitions = generation_inputs(())
    request = _request(raw, transitions, tmp_path / "out")
    with pytest.raises(ValueError, match="No self-play scenarios were selected"):
        generation._scenario_ids_from_request(request)


def test_generation_preflight_requires_canonical_gridfm_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    transitions = tmp_path / "transitions.csv"
    transitions.write_text("scenario_id\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="bus_data.parquet"):
        generation._preflight_generation_inputs(_request(raw, transitions, tmp_path / "out"))


def test_episode_uses_operational_action_mask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evidence = terminal_evidence("solved")

    class Env:
        mask_calls = 0
        def __init__(self, **kwargs):
            self.done = False; self.solved = False; self.current_state = object()
            self.termination_reason = None; self.terminal_outcome_evidence = None
        def reset(self, scenario_id): pass
        def operational_action_mask(self):
            type(self).mask_calls += 1
            return [False, True]
        def step(self, action):
            self.done = self.solved = True
            self.termination_reason = TerminationReason.SOLVED
            self.terminal_outcome_evidence = evidence
            return SimpleNamespace(reward=1.0, done=True)

    planner = SimpleNamespace(
        reset_rng=lambda seed: None,
        search_from_env=lambda env: SimpleNamespace(
            best_action_id=1, best_branch_id=10, policy={1: 1.0}, visit_counts={1: 2},
            root=SimpleNamespace(
                actions_by_id={1: SimpleNamespace(branch_id=10)},
                action_scores={},
                neural_value=None,
            ),
        ),
    )
    request = _request(tmp_path / "raw", tmp_path / "t.csv", tmp_path / "out")
    generation._WORKER_RUNTIME = dict(
        request=request, backend=SimpleNamespace(), action_space=SimpleNamespace(),
        evaluator=None, planner=planner, adapter=object(), reward_fn=object(),
    )
    monkeypatch.setattr(generation, "TopologySwitchingEnv", Env)
    result = generation._generate_scenario(7)
    assert Env.mask_calls == 1
    assert result.pending_examples[0]["action_mask"] == [False, True]
    assert result.pending_examples[0]["selected_action_id"] in result.pending_examples[0]["mcts_policy"]
