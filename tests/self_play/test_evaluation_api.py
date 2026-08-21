from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config import EvaluationConfig
import grid_topology_ai.evaluation as evaluation
from grid_topology_ai.evaluation import EvaluationRequest


class _FakeCache:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear_cache(self) -> None:
        self.clear_count += 1


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    transitions = tmp_path / "transitions.csv"
    pd.DataFrame({"scenario_id": [3, 1, 2, 1]}).to_csv(
        transitions,
        index=False,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return raw_dir, transitions, checkpoint


def _request(
    tmp_path: Path,
    *,
    scenario_ids: tuple[int, ...] | None = None,
    limit: int | None = None,
) -> EvaluationRequest:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path)
    return EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=EvaluationConfig(policy_mode="ungated"),
        scenario_ids=scenario_ids,
        limit=limit,
    )


def test_evaluation_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.raw_dir = tmp_path  # type: ignore[misc]

    assert not hasattr(request, "__dict__")


def test_evaluation_request_normalizes_explicit_scenario_ids(tmp_path: Path) -> None:
    request = _request(tmp_path, scenario_ids=(3, 1))
    assert request.scenario_ids == (3, 1)


def test_evaluation_request_rejects_duplicate_or_empty_scenario_ids(
    tmp_path: Path,
) -> None:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path)

    for scenario_ids in ((), (1, 1)):
        with pytest.raises(ValueError, match="scenario_ids"):
            EvaluationRequest(
                raw_dir=raw_dir,
                transitions_csv=transitions,
                checkpoint=checkpoint,
                config=EvaluationConfig(),
                scenario_ids=scenario_ids,
            )


def test_evaluation_request_rejects_limit_with_explicit_scenarios(
    tmp_path: Path,
) -> None:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="scenario_ids and limit"):
        EvaluationRequest(
            raw_dir=raw_dir,
            transitions_csv=transitions,
            checkpoint=checkpoint,
            config=EvaluationConfig(),
            scenario_ids=(1,),
            limit=1,
        )


def test_load_scenario_ids_is_sorted_and_applies_limit(tmp_path: Path) -> None:
    _, transitions, _ = _write_inputs(tmp_path)

    assert evaluation.load_scenario_ids(transitions, limit=None) == [1, 2, 3]
    assert evaluation.load_scenario_ids(transitions, limit=2) == [1, 2]


def test_chunk_list_preserves_order() -> None:
    assert evaluation.chunk_list([1, 2, 3, 4, 5], batch_size=2) == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_run_scenario_batch_clears_worker_caches_per_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    clears: list[None] = []

    def fake_episode(scenario_id: int) -> dict[str, int]:
        calls.append(int(scenario_id))
        return {"scenario_id": int(scenario_id)}

    monkeypatch.setattr(
        evaluation,
        "run_episode_from_worker_context",
        fake_episode,
    )
    monkeypatch.setattr(
        evaluation,
        "clear_worker_caches_if_needed",
        lambda: clears.append(None),
    )

    rows = evaluation.run_scenario_batch([3, 1, 2])

    assert [row["scenario_id"] for row in rows] == [3, 1, 2]
    assert calls == [3, 1, 2]
    assert len(clears) == 3


def test_release_worker_context_clears_global_and_caches() -> None:
    backend = _FakeCache()
    action_space = _FakeCache()
    evaluator = _FakeCache()
    evaluation._WORKER_CONTEXT = {
        "backend": backend,
        "action_space": action_space,
        "evaluator": evaluator,
        "planner": object(),
    }

    evaluation._release_worker_context()

    assert evaluation._WORKER_CONTEXT is None
    assert backend.clear_count == 1
    assert action_space.clear_count == 1
    assert evaluator.clear_count == 1


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("raw", "Raw directory"),
        ("transitions", "Transitions CSV"),
        ("checkpoint", "Checkpoint"),
    ],
)
def test_missing_evaluation_input_raises(
    tmp_path: Path,
    missing: str,
    message: str,
) -> None:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path)
    if missing == "raw":
        raw_dir = tmp_path / "missing-raw"
    elif missing == "transitions":
        transitions = tmp_path / "missing.csv"
    else:
        checkpoint = tmp_path / "missing.pt"

    request = EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=EvaluationConfig(),
    )

    with pytest.raises(FileNotFoundError, match=message):
        evaluation.evaluate_checkpoint(request)
