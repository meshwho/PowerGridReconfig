from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from grid_topology_ai.config import EvaluationConfig
import grid_topology_ai.evaluation as evaluation
from grid_topology_ai.evaluation import EvaluationRequest


def _request(
    tmp_path: Path,
    *,
    scenario_ids: tuple[int, ...],
) -> EvaluationRequest:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    transitions = tmp_path / "transitions.csv"
    pd.DataFrame({"scenario_id": [1, 2, 3]}).to_csv(
        transitions,
        index=False,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    return EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=EvaluationConfig(batch_size=10, policy_mode="ungated"),
        scenario_ids=scenario_ids,
    )


def test_evaluation_uses_explicit_scenario_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_sequential(**kwargs: object):
        captured["scenario_batches"] = kwargs["scenario_batches"]
        return [{"scenario_id": 3}, {"scenario_id": 1}], []

    monkeypatch.setattr(
        evaluation,
        "_make_task_config",
        lambda request: {"policy_mode": request.config.policy_mode},
    )
    monkeypatch.setattr(evaluation, "run_sequential", fake_sequential)
    monkeypatch.setattr(
        evaluation,
        "_prepare_results_frame",
        lambda rows, transitions_path: pd.DataFrame(rows),
    )
    monkeypatch.setattr(
        evaluation,
        "build_evaluation_metrics",
        lambda **kwargs: {"evaluated_scenarios": len(kwargs["df"])},
    )
    monkeypatch.setattr(evaluation, "print_summary", lambda *args: None)
    cache = SimpleNamespace(cache_info=lambda: {})
    monkeypatch.setattr(
        evaluation,
        "_require_worker_context",
        lambda: {"backend": cache, "action_space": cache, "evaluator": cache},
    )

    metrics = evaluation.evaluate_checkpoint(
        _request(tmp_path, scenario_ids=(3, 1))
    )

    assert captured["scenario_batches"] == [[3, 1]]
    assert metrics["evaluated_scenarios"] == 2


def test_explicit_scenario_ids_must_exist_in_transitions(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, scenario_ids=(1, 4))

    with pytest.raises(ValueError, match="missing from"):
        evaluation.evaluate_checkpoint(request)
