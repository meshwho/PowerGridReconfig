from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.evaluation import checkpoint as evaluation
from grid_topology_ai.evaluation.checkpoint import EvaluationRequest
from grid_topology_ai.evaluation.metrics import compute_safety_score


class _FakeReward:
    def config_dict(self) -> dict[str, object]:
        return {"reward": "fake"}


class _FakeCache:
    def cache_info(self) -> str:
        return "cache-info"


def _write_inputs(tmp_path: Path, scenario_ids: list[int] | None = None) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    transitions = tmp_path / "transitions.csv"
    ids = [1, 2, 3] if scenario_ids is None else scenario_ids
    pd.DataFrame({"scenario_id": ids}).to_csv(transitions, index=False)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return raw_dir, transitions, checkpoint


def _request(
    tmp_path: Path,
    *,
    config: EvaluationConfig | None = None,
    scenario_ids: list[int] | None = None,
    **kwargs: object,
) -> EvaluationRequest:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path, scenario_ids)
    return EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=config or EvaluationConfig(use_continuation_gate=False),
        **kwargs,
    )


def _row(scenario_id: int, *, solved: bool = True) -> dict[str, object]:
    row = {
        "scenario_id": scenario_id,
        "steps": scenario_id,
        "use_continuation_gate": False,
        "actions": "[]",
        "branches": "[]",
        "rewards": "[]",
        "total_reward": float(scenario_id),
        "discounted_return": float(scenario_id),
        "done": True,
        "solved": solved,
        "termination_reason": "solved" if solved else "max_steps_reached",
        "final_max_loading_percent": 90.0 + scenario_id,
        "final_num_overloaded_branches": 0,
        "final_num_hard_overloaded_branches": 0,
        "final_num_outaged_branches": 0,
    }
    row["safety_score"] = compute_safety_score(row)
    return row


@pytest.fixture(autouse=True)
def fake_task_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "GridFMReward", _FakeReward)
    monkeypatch.setattr(
        evaluation,
        "_require_worker_context",
        lambda: {
            "backend": _FakeCache(),
            "action_space": _FakeCache(),
            "evaluator": _FakeCache(),
        },
    )


def test_evaluation_request_is_frozen_and_slotted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.raw_dir = tmp_path  # type: ignore[misc]

    assert not hasattr(request, "__dict__")


def test_missing_raw_dir_raises(tmp_path: Path) -> None:
    _, transitions, checkpoint = _write_inputs(tmp_path)
    request = EvaluationRequest(
        raw_dir=tmp_path / "missing",
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=EvaluationConfig(use_continuation_gate=False),
    )

    with pytest.raises(FileNotFoundError, match="Raw directory"):
        evaluation.evaluate_checkpoint(request)


def test_missing_transitions_csv_raises(tmp_path: Path) -> None:
    raw_dir, _, checkpoint = _write_inputs(tmp_path)
    request = EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=tmp_path / "missing.csv",
        checkpoint=checkpoint,
        config=EvaluationConfig(use_continuation_gate=False),
    )

    with pytest.raises(FileNotFoundError, match="Transitions CSV"):
        evaluation.evaluate_checkpoint(request)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    raw_dir, transitions, _ = _write_inputs(tmp_path)
    request = EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=tmp_path / "missing.pt",
        config=EvaluationConfig(use_continuation_gate=False),
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        evaluation.evaluate_checkpoint(request)


def test_load_scenario_ids_is_sorted_and_applies_limit(tmp_path: Path) -> None:
    transitions = tmp_path / "transitions.csv"
    pd.DataFrame({"scenario_id": [3, 1, 2, 1]}).to_csv(transitions, index=False)

    assert evaluation.load_scenario_ids(transitions, limit=2) == [1, 2]


def test_chunk_list_preserves_order() -> None:
    assert evaluation.chunk_list([1, 2, 3, 4, 5], batch_size=2) == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_task_config_uses_evaluation_config_and_request_values(tmp_path: Path) -> None:
    config = EvaluationConfig(
        simulations=17,
        depth=2,
        max_steps=3,
        top_k=11,
        gamma=0.91,
        c_puct=1.7,
        prior_exponent=0.6,
        use_continuation_gate=False,
        allow_handoff_with_hard_overloads=True,
        num_workers=4,
        batch_size=6,
        device="cpu",
    )
    request = _request(
        tmp_path,
        config=config,
        pf_alg=2,
        disable_cache=True,
        leaf_penalty_weight=0.25,
        stop_policy="solved_only",
        min_hard_improvement=7.0,
        min_soft_improvement=3.0,
        min_gate_visits=9,
        min_gate_visit_fraction=0.2,
        clear_caches_every=8,
        use_dc_screening=True,
        dc_top_k=13,
        dc_candidate_pool=31,
        dc_keep_policy_actions=4,
        dc_keep_loading_actions=5,
        dc_policy_weight=0.4,
        dc_failure_penalty=123.0,
        dc_max_depth=-1,
    )

    task = evaluation._make_task_config(request)

    assert task["simulations"] == 17
    assert task["depth"] == 2
    assert task["max_steps"] == 3
    assert task["top_k"] == 11
    assert task["gamma"] == 0.91
    assert task["c_puct"] == 1.7
    assert task["prior_exponent"] == 0.6
    assert task["leaf_penalty_weight"] == 0.25
    assert task["stop_policy"] == "solved_only"
    assert task["device"] == "cpu"
    assert task["pf_alg"] == 2
    assert task["disable_cache"] is True
    assert task["use_continuation_gate"] is False
    assert task["allow_handoff_with_hard_overloads"] is True
    assert task["min_hard_improvement"] == 7.0
    assert task["min_soft_improvement"] == 3.0
    assert task["min_gate_visits"] == 9
    assert task["min_gate_visit_fraction"] == 0.2
    assert task["clear_caches_every"] == 8
    assert task["use_dc_screening"] is True
    assert task["dc_top_k"] == 13
    assert task["dc_candidate_pool"] == 31
    assert task["dc_keep_policy_actions"] == 4
    assert task["dc_keep_loading_actions"] == 5
    assert task["dc_policy_weight"] == 0.4
    assert task["dc_failure_penalty"] == 123.0
    assert task["dc_max_depth"] == -1
    assert task["reward_config"] == {"reward": "fake"}


def test_evaluate_checkpoint_uses_sequential_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"sequential": False}

    def fake_sequential(**kwargs: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        called["sequential"] = True
        return [_row(1)], []

    monkeypatch.setattr(evaluation, "run_sequential", fake_sequential)
    monkeypatch.setattr(
        evaluation,
        "run_parallel",
        lambda **kwargs: pytest.fail("parallel runner should not be called"),
    )

    metrics = evaluation.evaluate_checkpoint(_request(tmp_path))

    assert called["sequential"] is True
    assert metrics["evaluated_scenarios"] == 1


def test_evaluate_checkpoint_uses_parallel_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"parallel": False}

    def fake_parallel(**kwargs: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        called["parallel"] = True
        return [_row(1)], []

    config = EvaluationConfig(num_workers=2, use_continuation_gate=False)
    monkeypatch.setattr(
        evaluation,
        "run_sequential",
        lambda **kwargs: pytest.fail("sequential runner should not be called"),
    )
    monkeypatch.setattr(evaluation, "run_parallel", fake_parallel)

    metrics = evaluation.evaluate_checkpoint(_request(tmp_path, config=config))

    assert called["parallel"] is True
    assert metrics["evaluated_scenarios"] == 1


def test_evaluation_sorts_output_rows_by_scenario_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_csv = tmp_path / "out" / "eval.csv"
    request = _request(tmp_path, output_csv=output_csv)
    monkeypatch.setattr(
        evaluation,
        "run_sequential",
        lambda **kwargs: ([_row(3), _row(1), _row(2)], []),
    )

    evaluation.evaluate_checkpoint(request)

    df = pd.read_csv(output_csv)
    assert df["scenario_id"].tolist() == [1, 2, 3]


def test_evaluation_saves_csv_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_csv = tmp_path / "eval.csv"
    output_json = tmp_path / "eval.json"
    request = _request(tmp_path, output_csv=output_csv, output_json=output_json)
    monkeypatch.setattr(evaluation, "run_sequential", lambda **kwargs: ([_row(1)], []))

    metrics = evaluation.evaluate_checkpoint(request)

    assert output_csv.exists()
    assert output_json.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["solve_rate"] == 1.0
    assert metrics["solve_rate"] == 1.0


def test_evaluation_returns_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "run_sequential", lambda **kwargs: ([_row(1)], []))

    metrics = evaluation.evaluate_checkpoint(_request(tmp_path))

    assert metrics["requested_scenarios"] == 3
    assert metrics["solve_count"] == 1


def test_evaluation_rejects_zero_successful_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = [{"ok": False, "scenario_id": 1, "row": None, "traceback": "boom"}]
    monkeypatch.setattr(evaluation, "run_sequential", lambda **kwargs: ([], failed))

    with pytest.raises(RuntimeError, match="No scenarios"):
        evaluation.evaluate_checkpoint(_request(tmp_path))


def test_failed_scenarios_are_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = [
        {"ok": False, "scenario_id": 2, "row": None, "traceback": "boom"},
        {"ok": False, "scenario_id": 3, "row": None, "traceback": "boom"},
    ]
    monkeypatch.setattr(evaluation, "run_sequential", lambda **kwargs: ([_row(1)], failed))

    metrics = evaluation.evaluate_checkpoint(_request(tmp_path))

    assert metrics["failed_scenarios"] == 2


def test_difficulty_metrics_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir, transitions, checkpoint = _write_inputs(tmp_path, [1, 2, 3])
    pd.DataFrame(
        {
            "scenario_id": [1, 2, 3],
            "difficulty_class": ["simple", "medium", "hard"],
        }
    ).to_csv(transitions, index=False)
    request = EvaluationRequest(
        raw_dir=raw_dir,
        transitions_csv=transitions,
        checkpoint=checkpoint,
        config=EvaluationConfig(use_continuation_gate=False),
    )
    monkeypatch.setattr(
        evaluation,
        "run_sequential",
        lambda **kwargs: ([_row(1), _row(2), _row(3)], []),
    )

    metrics = evaluation.evaluate_checkpoint(request)

    assert metrics["count_simple"] == 1
    assert metrics["count_medium"] == 1
    assert metrics["count_hard"] == 1
    assert set(metrics["difficulty_metrics"]) == {"simple", "medium", "hard"}


def test_safety_score_formula_is_unchanged() -> None:
    row = {
        "solved": False,
        "termination_reason": "max_steps_reached",
        "final_max_loading_percent": 130.0,
        "final_num_overloaded_branches": 2,
        "final_num_hard_overloaded_branches": 1,
        "discounted_return": 40.0,
    }

    assert compute_safety_score(row) == -848.0


def test_request_validation_rejects_invalid_pf_alg(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pf_alg"):
        _request(tmp_path, pf_alg=9)


def test_request_validation_rejects_invalid_stop_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stop_policy"):
        _request(tmp_path, stop_policy="sometimes")
