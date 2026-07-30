from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.evaluation import checkpoint as evaluation
from grid_topology_ai.evaluation.checkpoint import EvaluationRequest
from tests.topology_contract_helpers import topology_metadata


class _FakeReward:
    def __init__(
        self,
        *,
        physics_config=None,
        discount_factor: float = 0.95,
    ) -> None:
        self.physics_config = physics_config
        self.discount_factor = float(discount_factor)

    def config_dict(self) -> dict[str, object]:
        return {
            "reward": "fake",
            "discount_factor": self.discount_factor,
        }


class _FakeCache:
    def cache_info(self) -> str:
        return "cache-info"

    def clear_cache(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        evaluation,
        "GridFMReward",
        _FakeReward,
    )
    monkeypatch.setattr(
        evaluation,
        "_load_checkpoint_topology_action_provenance",
        lambda checkpoint_path: topology_metadata(),
    )
    evaluation._WORKER_CONTEXT = None
    yield
    evaluation._WORKER_CONTEXT = None


def _request(
    tmp_path: Path,
    *,
    scenario_ids: tuple[int, ...],
) -> EvaluationRequest:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for file_name in (
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
    ):
        (raw_dir / file_name).write_bytes(
            file_name.encode("utf-8")
        )
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
        config=EvaluationConfig(
            batch_size=10,
            use_continuation_gate=False,
        ),
        scenario_ids=scenario_ids,
    )


def test_evaluation_uses_explicit_scenario_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_sequential(**kwargs: object):
        captured["scenario_batches"] = kwargs["scenario_batches"]
        evaluation._WORKER_CONTEXT = {
            "backend": _FakeCache(),
            "action_space": _FakeCache(),
            "evaluator": _FakeCache(),
        }
        return [
            {"scenario_id": 3},
            {"scenario_id": 1},
        ], []

    monkeypatch.setattr(evaluation, "run_sequential", fake_sequential)
    monkeypatch.setattr(
        evaluation,
        "_prepare_results_frame",
        lambda rows, transitions_path: pd.DataFrame(rows),
    )
    monkeypatch.setattr(
        evaluation,
        "build_policy_comparison_metrics",
        lambda **kwargs: {"evaluated_scenarios": len(kwargs["df"])},
    )
    monkeypatch.setattr(
        evaluation,
        "_print_mode_summaries",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluation,
        "print_policy_comparison_summary",
        lambda *args, **kwargs: None,
    )

    evaluation.evaluate_checkpoint(
        _request(tmp_path, scenario_ids=(3, 1))
    )

    assert captured["scenario_batches"] == [[3, 1]]


def test_explicit_scenario_ids_must_exist_in_transitions(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, scenario_ids=(1, 4))

    with pytest.raises(ValueError, match="missing from"):
        evaluation.evaluate_checkpoint(request)
