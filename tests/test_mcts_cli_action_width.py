from __future__ import annotations

from pathlib import Path

import pytest

import grid_topology_ai.cli as light_cli
import grid_topology_ai.evaluation as evaluation_runtime
from grid_topology_ai.evaluation import EvaluationRequest


def _base_args() -> list[str]:
    return [
        "evaluate",
        "raw",
        "--transitions",
        "transitions.csv",
        "--checkpoint",
        "checkpoint.pt",
    ]


def test_evaluation_cli_exposes_progressive_widening_defaults() -> None:
    parser = light_cli.build_parser()
    args = parser.parse_args(_base_args())

    assert args.top_k == 30
    assert args.widening_coefficient == pytest.approx(2.0)
    assert args.widening_exponent == pytest.approx(0.5)
    assert args.exploration_quota == 2
    assert args.seed == 42

    help_text = parser.format_help()
    assert "evaluate" in help_text


def test_evaluation_cli_parses_action_width_overrides() -> None:
    args = light_cli.build_parser().parse_args(
        [
            *_base_args(),
            "--top-k", "11",
            "--widening-coefficient", "1.25",
            "--widening-exponent", "0.4",
            "--exploration-quota", "3",
            "--seed", "17",
        ]
    )

    assert args.top_k == 11
    assert args.widening_coefficient == pytest.approx(1.25)
    assert args.widening_exponent == pytest.approx(0.4)
    assert args.exploration_quota == 3
    assert args.seed == 17


def test_evaluation_cli_propagates_action_width_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, EvaluationRequest] = {}

    def fake_evaluate(request: EvaluationRequest) -> dict[str, object]:
        captured["request"] = request
        return {"solve_rate": 1.0}

    monkeypatch.setattr(
        evaluation_runtime,
        "evaluate_checkpoint",
        fake_evaluate,
    )

    result = light_cli.main(
        [
            "evaluate",
            str(tmp_path / "raw"),
            "--transitions", str(tmp_path / "transitions.csv"),
            "--checkpoint", str(tmp_path / "checkpoint.pt"),
            "--top-k", "13",
            "--widening-coefficient", "1.75",
            "--widening-exponent", "0.35",
            "--exploration-quota", "4",
            "--seed", "23",
        ]
    )

    assert result == 0
    config = captured["request"].config
    assert config.top_k == 13
    assert config.widening_coefficient == pytest.approx(1.75)
    assert config.widening_exponent == pytest.approx(0.35)
    assert config.exploration_quota == 4
    assert config.random_seed == 23
