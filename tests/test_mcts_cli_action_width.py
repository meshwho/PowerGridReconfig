from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import argparse
import pytest

from grid_topology_ai.evaluation.checkpoint import EvaluationRequest
from scripts.evaluation import evaluate_checkpoint as evaluation_cli
from scripts.planning import run_mcts, run_mcts_episode


ParserFactory = Callable[[], argparse.ArgumentParser]


@pytest.mark.parametrize(
    ("parser_factory", "base_args", "default_top_k"),
    [
        (run_mcts.build_parser, ["raw"], 30),
        (run_mcts_episode.build_parser, ["raw"], 40),
        (
            evaluation_cli.build_parser,
            [
                "raw",
                "--transitions",
                "transitions.csv",
                "--checkpoint",
                "checkpoint.pt",
            ],
            30,
        ),
    ],
)
def test_mcts_clis_expose_progressive_widening_defaults(
    parser_factory: ParserFactory,
    base_args: list[str],
    default_top_k: int,
) -> None:
    parser = parser_factory()
    args = parser.parse_args(base_args)

    assert args.top_k == default_top_k
    assert args.widening_coefficient == pytest.approx(2.0)
    assert args.widening_exponent == pytest.approx(0.5)
    assert args.exploration_quota == 2
    assert args.seed == 42

    help_text = parser.format_help()
    assert "Initial number of switch actions" in help_text
    assert "--widening-coefficient" in help_text
    assert "--widening-exponent" in help_text
    assert "--exploration-quota" in help_text


@pytest.mark.parametrize(
    ("parser_factory", "base_args"),
    [
        (run_mcts.build_parser, ["raw"]),
        (run_mcts_episode.build_parser, ["raw"]),
        (
            evaluation_cli.build_parser,
            [
                "raw",
                "--transitions",
                "transitions.csv",
                "--checkpoint",
                "checkpoint.pt",
            ],
        ),
    ],
)
def test_mcts_clis_parse_action_width_overrides(
    parser_factory: ParserFactory,
    base_args: list[str],
) -> None:
    args = parser_factory().parse_args(
        [
            *base_args,
            "--top-k",
            "11",
            "--widening-coefficient",
            "1.25",
            "--widening-exponent",
            "0.4",
            "--exploration-quota",
            "3",
            "--seed",
            "17",
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
        evaluation_cli,
        "evaluate_checkpoint",
        fake_evaluate,
    )

    result = evaluation_cli.main(
        [
            str(tmp_path / "raw"),
            "--transitions",
            str(tmp_path / "transitions.csv"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--top-k",
            "13",
            "--widening-coefficient",
            "1.75",
            "--widening-exponent",
            "0.35",
            "--exploration-quota",
            "4",
            "--seed",
            "23",
        ]
    )

    assert result == 0

    config = captured["request"].config
    assert config.top_k == 13
    assert config.widening_coefficient == pytest.approx(1.75)
    assert config.widening_exponent == pytest.approx(0.35)
    assert config.exploration_quota == 4
    assert config.random_seed == 23
