from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.self_play.generation import GenerationRequest
from scripts.self_play import generate as generate_cli


def _capture_request(
    monkeypatch: pytest.MonkeyPatch,
    result_path: Path,
    captured: list[GenerationRequest],
) -> None:
    def fake_generate_self_play_examples(request: GenerationRequest) -> Path:
        captured.append(request)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("", encoding="utf-8")
        return result_path

    monkeypatch.setattr(
        generate_cli,
        "generate_self_play_examples",
        fake_generate_self_play_examples,
    )
    monkeypatch.setattr(
        generate_cli,
        "ensure_outcome_value_targets",
        lambda examples_csv, *, gamma: Path(examples_csv),
    )


def test_cli_builds_generation_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[GenerationRequest] = []
    result_path = tmp_path / "out" / "examples.csv"
    _capture_request(monkeypatch, result_path, captured)
    raw_dir = tmp_path / "raw"
    transitions_csv = tmp_path / "transitions.csv"
    checkpoint = tmp_path / "best.pt"

    assert generate_cli.main(
        [
            str(raw_dir),
            "--transitions",
            str(transitions_csv),
            "--output-dir",
            str(tmp_path / "out"),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            "123",
            "--device",
            "cuda",
            "--disable-cache",
            "--clear-cache-between-scenarios",
        ]
    ) == 0

    request = captured[0]
    assert request.raw_dir == raw_dir
    assert request.transitions_csv == transitions_csv
    assert request.output_dir == tmp_path / "out"
    assert request.checkpoint == checkpoint
    assert request.seed == 123
    assert request.device == "cuda"
    assert request.enable_cache is False
    assert request.clear_cache_between_scenarios is True


def test_cli_builds_generation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[GenerationRequest] = []
    _capture_request(monkeypatch, tmp_path / "examples.csv", captured)

    assert generate_cli.main(
        [
            str(tmp_path / "raw"),
            "--transitions",
            str(tmp_path / "transitions.csv"),
            "--simulations",
            "17",
            "--depth",
            "2",
            "--max-steps",
            "3",
            "--top-k",
            "11",
            "--gamma",
            "0.91",
            "--c-puct",
            "1.7",
            "--prior-exponent",
            "0.6",
            "--selection-temperature",
            "0.25",
            "--pf-alg",
            "2",
            "--stop-policy",
            "solved_only",
            "--root-dirichlet-alpha",
            "0.4",
            "--root-exploration-fraction",
            "0.35",
            "--min-hard-improvement",
            "40.0",
            "--min-soft-improvement",
            "12.0",
            "--min-gate-visits",
            "7",
            "--min-gate-visit-fraction",
            "0.2",
            "--use-root-noise",
            "--use-continuation-gate",
        ]
    ) == 0

    request = captured[0]
    config = request.config
    assert config.simulations == 17
    assert config.depth == 2
    assert config.max_steps == 3
    assert config.top_k == 11
    assert config.gamma == 0.91
    assert config.c_puct == 1.7
    assert config.prior_exponent == 0.6
    assert config.selection_temperature == 0.25
    assert config.pf_alg == 2
    assert config.stop_policy == "solved_only"
    assert config.use_root_noise is True
    assert config.use_continuation_gate is True
    assert request.root_dirichlet_alpha == 0.4
    assert request.root_exploration_fraction == 0.35
    assert request.min_hard_improvement == 40.0
    assert request.min_soft_improvement == 12.0
    assert request.min_gate_visits == 7
    assert request.min_gate_visit_fraction == 0.2


def test_cli_preserves_boolean_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[GenerationRequest] = []
    _capture_request(monkeypatch, tmp_path / "examples.csv", captured)
    base_args = [
        str(tmp_path / "raw"),
        "--transitions",
        str(tmp_path / "transitions.csv"),
    ]

    assert generate_cli.main(
        [*base_args, "--use-root-noise", "--use-continuation-gate"]
    ) == 0
    assert captured[-1].config.use_root_noise is True
    assert captured[-1].config.use_continuation_gate is True

    assert generate_cli.main(
        [*base_args, "--no-use-root-noise", "--no-use-continuation-gate"]
    ) == 0
    assert captured[-1].config.use_root_noise is False
    assert captured[-1].config.use_continuation_gate is False


def test_cli_help_exposes_no_terminal_penalty_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        generate_cli.build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--transitions" in output
    assert "--use-root-noise" in output
    assert "--no-use-continuation-gate" in output
    assert "terminal-unsolved-penalty" not in output
    assert "terminal-handoff-penalty" not in output
    assert "terminal-failure-penalty" not in output
    assert "terminal-penalty-weight" not in output
