from __future__ import annotations

from pathlib import Path

import pytest

from grid_topology_ai.evaluation.checkpoint import EvaluationRequest
from scripts.evaluation import evaluate_checkpoint as cli


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> EvaluationRequest:
    captured: dict[str, EvaluationRequest] = {}

    def fake_evaluate(request: EvaluationRequest) -> dict[str, object]:
        captured["request"] = request
        return {"solve_rate": 1.0}

    monkeypatch.setattr(cli, "evaluate_checkpoint", fake_evaluate)
    args = [
        str(tmp_path / "raw"),
        "--transitions",
        str(tmp_path / "transitions.csv"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
    ]
    if extra_args is not None:
        args.extend(extra_args)

    assert cli.main(args) == 0
    return captured["request"]


def test_cli_creates_input_and_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _run_cli(
        monkeypatch,
        tmp_path,
        [
            "--output-csv",
            str(tmp_path / "eval.csv"),
            "--output-json",
            str(tmp_path / "eval.json"),
        ],
    )

    assert request.raw_dir == tmp_path / "raw"
    assert request.transitions_csv == tmp_path / "transitions.csv"
    assert request.checkpoint == tmp_path / "checkpoint.pt"
    assert request.output_csv == tmp_path / "eval.csv"
    assert request.output_json == tmp_path / "eval.json"


def test_cli_creates_evaluation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _run_cli(
        monkeypatch,
        tmp_path,
        [
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
            "--num-workers",
            "4",
            "--batch-size",
            "6",
            "--device",
            "cpu",
        ],
    )

    assert request.config.simulations == 17
    assert request.config.depth == 2
    assert request.config.max_steps == 3
    assert request.config.top_k == 11
    assert request.config.gamma == 0.91
    assert request.config.c_puct == 1.7
    assert request.config.prior_exponent == 0.6
    assert request.config.num_workers == 4
    assert request.config.batch_size == 6
    assert request.config.device == "cpu"


def test_cli_passes_continuation_gate_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_request = _run_cli(monkeypatch, tmp_path / "default")
    enabled_request = _run_cli(
        monkeypatch,
        tmp_path / "enabled",
        ["--use-continuation-gate"],
    )

    assert default_request.config.use_continuation_gate is False
    assert enabled_request.config.use_continuation_gate is True


def test_cli_passes_request_only_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _run_cli(
        monkeypatch,
        tmp_path,
        [
            "--allow-handoff-with-hard-overloads",
            "--pf-alg",
            "2",
            "--disable-cache",
            "--min-hard-improvement",
            "7.0",
            "--min-soft-improvement",
            "3.0",
            "--min-gate-visits",
            "9",
            "--min-gate-visit-fraction",
            "0.2",
            "--leaf-penalty-weight",
            "0.25",
            "--stop-policy",
            "solved_only",
            "--clear-caches-every",
            "8",
            "--use-dc-screening",
            "--dc-top-k",
            "13",
            "--dc-candidate-pool",
            "31",
            "--dc-keep-policy-actions",
            "4",
            "--dc-keep-loading-actions",
            "5",
            "--dc-policy-weight",
            "0.4",
            "--dc-failure-penalty",
            "123.0",
            "--dc-max-depth",
            "-1",
            "--limit",
            "10",
            "--quiet",
        ],
    )

    assert request.config.allow_handoff_with_hard_overloads is True
    assert request.pf_alg == 2
    assert request.disable_cache is True
    assert request.min_hard_improvement == 7.0
    assert request.min_soft_improvement == 3.0
    assert request.min_gate_visits == 9
    assert request.min_gate_visit_fraction == 0.2
    assert request.leaf_penalty_weight == 0.25
    assert request.stop_policy == "solved_only"
    assert request.clear_caches_every == 8
    assert request.use_dc_screening is True
    assert request.dc_top_k == 13
    assert request.dc_candidate_pool == 31
    assert request.dc_keep_policy_actions == 4
    assert request.dc_keep_loading_actions == 5
    assert request.dc_policy_weight == 0.4
    assert request.dc_failure_penalty == 123.0
    assert request.dc_max_depth == -1
    assert request.limit == 10
    assert request.quiet is True


def test_cli_main_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_cli(monkeypatch, tmp_path)


def test_cli_help_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--help"])

    assert excinfo.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out
