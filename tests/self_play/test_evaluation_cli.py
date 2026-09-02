from __future__ import annotations

from pathlib import Path

import pytest

import grid_topology_ai.cli as light_cli
import grid_topology_ai.evaluation as evaluation_runtime
from grid_topology_ai.evaluation import EvaluationRequest


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> EvaluationRequest:
    captured: dict[str, EvaluationRequest] = {}

    def fake_evaluate(request: EvaluationRequest) -> dict[str, object]:
        captured["request"] = request
        return {"solve_rate": 1.0}

    monkeypatch.setattr(evaluation_runtime, "evaluate_checkpoint", fake_evaluate)
    args = [
        "evaluate",
        str(tmp_path / "raw"),
        "--transitions",
        str(tmp_path / "transitions.csv"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
    ]
    if extra_args is not None:
        args.extend(extra_args)

    assert light_cli.main(args) == 0
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
            "--simulations", "17",
            "--depth", "2",
            "--max-steps", "3",
            "--top-k", "11",
            "--gamma", "1.0",
            "--c-puct", "1.7",
            "--prior-exponent", "0.6",
            "--workers", "4",
            "--batch-size", "6",
            "--device", "cpu",
        ],
    )
    config = request.config
    assert config.simulations == 17
    assert config.depth == 2
    assert config.max_steps == 3
    assert config.top_k == 11
    assert config.gamma == 1.0
    assert config.c_puct == 1.7
    assert config.prior_exponent == 0.6
    assert config.num_workers == 4
    assert config.batch_size == 6
    assert config.device == "cpu"


def test_cli_selects_one_policy_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_request = _run_cli(monkeypatch, tmp_path / "default")
    constrained_request = _run_cli(
        monkeypatch,
        tmp_path / "constrained",
        ["--policy-mode", "constrained"],
    )
    assert default_request.config.policy_mode == "ungated"
    assert constrained_request.config.policy_mode == "constrained"


def test_cli_passes_request_only_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _run_cli(
        monkeypatch,
        tmp_path,
        [
            "--allow-handoff-with-hard-overloads",
            "--pf-alg", "2",
            "--disable-cache",
            "--min-hard-improvement", "7.0",
            "--min-soft-improvement", "3.0",
            "--min-constraint-visits", "9",
            "--min-constraint-visit-fraction", "0.2",
            "--stop-policy", "solved_only",
            "--clear-caches-every", "8",
            "--use-dc-screening",
            "--dc-top-k", "13",
            "--dc-candidate-pool", "31",
            "--dc-keep-policy-actions", "4",
            "--dc-keep-loading-actions", "5",
            "--dc-policy-weight", "0.4",
            "--dc-failure-penalty", "123.0",
            "--dc-max-depth", "-1",
            "--limit", "10",
            "--quiet",
        ],
    )

    assert request.config.allow_handoff_with_hard_overloads is True
    assert request.physics_config.pf_alg == 2
    assert request.disable_cache is True
    assert request.min_hard_improvement == 7.0
    assert request.min_soft_improvement == 3.0
    assert request.min_constraint_visits == 9
    assert request.min_constraint_visit_fraction == 0.2
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


def test_cli_rejects_removed_continuation_gate_alias() -> None:
    with pytest.raises(SystemExit) as excinfo:
        light_cli.main(
            [
                "evaluate",
                "raw",
                "--transitions", "transitions.csv",
                "--checkpoint", "checkpoint.pt",
                "--use-continuation-gate",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_help_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        light_cli.main(["evaluate", "--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--checkpoint" in output
    assert "--policy-mode" in output
    assert "--use-continuation-gate" not in output
