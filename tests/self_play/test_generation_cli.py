from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid_topology_ai.self_play.generation import GenerationRequest
from scripts.self_play import generate as generate_cli


def _capture(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[GenerationRequest]:
    captured: list[GenerationRequest] = []
    def fake(request: GenerationRequest) -> Path:
        captured.append(request)
        return path
    monkeypatch.setattr(generate_cli, "generate_self_play_examples", fake)
    return captured


def _expected_seeds(seed: int, iteration: int) -> tuple[int, int]:
    _, mcts, action = np.random.SeedSequence([seed, iteration]).spawn(3)
    return (
        int(mcts.generate_state(1, dtype=np.uint64)[0]),
        int(action.generate_state(1, dtype=np.uint64)[0]),
    )


def test_cli_builds_current_generation_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, tmp_path / "out" / "examples.csv")
    raw, transitions, checkpoint = tmp_path / "raw", tmp_path / "t.csv", tmp_path / "best.pt"
    assert generate_cli.main([
        str(raw), "--transitions", str(transitions), "--output-dir", str(tmp_path / "out"),
        "--checkpoint", str(checkpoint), "--seed", "123", "--iteration", "4",
        "--device", "cuda", "--disable-cache", "--clear-cache-between-scenarios",
        "--workers", "3", "--resume",
    ]) == 0
    request = captured[0]
    assert (request.mcts_seed, request.action_seed) == _expected_seeds(123, 4)
    assert request.raw_dir == raw and request.transitions_csv == transitions
    assert request.checkpoint == checkpoint and request.iteration == 4
    assert request.device == "cuda" and request.enable_cache is False
    assert request.clear_cache_between_scenarios is True
    assert request.workers == 3 and request.resume is True


def test_cli_builds_generation_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, tmp_path / "examples.csv")
    assert generate_cli.main([
        str(tmp_path / "raw"), "--transitions", str(tmp_path / "t.csv"),
        "--simulations", "17", "--depth", "2", "--max-steps", "3", "--top-k", "11",
        "--gamma", "1.0", "--c-puct", "1.7", "--prior-exponent", "0.6",
        "--selection-temperature", "0.25", "--temperature-steps", "2",
        "--temperature-iterations", "5", "--pf-alg", "2", "--stop-policy", "solved_only",
        "--use-root-noise",
    ]) == 0
    config = captured[0].config
    assert (config.simulations, config.depth, config.max_steps, config.top_k) == (17, 2, 3, 11)
    assert config.gamma == 1.0 and config.c_puct == 1.7 and config.prior_exponent == 0.6
    assert config.selection_temperature == 0.25
    assert (config.temperature_steps, config.temperature_iterations) == (2, 5)
    assert config.pf_alg == 2 and config.stop_policy == "solved_only"
    assert config.use_root_noise is True


def test_cli_help_exposes_light_parallel_resume_controls(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        generate_cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--workers" in output and "--resume" in output
