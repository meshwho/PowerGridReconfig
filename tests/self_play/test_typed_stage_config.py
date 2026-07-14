from __future__ import annotations

from pathlib import Path

from grid_topology_ai.config import (
    EvaluationConfig,
    GenerationConfig,
    TrainingConfig,
)
from grid_topology_ai.self_play.artifacts import save_json
from scripts.self_play import run_iteration


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_run_generate_uses_generation_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path | None = None,
    ) -> None:
        captured.append(command)
        output_dir = Path(_value_after(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "examples.csv").write_text(
            "scenario_id,outcome_value_target\n1,0.5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(run_iteration, "run_command", fake_run_command)
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text("scenario_id\n1\n", encoding="utf-8")
    config = GenerationConfig(
        simulations=17,
        depth=2,
        max_steps=3,
        top_k=11,
        gamma=0.91,
        c_puct=1.7,
        prior_exponent=0.6,
        selection_temperature=0.25,
        use_root_noise=False,
        use_continuation_gate=False,
        pf_alg=2,
        stop_policy="solved_only",
        terminal_unsolved_penalty=321.0,
        terminal_handoff_penalty=123.0,
        terminal_failure_penalty=777.0,
        terminal_penalty_weight=0.2,
    )

    run_iteration.run_generate(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_csv,
        scenario_ids=[1],
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "generated",
        config=config,
        base_seed=100,
        iteration=3,
    )

    command = captured[0]
    assert _value_after(command, "--simulations") == "17"
    assert _value_after(command, "--depth") == "2"
    assert _value_after(command, "--max-steps") == "3"
    assert _value_after(command, "--top-k") == "11"
    assert _value_after(command, "--gamma") == "0.91"
    assert _value_after(command, "--c-puct") == "1.7"
    assert _value_after(command, "--prior-exponent") == "0.6"
    assert _value_after(command, "--selection-temperature") == "0.25"
    assert _value_after(command, "--seed") == "103"
    assert _value_after(command, "--pf-alg") == "2"
    assert _value_after(command, "--terminal-unsolved-penalty") == "321.0"
    assert _value_after(command, "--terminal-handoff-penalty") == "123.0"
    assert _value_after(command, "--terminal-failure-penalty") == "777.0"
    assert _value_after(command, "--terminal-penalty-weight") == "0.2"
    assert _value_after(command, "--stop-policy") == "solved_only"
    assert "--clear-cache-between-scenarios" in command
    assert "--use-root-noise" not in command
    assert "--use-continuation-gate" not in command


def test_run_generate_adds_enabled_boolean_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path | None = None,
    ) -> None:
        captured.append(command)
        output_dir = Path(_value_after(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "examples.csv").write_text(
            "scenario_id,outcome_value_target\n1,0.5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(run_iteration, "run_command", fake_run_command)
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text("scenario_id\n1\n", encoding="utf-8")

    run_iteration.run_generate(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_csv,
        scenario_ids=[1],
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "generated",
        config=GenerationConfig(
            use_root_noise=True,
            use_continuation_gate=True,
        ),
        base_seed=1,
        iteration=1,
    )

    command = captured[0]
    assert "--use-root-noise" in command
    assert "--use-continuation-gate" in command


def test_run_train_uses_training_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path | None = None,
    ) -> None:
        captured.append(command)
        Path(_value_after(command, "--output")).write_bytes(b"checkpoint")

    monkeypatch.setattr(run_iteration, "run_command", fake_run_command)
    config = TrainingConfig(
        epochs=4,
        batch_size=9,
        learning_rate=0.002,
        value_loss_weight=0.7,
        value_huber_delta=1.5,
        num_workers=2,
        device="cpu",
        model_type="mlp",
        hidden_dim=96,
        num_layers=4,
        dropout=0.2,
        save_multiple_best=True,
        no_tensorboard=True,
    )

    run_iteration.run_train(
        project_root=tmp_path,
        examples_csv=tmp_path / "examples.csv",
        init_checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "train",
        config=config,
        iteration=5,
    )

    command = captured[0]
    assert _value_after(command, "--epochs") == "4"
    assert _value_after(command, "--batch-size") == "9"
    assert _value_after(command, "--lr") == "0.002"
    assert _value_after(command, "--value-loss-weight") == "0.7"
    assert _value_after(command, "--value-huber-delta") == "1.5"
    assert _value_after(command, "--device") == "cpu"
    assert _value_after(command, "--num-workers") == "2"
    assert _value_after(command, "--model-type") == "mlp"
    assert _value_after(command, "--hidden-dim") == "96"
    assert _value_after(command, "--num-layers") == "4"
    assert _value_after(command, "--dropout") == "0.2"
    assert "--save-multiple-best" in command
    assert "--no-tensorboard" in command


def test_run_evaluate_uses_evaluation_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path | None = None,
    ) -> None:
        captured.append(command)
        save_json(
            {"solve_rate": 0.8},
            Path(_value_after(command, "--output-json")),
        )

    monkeypatch.setattr(run_iteration, "run_command", fake_run_command)
    config = EvaluationConfig(
        simulations=19,
        depth=3,
        max_steps=6,
        top_k=13,
        gamma=0.88,
        c_puct=1.9,
        prior_exponent=0.7,
        use_continuation_gate=True,
        allow_handoff_with_hard_overloads=True,
        num_workers=3,
        batch_size=7,
        device="cpu",
        output_csv_name="custom_eval.csv",
        output_json_name="custom_metrics.json",
    )

    run_iteration.run_evaluate(
        project_root=tmp_path,
        checkpoint=tmp_path / "candidate.pt",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "eval",
        config=config,
    )

    command = captured[0]
    assert Path(_value_after(command, "--output-csv")).name == "custom_eval.csv"
    assert Path(_value_after(command, "--output-json")).name == "custom_metrics.json"
    assert _value_after(command, "--simulations") == "19"
    assert _value_after(command, "--depth") == "3"
    assert _value_after(command, "--max-steps") == "6"
    assert _value_after(command, "--top-k") == "13"
    assert _value_after(command, "--gamma") == "0.88"
    assert _value_after(command, "--c-puct") == "1.9"
    assert _value_after(command, "--prior-exponent") == "0.7"
    assert _value_after(command, "--num-workers") == "3"
    assert _value_after(command, "--batch-size") == "7"
    assert _value_after(command, "--device") == "cpu"
    assert "--use-continuation-gate" in command
    assert "--allow-handoff-with-hard-overloads" in command
