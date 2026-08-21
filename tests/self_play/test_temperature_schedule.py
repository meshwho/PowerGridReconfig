from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play.generation import (
    GenerationRequest,
    _select_generation_action,
    selection_temperature_for_step,
)
from scripts.self_play import generate as generation_cli


@pytest.mark.parametrize(
    ("iteration", "step", "expected"),
    [
        (1, 0, 1.0),
        (1, 1, 1.0),
        (5, 0, 1.0),
        (5, 1, 1.0),
        (1, 2, 0.0),
        (5, 2, 0.0),
        (6, 0, 0.0),
    ],
)
def test_temperature_schedule_uses_early_iteration_and_step_cutoffs(
    iteration: int,
    step: int,
    expected: float,
) -> None:
    config = GenerationConfig(
        selection_temperature=1.0,
        temperature_steps=2,
        temperature_iterations=5,
    )

    assert selection_temperature_for_step(
        config,
        iteration=iteration,
        step=step,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "config",
    [
        GenerationConfig(
            selection_temperature=0.0,
            temperature_steps=2,
            temperature_iterations=5,
        ),
        GenerationConfig(
            selection_temperature=1.0,
            temperature_steps=0,
            temperature_iterations=5,
        ),
        GenerationConfig(
            selection_temperature=1.0,
            temperature_steps=2,
            temperature_iterations=0,
        ),
    ],
)
def test_temperature_schedule_can_be_disabled(
    config: GenerationConfig,
) -> None:
    assert selection_temperature_for_step(
        config,
        iteration=1,
        step=0,
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selection_temperature": True},
        {"selection_temperature": -0.1},
        {"selection_temperature": float("nan")},
        {"selection_temperature": float("inf")},
        {"temperature_steps": True},
        {"temperature_steps": 1.5},
        {"temperature_steps": -1},
        {"temperature_iterations": True},
        {"temperature_iterations": 1.5},
        {"temperature_iterations": -1},
    ],
)
def test_generation_config_rejects_invalid_temperature_schedule(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_generation_config_reads_temperature_schedule_from_mapping() -> None:
    config = GenerationConfig.from_mapping(
        {
            "selection_temperature": 0.75,
            "temperature_steps": 3,
            "temperature_iterations": 4,
        }
    )

    assert config.selection_temperature == pytest.approx(0.75)
    assert config.temperature_steps == 3
    assert config.temperature_iterations == 4


@pytest.mark.parametrize("iteration", [0, -1, True, 1.5])
def test_generation_request_rejects_invalid_iteration(
    iteration: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="iteration must be a positive integer"):
        GenerationRequest(
            raw_dir=tmp_path / "raw",
            transitions_csv=tmp_path / "transitions.csv",
            output_dir=tmp_path / "out",
            checkpoint=None,
            config=GenerationConfig(),
            mcts_seed=42,
            action_seed=43,
            clear_cache_between_scenarios=False,
            iteration=iteration,  # type: ignore[arg-type]
        )


def test_effective_temperature_controls_behavior_policy() -> None:
    config = GenerationConfig(
        selection_temperature=1.0,
        temperature_steps=1,
        temperature_iterations=1,
    )
    search_result = SimpleNamespace(
        policy={1: 0.75, 2: 0.25},
        root=SimpleNamespace(
            actions_by_id={
                1: SimpleNamespace(branch_id=101),
                2: SimpleNamespace(branch_id=102),
            }
        ),
    )

    sampled = _select_generation_action(
        search_result=search_result,
        temperature=selection_temperature_for_step(
            config,
            iteration=1,
            step=0,
        ),
        rng=np.random.default_rng(7),
        scenario_id=1,
        step=0,
    )
    greedy = _select_generation_action(
        search_result=search_result,
        temperature=selection_temperature_for_step(
            config,
            iteration=1,
            step=1,
        ),
        rng=np.random.default_rng(7),
        scenario_id=1,
        step=1,
    )

    assert sampled.policy_target == pytest.approx({1: 0.75, 2: 0.25})
    assert greedy.policy_target == {1: 1.0}
    assert greedy.selected_action_id == 1


def test_generation_cli_propagates_temperature_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, GenerationRequest] = {}
    examples_path = tmp_path / "examples.csv"

    def fake_generate(request: GenerationRequest) -> Path:
        captured["request"] = request
        return examples_path

    monkeypatch.setattr(
        generation_cli,
        "generate_self_play_examples",
        fake_generate,
    )

    result = generation_cli.main(
        [
            str(tmp_path / "raw"),
            "--transitions",
            str(tmp_path / "transitions.csv"),
            "--output-dir",
            str(tmp_path / "out"),
            "--selection-temperature",
            "0.8",
            "--temperature-steps",
            "2",
            "--temperature-iterations",
            "6",
            "--iteration",
            "4",
        ]
    )

    assert result == 0
    request = captured["request"]
    assert request.iteration == 4
    assert request.config.selection_temperature == pytest.approx(0.8)
    assert request.config.temperature_steps == 2
    assert request.config.temperature_iterations == 6
