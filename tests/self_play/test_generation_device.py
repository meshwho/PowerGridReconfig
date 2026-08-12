from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.self_play import stages


def test_generation_device_config_accepts_supported_values() -> None:
    assert GenerationConfig.from_mapping({"device": "cpu"}).device == "cpu"
    assert GenerationConfig.from_mapping({"device": "CUDA"}).device == "cuda"


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(False, "cpu"), (True, "cuda")],
)
def test_generation_device_auto_resolves_available_backend(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: cuda_available,
    )

    assert GenerationConfig.from_mapping({"device": " auto "}).device == expected


@pytest.mark.parametrize("device", ["mps", "gpu", "", "cuda:0"])
def test_generation_device_config_rejects_unsupported_values(device: str) -> None:
    with pytest.raises(ValueError, match="generation.device"):
        GenerationConfig(device=device)


def test_run_generate_passes_configured_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions_csv = tmp_path / "transitions.csv"
    transitions_csv.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )
    generated_csv = tmp_path / "generation" / "examples.csv"
    captured = []

    def fake_generate(request):
        captured.append(request)
        generated_csv.parent.mkdir(parents=True, exist_ok=True)
        generated_csv.write_text(
            "outcome_value_target,outcome_gamma\n1.0,1.0\n",
            encoding="utf-8",
        )
        return generated_csv

    monkeypatch.setattr(
        stages,
        "generate_self_play_examples",
        fake_generate,
    )
    monkeypatch.setattr(
        stages,
        "annotate_transitions_csv",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        stages,
        "annotate_examples_csv",
        lambda **_kwargs: None,
    )

    result = stages.run_generate(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_csv,
        scenario_ids=[1],
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "generation",
        config=GenerationConfig(device="cuda"),
        physics_config=DEFAULT_PHYSICS_CONFIG,
        mcts_seed=7,
        action_seed=8,
        iteration=2,
    )

    assert result == generated_csv
    assert len(captured) == 1
    assert captured[0].device == "cuda"
