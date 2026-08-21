from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.self_play.generation import GenerationRequest


def test_generation_device_config_accepts_supported_values() -> None:
    assert GenerationConfig.from_mapping({"device": "cpu"}).device == "cpu"
    assert GenerationConfig.from_mapping({"device": "CUDA"}).device == "cuda"


@pytest.mark.parametrize(("cuda_available", "expected"), [(False, "cpu"), (True, "cuda")])
def test_generation_device_auto_resolves_available_backend(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert GenerationConfig.from_mapping({"device": " auto "}).device == expected


@pytest.mark.parametrize("device", ["mps", "gpu", "", "cuda:0"])
def test_generation_device_config_rejects_unsupported_values(device: str) -> None:
    with pytest.raises(ValueError, match="generation.device"):
        GenerationConfig(device=device)


def test_generation_request_keeps_runtime_device(tmp_path: Path) -> None:
    request = GenerationRequest(
        raw_dir=tmp_path / "raw", transitions_csv=tmp_path / "transitions.csv",
        output_dir=tmp_path / "out", checkpoint=None,
        config=GenerationConfig(device="cuda"), mcts_seed=7, action_seed=8,
        clear_cache_between_scenarios=False, device="cuda",
    )
    assert request.device == "cuda"
