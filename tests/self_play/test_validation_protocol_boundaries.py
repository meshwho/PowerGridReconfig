from __future__ import annotations

from pathlib import Path

from grid_topology_ai.training.graph_policy_value import TrainingRequest


def test_validation_protocol_static_boundaries() -> None:
    stages = Path("grid_topology_ai/self_play/stages.py").read_text()
    iteration = Path("grid_topology_ai/self_play/iteration.py").read_text()
    training = Path("grid_topology_ai/training/graph_policy_value.py").read_text()
    checkpoints = Path("grid_topology_ai/training/checkpoints.py").read_text()

    assert "validation_examples_csv=None" not in stages
    assert "split_examples_by_scenario" in iteration
    assert hasattr(TrainingRequest, "seed") or "seed: int = 42" in training
    assert "generator=train_generator" in training
    assert "checkpoint_selection_metric" in checkpoints

    for config in [
        "configs/self_play_loop.yaml",
        "configs/self_play_loop_pilot.yaml",
        "configs/self_play_loop_smoke.yaml",
    ]:
        assert "validation_fraction" in Path(config).read_text()
