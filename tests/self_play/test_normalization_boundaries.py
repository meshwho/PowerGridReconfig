from pathlib import Path


def test_training_pipeline_has_fine_tuning_normalization_boundaries() -> None:
    text = Path("grid_topology_ai/training/graph_policy_value.py").read_text(encoding="utf-8")
    branch = text.split("if request.init_checkpoint is not None:", 1)[1].split("dataset = GraphSelfPlayDataset", 1)[0]
    assert "extract_normalization_stats" in branch
    assert "normalization_stats=checkpoint_normalization_stats" in text
    assert "normalization_stats=None" not in branch
    assert "effective_normalization_stats = dataset.normalization_state_dict()" in text
    assert "normalization_source" in text
    assert "normalization_frozen_from_init_checkpoint" in text
