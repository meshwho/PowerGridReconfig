from pathlib import Path

import pytest
import torch

from grid_topology_ai.models.graph_self_play_dataset import GraphSelfPlayDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIXED_VAL_CSV = (
    PROJECT_ROOT
    / "data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_val.csv"
)

MIXED_TRAIN_CSV = (
    PROJECT_ROOT
    / "data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_train.csv"
)


@pytest.mark.skipif(
    not MIXED_VAL_CSV.exists(),
    reason="Mixed validation dataset is not available locally.",
)
def test_graph_self_play_dataset_reads_mixed_val_sample():
    dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_VAL_CSV,
        value_scale=10000.0,
        normalize_features=True,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    required_keys = {
        "bus_features",
        "branch_features",
        "edge_index",
        "action_mask",
        "target_policy",
        "target_value",
        "scenario_id",
        "step",
        "state_id",
    }

    assert required_keys.issubset(sample.keys())

    bus_features = sample["bus_features"]
    branch_features = sample["branch_features"]
    edge_index = sample["edge_index"]
    action_mask = sample["action_mask"]
    target_policy = sample["target_policy"]
    target_value = sample["target_value"]

    assert bus_features.ndim == 2
    assert branch_features.ndim == 2
    assert edge_index.ndim == 2
    assert action_mask.ndim == 1
    assert target_policy.ndim == 1

    assert edge_index.shape[0] == 2
    assert action_mask.shape[0] == branch_features.shape[0] + 1
    assert target_policy.shape[0] == action_mask.shape[0]

    assert torch.isfinite(bus_features).all()
    assert torch.isfinite(branch_features).all()
    assert torch.isfinite(target_policy).all()
    assert torch.isfinite(target_value)

    assert -1.0 <= float(target_value.item()) <= 1.0

    # Target policy must not assign probability to invalid actions.
    invalid_mask = ~action_mask
    assert torch.all(target_policy[invalid_mask] == 0.0)

    # Target policy should either be normalized or completely empty.
    policy_sum = float(target_policy.sum().item())
    assert abs(policy_sum - 1.0) < 1e-5 or abs(policy_sum) < 1e-8


@pytest.mark.skipif(
    not MIXED_TRAIN_CSV.exists() or not MIXED_VAL_CSV.exists(),
    reason="Mixed train/validation datasets are not available locally.",
)
def test_val_dataset_can_use_train_normalization_stats():
    train_dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_TRAIN_CSV,
        value_scale=10000.0,
        normalize_features=True,
    )

    # This test assumes you implemented either get_normalization_stats()
    # or normalization_state_dict().
    if hasattr(train_dataset, "get_normalization_stats"):
        stats = train_dataset.get_normalization_stats()
    else:
        stats = train_dataset.normalization_state_dict()

    val_dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_VAL_CSV,
        value_scale=10000.0,
        normalize_features=True,
        normalization_stats=stats,
    )

    assert len(val_dataset) > 0

    assert torch.tensor(val_dataset.bus_feature_mean).shape == torch.tensor(
        train_dataset.bus_feature_mean
    ).shape

    assert torch.tensor(val_dataset.branch_feature_mean).shape == torch.tensor(
        train_dataset.branch_feature_mean
    ).shape

    assert torch.allclose(
        torch.tensor(val_dataset.bus_feature_mean),
        torch.tensor(train_dataset.bus_feature_mean),
    )

    assert torch.allclose(
        torch.tensor(val_dataset.branch_feature_mean),
        torch.tensor(train_dataset.branch_feature_mean),
    )
