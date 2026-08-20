import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
)
from grid_topology_ai.models.graph_self_play_dataset import (
    GraphSelfPlayDataset,
)
from grid_topology_ai.physics.objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)
from grid_topology_ai.return_contract import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIXED_VAL_CSV = (
    PROJECT_ROOT
    / "data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_val.csv"
)
MIXED_TRAIN_CSV = (
    PROJECT_ROOT
    / "data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_train.csv"
)
RUN_LOCAL_GRAPH_DATA_TESTS = (
    os.environ.get("RUN_LOCAL_GRAPH_DATA_TESTS") == "1"
)


def _csv_provenance() -> dict[str, object]:
    provenance = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    return {
        **provenance,
        "physics_config": json.dumps(
            provenance["physics_config"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _state_metadata() -> np.ndarray:
    return np.array(
        json.dumps(physics_provenance(DEFAULT_PHYSICS_CONFIG))
    )


def _example_row(
    state_path: Path,
    *,
    policy: str,
    target: float,
    solved: bool,
    termination_reason: str,
    outcome_class: str,
) -> dict[str, object]:
    return {
        "state_path": str(state_path),
        "mcts_policy_json": policy,
        "scenario_id": 1,
        "step": 0,
        "state_id": "state-1",
        "selected_action_id": 0,
        "outcome_value_target": target,
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        **_csv_provenance(),
        "solved": solved,
        "done": True,
        "termination_reason": termination_reason,
        "outcome_class": outcome_class,
        "outcome_steps_to_terminal": 1,
        "outcome_value_target_mode": VALUE_TARGET_MODE,
        "outcome_gamma": TERMINAL_UTILITY_GAMMA,
    }


def _statistics_dataset(
    bus_batches: list[np.ndarray],
    branch_batches: list[np.ndarray],
) -> GraphSelfPlayDataset:
    if len(bus_batches) != len(branch_batches):
        raise ValueError("Bus and branch batch counts must match.")

    dataset = object.__new__(GraphSelfPlayDataset)
    dataset.examples = pd.DataFrame(
        {"row": range(len(bus_batches))}
    )
    dataset.num_bus_features = int(bus_batches[0].shape[1])
    dataset.num_branch_features = int(branch_batches[0].shape[1])

    def load_batch(index: int) -> dict[str, np.ndarray]:
        return {
            "bus_features": bus_batches[index],
            "branch_features": branch_batches[index],
        }

    dataset._load_npz_by_index = load_batch
    return dataset


@pytest.mark.skipif(
    not RUN_LOCAL_GRAPH_DATA_TESTS or not MIXED_VAL_CSV.exists(),
    reason=(
        "Local graph dataset integration tests are opt-in. "
        "Set RUN_LOCAL_GRAPH_DATA_TESTS=1 to run them."
    ),
)
def test_graph_self_play_dataset_reads_mixed_val_sample():
    dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_VAL_CSV,
        normalize_features=True,
    )
    sample = dataset[0]

    required_keys = {
        "bus_features",
        "branch_features",
        "edge_index",
        "edge_active_mask",
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
    edge_active_mask = sample["edge_active_mask"]
    action_mask = sample["action_mask"]
    target_policy = sample["target_policy"]
    target_value = sample["target_value"]

    assert bus_features.ndim == 2
    assert branch_features.ndim == 2
    assert edge_index.ndim == 2
    assert edge_active_mask.ndim == 1
    assert action_mask.ndim == 1
    assert target_policy.ndim == 1
    assert edge_index.shape[0] == 2
    assert edge_active_mask.shape[0] == branch_features.shape[0]
    assert action_mask.shape[0] == branch_features.shape[0] + 1
    assert target_policy.shape[0] == action_mask.shape[0]
    assert edge_active_mask.dtype == torch.bool
    assert torch.isfinite(bus_features).all()
    assert torch.isfinite(branch_features).all()
    assert torch.isfinite(target_policy).all()
    assert torch.isfinite(target_value)
    assert -1.0 <= float(target_value.item()) <= 1.0

    invalid_mask = ~action_mask
    assert torch.all(target_policy[invalid_mask] == 0.0)
    policy_sum = float(target_policy.sum().item())
    assert abs(policy_sum - 1.0) < 1e-5 or abs(policy_sum) < 1e-8


@pytest.mark.skipif(
    not RUN_LOCAL_GRAPH_DATA_TESTS
    or not MIXED_TRAIN_CSV.exists()
    or not MIXED_VAL_CSV.exists(),
    reason=(
        "Local graph dataset integration tests are opt-in. "
        "Set RUN_LOCAL_GRAPH_DATA_TESTS=1 to run them."
    ),
)
def test_val_dataset_can_use_train_normalization_stats():
    train_dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_TRAIN_CSV,
        normalize_features=True,
    )
    stats = train_dataset.normalization_state_dict()
    val_dataset = GraphSelfPlayDataset(
        examples_csv=MIXED_VAL_CSV,
        normalize_features=True,
        normalization_stats=stats,
    )

    assert len(val_dataset) > 0
    assert torch.allclose(
        torch.tensor(val_dataset.bus_feature_mean),
        torch.tensor(train_dataset.bus_feature_mean),
    )
    assert torch.allclose(
        torch.tensor(val_dataset.branch_feature_mean),
        torch.tensor(train_dataset.branch_feature_mean),
    )


def test_graph_dataset_uses_mcts_policy_not_selected_action(
    tmp_path: Path,
):
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        action_mask=np.array([True, True, True], dtype=bool),
        metadata_json=_state_metadata(),
    )
    csv_path = tmp_path / "examples.csv"
    pd.DataFrame(
        [
            _example_row(
                state_path,
                policy='{"1": 0.7, "2": 0.3}',
                target=1.0,
                solved=True,
                termination_reason="solved",
                outcome_class="solved",
            )
        ]
    ).to_csv(csv_path, index=False)

    target_policy = GraphSelfPlayDataset(
        csv_path,
        normalize_features=False,
    )[0]["target_policy"]
    assert float(target_policy[0].item()) == pytest.approx(0.0)
    assert float(target_policy[1].item()) == pytest.approx(0.7)
    assert float(target_policy[2].item()) == pytest.approx(0.3)


def test_graph_dataset_derives_edge_mask_only_from_branch_status(
    tmp_path: Path,
):
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        bus_features=np.zeros((3, 2), dtype=np.float32),
        branch_features=np.zeros((3, 4), dtype=np.float32),
        edge_index=np.array(
            [[0, 1, 2], [1, 2, 0]],
            dtype=np.int64,
        ),
        branch_status=np.array(
            [1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        action_mask=np.array(
            [True, True, False, False],
            dtype=bool,
        ),
        metadata_json=_state_metadata(),
    )
    csv_path = tmp_path / "examples.csv"
    pd.DataFrame(
        [
            _example_row(
                state_path,
                policy='{"1": 1.0}',
                target=-1.0,
                solved=False,
                termination_reason="handoff_to_redispatch",
                outcome_class="handoff_to_redispatch",
            )
        ]
    ).to_csv(csv_path, index=False)

    sample = GraphSelfPlayDataset(
        csv_path,
        normalize_features=False,
    )[0]
    assert torch.equal(
        sample["edge_active_mask"],
        torch.tensor([True, False, True]),
    )
    assert torch.equal(
        sample["action_mask"][1:],
        torch.tensor([True, False, False]),
    )


def test_normalization_state_dict_returns_copies(tmp_path: Path):
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        bus_features=np.array(
            [[1, 2, 3], [4, 5, 6]],
            dtype=np.float32,
        ),
        branch_features=np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=np.float32,
        ),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        action_mask=np.array([True, True, True], dtype=bool),
        metadata_json=_state_metadata(),
    )
    csv_path = tmp_path / "examples.csv"
    pd.DataFrame(
        [
            _example_row(
                state_path,
                policy='{"1": 1.0}',
                target=-1.0,
                solved=False,
                termination_reason="handoff_to_redispatch",
                outcome_class="handoff_to_redispatch",
            )
        ]
    ).to_csv(csv_path, index=False)

    dataset = GraphSelfPlayDataset(
        csv_path,
        normalize_features=True,
    )
    stats = dataset.normalization_state_dict()
    stats["bus_feature_mean"][0] = 999.0
    fresh = dataset.normalization_state_dict()
    assert fresh["bus_feature_mean"][0] != 999.0


def test_graph_dataset_rejects_semantic_invalid_handoff_before_state_io(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "examples.csv"
    row = _example_row(
        tmp_path / "does-not-exist.npz",
        policy='{"0": 1.0}',
        target=1.0,
        solved=False,
        termination_reason="handoff_to_redispatch",
        outcome_class="handoff_to_redispatch",
    )
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="outcome_value_target"):
        GraphSelfPlayDataset(csv_path)


def test_streaming_feature_statistics_match_numpy() -> None:
    bus_batches = [
        np.array(
            [[1.0, 10.0], [3.0, 10.0]],
            dtype=np.float32,
        ),
        np.array(
            [[5.0, 10.0], [7.0, 10.0]],
            dtype=np.float32,
        ),
    ]
    branch_batches = [
        np.array(
            [[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]],
            dtype=np.float32,
        ),
        np.array(
            [[6.0, 8.0, 10.0], [8.0, 10.0, 12.0]],
            dtype=np.float32,
        ),
    ]
    all_bus = np.concatenate(bus_batches, axis=0).astype(np.float64)
    all_branch = np.concatenate(branch_batches, axis=0).astype(
        np.float64
    )

    dataset = _statistics_dataset(bus_batches, branch_batches)
    bus_mean, bus_std, branch_mean, branch_std = (
        dataset._compute_feature_statistics()
    )

    expected_bus_std = all_bus.std(axis=0).astype(np.float32)
    expected_bus_std[expected_bus_std < 1e-6] = 1.0
    expected_branch_std = all_branch.std(axis=0).astype(np.float32)
    expected_branch_std[expected_branch_std < 1e-6] = 1.0

    np.testing.assert_allclose(
        bus_mean,
        all_bus.mean(axis=0).astype(np.float32),
    )
    np.testing.assert_allclose(bus_std, expected_bus_std)
    np.testing.assert_allclose(
        branch_mean,
        all_branch.mean(axis=0).astype(np.float32),
    )
    np.testing.assert_allclose(branch_std, expected_branch_std)


def test_streaming_feature_statistics_do_not_concatenate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus_batches = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    ]
    branch_batches = [
        np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        np.array([[9.0, 11.0], [13.0, 15.0]], dtype=np.float32),
    ]
    dataset = _statistics_dataset(bus_batches, branch_batches)

    def fail_on_concatenate(*args, **kwargs):
        pytest.fail("Streaming statistics must not call np.concatenate.")

    monkeypatch.setattr(
        "grid_topology_ai.models.graph_self_play_dataset.np.concatenate",
        fail_on_concatenate,
    )

    dataset._compute_feature_statistics()


def test_streaming_feature_statistics_replace_zero_std() -> None:
    bus_batches = [
        np.array([[3.0, 1.0], [3.0, 3.0]], dtype=np.float32),
        np.array([[3.0, 5.0], [3.0, 7.0]], dtype=np.float32),
    ]
    branch_batches = [
        np.array([[4.0, 2.0], [4.0, 6.0]], dtype=np.float32),
        np.array([[4.0, 10.0], [4.0, 14.0]], dtype=np.float32),
    ]
    dataset = _statistics_dataset(bus_batches, branch_batches)

    _, bus_std, _, branch_std = dataset._compute_feature_statistics()

    assert bus_std[0] == pytest.approx(1.0)
    assert branch_std[0] == pytest.approx(1.0)
    assert bus_std[1] > 1.0
    assert branch_std[1] > 1.0


def test_streaming_feature_statistics_are_stable_for_large_offsets() -> None:
    bus_batches = [
        np.array(
            [[1_000_000.0], [1_000_001.0]],
            dtype=np.float32,
        ),
        np.array(
            [[1_000_002.0], [1_000_003.0]],
            dtype=np.float32,
        ),
    ]
    branch_batches = [
        np.array(
            [[2_000_000.0], [2_000_002.0]],
            dtype=np.float32,
        ),
        np.array(
            [[2_000_004.0], [2_000_006.0]],
            dtype=np.float32,
        ),
    ]
    dataset = _statistics_dataset(bus_batches, branch_batches)

    bus_mean, bus_std, branch_mean, branch_std = (
        dataset._compute_feature_statistics()
    )

    bus_reference = np.array(
        [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0],
        dtype=np.float64,
    )
    branch_reference = np.array(
        [2_000_000.0, 2_000_002.0, 2_000_004.0, 2_000_006.0],
        dtype=np.float64,
    )
    assert bus_mean[0] == pytest.approx(bus_reference.mean())
    assert bus_std[0] == pytest.approx(bus_reference.std())
    assert branch_mean[0] == pytest.approx(branch_reference.mean())
    assert branch_std[0] == pytest.approx(branch_reference.std())


def test_provided_normalization_stats_skip_statistics_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        bus_features=np.zeros((2, 3), dtype=np.float32),
        branch_features=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        branch_status=np.ones(2, dtype=np.float32),
        action_mask=np.array([True, True, True], dtype=bool),
        metadata_json=_state_metadata(),
    )
    csv_path = tmp_path / "examples.csv"
    pd.DataFrame(
        [
            _example_row(
                state_path,
                policy='{"1": 1.0}',
                target=-1.0,
                solved=False,
                termination_reason="handoff_to_redispatch",
                outcome_class="handoff_to_redispatch",
            )
        ]
    ).to_csv(csv_path, index=False)

    stats = {
        "bus_feature_mean": np.array([1.0, 2.0, 3.0]),
        "bus_feature_std": np.array([4.0, 5.0, 6.0]),
        "branch_feature_mean": np.array([7.0, 8.0, 9.0, 10.0]),
        "branch_feature_std": np.array([11.0, 12.0, 13.0, 14.0]),
    }
    original_loader = GraphSelfPlayDataset._load_npz_by_index
    load_count = 0

    def counted_loader(self, index):
        nonlocal load_count
        load_count += 1
        return original_loader(self, index)

    monkeypatch.setattr(
        GraphSelfPlayDataset,
        "_load_npz_by_index",
        counted_loader,
    )

    dataset = GraphSelfPlayDataset(
        csv_path,
        normalize_features=True,
        normalization_stats=stats,
    )

    assert load_count == 1
    np.testing.assert_array_equal(
        dataset.bus_feature_mean,
        stats["bus_feature_mean"],
    )
    np.testing.assert_array_equal(
        dataset.branch_feature_std,
        stats["branch_feature_std"],
    )
