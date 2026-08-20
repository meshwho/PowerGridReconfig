from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grid_topology_ai.contracts import (
    require_graph_batching_checkpoint_contract,
)
from grid_topology_ai.models.graph_policy_value_net_v2 import (
    GraphPolicyValueNetV2,
)
from grid_topology_ai.models.graph_self_play_dataset import (
    GRAPH_BATCHING_CONTRACT_VERSION,
    collate_graph_samples,
)
from scripts.evaluation.evaluate_examples_with_checkpoint import (
    validate_checkpoint_dataset_compatibility,
)
from tests.topology_contract_helpers import (
    TEST_ACTION_SPACE_CONFIG,
    checkpoint_topology_fields,
    test_action_layout,
)


def _sample(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    scenario_id: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    num_edges = int(edge_index.shape[1])
    num_actions = num_edges + 1

    return {
        "bus_features": torch.randn(
            num_nodes,
            4,
            generator=generator,
        ),
        "branch_features": torch.randn(
            num_edges,
            6,
            generator=generator,
        ),
        "edge_index": edge_index.clone(),
        "edge_active_mask": torch.ones(
            num_edges,
            dtype=torch.bool,
        ),
        "action_mask": torch.ones(
            num_actions,
            dtype=torch.bool,
        ),
        "target_policy": torch.full(
            (num_actions,),
            1.0 / num_actions,
        ),
        "target_value": torch.tensor(0.0),
        "scenario_id": scenario_id,
        "step": 0,
        "state_id": f"state-{scenario_id}",
    }


def _variable_samples() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    small = _sample(
        num_nodes=3,
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
        ),
        scenario_id=1,
        seed=11,
    )

    large = _sample(
        num_nodes=5,
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        scenario_id=2,
        seed=22,
    )

    return small, large


def test_collate_packs_variable_graphs_without_node_or_edge_padding() -> None:
    small, large = _variable_samples()

    batch = collate_graph_samples(
        [small, large]
    )

    assert batch["bus_features"].shape == (8, 4)
    assert batch["branch_features"].shape == (6, 6)
    assert batch["edge_index"].shape == (2, 6)
    assert batch["edge_active_mask"].shape == (6,)

    assert batch["node_batch"].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
    assert batch["edge_batch"].tolist() == [
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert batch["node_ptr"].tolist() == [0, 3, 8]
    assert batch["edge_ptr"].tolist() == [0, 2, 6]

    expected_edges = torch.tensor(
        [
            [0, 1, 3, 4, 5, 6],
            [1, 2, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(
        batch["edge_index"],
        expected_edges,
    )

    assert batch["action_mask"].shape == (2, 5)
    assert batch["action_mask"][0].tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert batch["action_mask"][1].tolist() == [
        True,
        True,
        True,
        True,
        True,
    ]

    torch.testing.assert_close(
        batch["target_policy"][0],
        torch.tensor(
            [
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_collate_accepts_one_based_edge_indices() -> None:
    sample = _sample(
        num_nodes=3,
        edge_index=torch.tensor(
            [
                [1, 2],
                [2, 3],
            ],
            dtype=torch.long,
        ),
        scenario_id=3,
        seed=33,
    )

    batch = collate_graph_samples([sample])

    torch.testing.assert_close(
        batch["edge_index"],
        torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
        ),
    )


def test_graph_v2_scores_variable_edge_counts_with_one_model() -> None:
    small, large = _variable_samples()
    batch = collate_graph_samples(
        [small, large]
    )

    torch.manual_seed(7)
    model = GraphPolicyValueNetV2(
        num_bus_features=4,
        num_branch_features=6,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        policy_logits, values = model(
            bus_features=batch["bus_features"],
            branch_features=batch[
                "branch_features"
            ],
            edge_index=batch["edge_index"],
            edge_active_mask=batch[
                "edge_active_mask"
            ],
            action_mask=batch["action_mask"],
            node_batch=batch["node_batch"],
            edge_batch=batch["edge_batch"],
        )

    assert policy_logits.shape == (2, 5)
    assert values.shape == (2,)
    assert torch.isfinite(
        policy_logits[0, :3]
    ).all()
    assert torch.isfinite(
        policy_logits[1]
    ).all()
    assert torch.isfinite(values).all()

    mask_value = torch.finfo(
        policy_logits.dtype
    ).min
    torch.testing.assert_close(
        policy_logits[0, 3:],
        torch.full_like(
            policy_logits[0, 3:],
            mask_value,
        ),
    )


def test_packed_graph_context_is_isolated_between_graphs() -> None:
    small, large = _variable_samples()
    mixed_batch = collate_graph_samples(
        [small, large]
    )
    small_batch = collate_graph_samples(
        [small]
    )

    torch.manual_seed(17)
    model = GraphPolicyValueNetV2(
        num_bus_features=4,
        num_branch_features=6,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        mixed_logits, mixed_values = model(
            bus_features=mixed_batch[
                "bus_features"
            ],
            branch_features=mixed_batch[
                "branch_features"
            ],
            edge_index=mixed_batch[
                "edge_index"
            ],
            edge_active_mask=mixed_batch[
                "edge_active_mask"
            ],
            action_mask=mixed_batch[
                "action_mask"
            ],
            node_batch=mixed_batch[
                "node_batch"
            ],
            edge_batch=mixed_batch[
                "edge_batch"
            ],
        )

        single_logits, single_values = model(
            bus_features=small_batch[
                "bus_features"
            ],
            branch_features=small_batch[
                "branch_features"
            ],
            edge_index=small_batch[
                "edge_index"
            ],
            edge_active_mask=small_batch[
                "edge_active_mask"
            ],
            action_mask=small_batch[
                "action_mask"
            ],
            node_batch=small_batch[
                "node_batch"
            ],
            edge_batch=small_batch[
                "edge_batch"
            ],
        )

    torch.testing.assert_close(
        mixed_logits[0, :3],
        single_logits[0],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        mixed_values[0],
        single_values[0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_graph_v2_checkpoint_compatibility_ignores_reference_cardinality() -> None:
    class Dataset:
        num_bus_features = 4
        num_branch_features = 6
        num_actions = 5
        topology_action_config = (
            TEST_ACTION_SPACE_CONFIG
        )
        action_layout = test_action_layout(
            (10, 11, 12, 13)
        )
        policy_layout = (
            "stop_plus_branch_status_v1"
        )
        action_layout_count = 1

    checkpoint = {
        **checkpoint_topology_fields((0, 1)),
        "num_bus_features": 4,
        "num_branch_features": 6,
        # The checkpoint was trained on a smaller
        # representative graph.
        "num_actions": 3,
    }

    validate_checkpoint_dataset_compatibility(
        checkpoint=checkpoint,
        checkpoint_path=Path("checkpoint.pt"),
        dataset=Dataset(),
    )


def test_graph_v2_checkpoint_requires_variable_graph_contract() -> None:
    legacy_checkpoint = {
        "model_type": "graph_policy_value_net_v2",
    }

    with pytest.raises(
        ValueError,
        match="graph-batching contract",
    ):
        require_graph_batching_checkpoint_contract(
            legacy_checkpoint,
            source="legacy.pt",
        )

    current_checkpoint = {
        "model_type": "graph_policy_value_net_v2",
        "graph_batching_contract_version": (
            GRAPH_BATCHING_CONTRACT_VERSION
        ),
        "topology_cardinality_independent": True,
    }

    require_graph_batching_checkpoint_contract(
        current_checkpoint,
        source="current.pt",
    )

    current_checkpoint[
        "topology_cardinality_independent"
    ] = False

    with pytest.raises(
        ValueError,
        match="topology_cardinality_independent",
    ):
        require_graph_batching_checkpoint_contract(
            current_checkpoint,
            source="invalid.pt",
        )


def test_graph_v2_checkpoint_weights_cross_topology_cardinality(
    tmp_path: Path,
) -> None:
    small, large = _variable_samples()

    torch.manual_seed(29)
    source_model = GraphPolicyValueNetV2(
        num_bus_features=4,
        num_branch_features=6,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    source_model.eval()

    checkpoint_path = tmp_path / "graph_v2.pt"
    torch.save(
        {
            "model_type": (
                "graph_policy_value_net_v2"
            ),
            "graph_batching_contract_version": (
                GRAPH_BATCHING_CONTRACT_VERSION
            ),
            "topology_cardinality_independent": True,
            "num_bus_features": 4,
            "num_branch_features": 6,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            # Reference cardinality from the small graph
            # remains metadata only.
            "num_actions": int(
                small["action_mask"].numel()
            ),
            "model_state_dict": (
                source_model.state_dict()
            ),
        },
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require_graph_batching_checkpoint_contract(
        checkpoint,
        source=str(checkpoint_path),
    )

    restored_model = GraphPolicyValueNetV2(
        num_bus_features=int(
            checkpoint["num_bus_features"]
        ),
        num_branch_features=int(
            checkpoint["num_branch_features"]
        ),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    restored_model.eval()

    batch = collate_graph_samples([large])

    with torch.no_grad():
        policy_logits, values = restored_model(
            bus_features=batch["bus_features"],
            branch_features=batch[
                "branch_features"
            ],
            edge_index=batch["edge_index"],
            edge_active_mask=batch[
                "edge_active_mask"
            ],
            action_mask=batch["action_mask"],
            node_batch=batch["node_batch"],
            edge_batch=batch["edge_batch"],
        )

    assert policy_logits.shape == (1, 5)
    assert values.shape == (1,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()
