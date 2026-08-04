from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from grid_topology_ai.contracts import topology_action_provenance
from grid_topology_ai.topology_actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    ActionSpaceConfig,
    action_layout_fingerprint,
    build_branch_action_slots,
)

TEST_ACTION_SPACE_CONFIG = ActionSpaceConfig()


def test_action_layout(
    branch_ids: Iterable[int] = (0,),
):
    return build_branch_action_slots(
        tuple(int(branch_id) for branch_id in branch_ids)
    )


# This is a fixture helper, not a test function. It is imported by
# test modules, so explicitly opt it out of pytest collection.
test_action_layout.__test__ = False


def topology_metadata(
    branch_ids: Iterable[int] = (0,),
    *,
    action_space_config: ActionSpaceConfig = TEST_ACTION_SPACE_CONFIG,
) -> dict[str, object]:
    return topology_action_provenance(
        action_space_config,
        test_action_layout(branch_ids),
    )


def topology_csv_fields(
    branch_ids: Iterable[int] = (0,),
    *,
    action_space_config: ActionSpaceConfig = TEST_ACTION_SPACE_CONFIG,
) -> dict[str, object]:
    provenance = topology_metadata(
        branch_ids,
        action_space_config=action_space_config,
    )
    return {
        "topology_action_contract_version": int(
            provenance["topology_action_contract_version"]
        ),
        "topology_action_config": json.dumps(
            provenance["topology_action_config"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "topology_action_config_fingerprint": str(
            provenance["topology_action_config_fingerprint"]
        ),
        "action_layout": json.dumps(
            provenance["action_layout"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "action_layout_fingerprint": str(
            provenance["action_layout_fingerprint"]
        ),
    }


def checkpoint_topology_fields(
    branch_ids: Iterable[int] = (0,),
    *,
    action_space_config: ActionSpaceConfig = TEST_ACTION_SPACE_CONFIG,
) -> dict[str, object]:
    return {
        **topology_metadata(
            branch_ids,
            action_space_config=action_space_config,
        ),
        "policy_layout": STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    }


def branch_ids_from_state_path(
    state_path: object,
) -> tuple[int, ...]:
    try:
        path = Path(str(state_path))
    except (TypeError, ValueError):
        return (0,)

    if not path.is_file():
        return (0,)

    try:
        with np.load(path, allow_pickle=False) as data:
            if "branch_ids" in data.files:
                values = np.asarray(data["branch_ids"])
                if values.ndim == 1 and values.size > 0:
                    return tuple(int(value) for value in values.tolist())

            if "branch_features" in data.files:
                branch_features = np.asarray(
                    data["branch_features"]
                )
                if branch_features.ndim >= 1:
                    return tuple(
                        range(int(branch_features.shape[0]))
                    )
    except (OSError, EOFError, ValueError, TypeError):
        return (0,)

    return (0,)


def enrich_state_arrays(
    arrays: Mapping[str, object],
) -> dict[str, object]:
    result = dict(arrays)

    if (
        "branch_features" not in result
        or "metadata_json" not in result
    ):
        return result

    branch_features = np.asarray(
        result["branch_features"]
    )
    if branch_features.ndim < 1:
        return result

    branch_ids = np.asarray(
        result.get(
            "branch_ids",
            np.arange(
                int(branch_features.shape[0]),
                dtype=np.int64,
            ),
        ),
        dtype=np.int64,
    )
    result["branch_ids"] = branch_ids

    raw_metadata = np.asarray(
        result["metadata_json"]
    )
    if raw_metadata.size != 1:
        return result

    try:
        metadata = json.loads(
            str(raw_metadata.item())
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return result

    if not isinstance(metadata, dict):
        return result

    metadata.update(
        topology_metadata(branch_ids.tolist())
    )
    result["metadata_json"] = np.array(
        json.dumps(metadata)
    )
    return result


def fake_dataset_topology_fields(
    num_branches: int,
) -> dict[str, object]:
    layout = test_action_layout(
        range(int(num_branches))
    )
    num_actions = int(num_branches) + 1
    return {
        "topology_action_config": TEST_ACTION_SPACE_CONFIG,
        "action_layout": layout,
        "action_layout_fingerprint": (
            action_layout_fingerprint(layout)
        ),
        "policy_layout": (
            STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT
        ),
        "action_layout_count": 1,
        "reference_num_buses": int(num_branches),
        "reference_num_branches": int(num_branches),
        "reference_num_actions": num_actions,
    }
