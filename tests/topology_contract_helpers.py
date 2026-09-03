from __future__ import annotations

import json
from collections.abc import Iterable

from grid_topology_ai.actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    build_branch_action_slots,
    topology_action_payload,
)
from grid_topology_ai.config import ActionSpaceConfig

TEST_ACTION_SPACE_CONFIG = ActionSpaceConfig()


def test_action_layout(branch_ids: Iterable[int] = (0,)):
    return build_branch_action_slots(tuple(int(branch_id) for branch_id in branch_ids))


test_action_layout.__test__ = False


def topology_metadata(
    branch_ids: Iterable[int] = (0,),
    *,
    action_space_config: ActionSpaceConfig = TEST_ACTION_SPACE_CONFIG,
) -> dict[str, object]:
    return topology_action_payload(
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
        "topology_action_config": json.dumps(
            provenance["topology_action_config"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "action_layout": json.dumps(
            provenance["action_layout"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
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
