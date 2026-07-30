from __future__ import annotations

import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    require_checkpoint_contracts,
    require_topology_action_provenance,
    topology_action_provenance,
)
from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)
from grid_topology_ai.topology_actions import (
    STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT,
    ActionSpaceConfig,
    action_layout_fingerprint,
    build_branch_action_slots,
    require_branch_status_policy_layout,
)


def test_topology_action_provenance_roundtrip() -> None:
    config = ActionSpaceConfig(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=25.0,
        closeable_branch_ids=(4, 9),
    )
    layout = build_branch_action_slots((9, 4))
    payload = topology_action_provenance(
        config,
        layout,
    )

    observed_config, observed_layout = (
        require_topology_action_provenance(
            payload,
            source="unit artifact",
        )
    )

    assert observed_config == config
    assert observed_layout == layout
    assert action_layout_fingerprint(
        observed_layout
    ) == payload["action_layout_fingerprint"]
    assert require_branch_status_policy_layout(
        observed_layout
    ) == STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT


def test_action_layout_order_mismatch_is_rejected() -> None:
    config = ActionSpaceConfig()
    payload = topology_action_provenance(
        config,
        build_branch_action_slots((9, 4)),
    )

    with pytest.raises(
        ValueError,
        match="Action layout mismatch",
    ):
        require_topology_action_provenance(
            payload,
            source="unit artifact",
            expected_action_layout=(
                build_branch_action_slots((4, 9))
            ),
        )


def test_cache_setting_does_not_change_action_contract() -> None:
    cached = ActionSpaceConfig(enable_cache=True)
    uncached = ActionSpaceConfig(enable_cache=False)

    assert cached.to_contract_dict() == (
        uncached.to_contract_dict()
    )
    assert cached.contract_fingerprint() == (
        uncached.contract_fingerprint()
    )


def test_checkpoint_rejects_missing_action_layout() -> None:
    payload = {
        "checkpoint_contract_version": (
            CHECKPOINT_CONTRACT_VERSION
        ),
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        **physics_provenance(
            DEFAULT_PHYSICS_CONFIG
        ),
        **topology_action_provenance(
            ActionSpaceConfig(),
            build_branch_action_slots((0,)),
        ),
    }
    payload.pop("action_layout")

    with pytest.raises(
        ValueError,
        match="Incomplete topology action provenance",
    ):
        require_checkpoint_contracts(
            payload,
            source="unit checkpoint",
        )
