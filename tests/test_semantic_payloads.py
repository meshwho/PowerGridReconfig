from __future__ import annotations

import pytest

from grid_topology_ai.config import (
    ActionSpaceConfig,
    PhysicsConfig,
    physics_config_payload,
    require_physics_config_payload,
)
from grid_topology_ai.actions import (
    build_branch_action_slots,
    require_topology_action_payload,
    topology_action_payload,
)


def test_physics_config_payload_round_trip_and_expected_match() -> None:
    config = PhysicsConfig(pf_alg=2, hard_overload_limit_percent=125.0)
    payload = physics_config_payload(config)

    assert payload == {"physics_config": config.to_dict()}
    assert require_physics_config_payload(
        payload,
        source="test payload",
        expected_physics_config=config,
    ) == config

    with pytest.raises(ValueError, match="PhysicsConfig mismatch"):
        require_physics_config_payload(
            payload,
            source="test payload",
            expected_physics_config=PhysicsConfig(),
        )


def test_topology_action_payload_round_trip_and_expected_match() -> None:
    config = ActionSpaceConfig(
        min_loading_for_switch_percent=25.0,
        closeable_branch_ids=(7,),
    )
    layout = build_branch_action_slots((3, 7))
    payload = topology_action_payload(config, layout)

    observed_config, observed_layout = require_topology_action_payload(
        payload,
        source="test payload",
        expected_action_space_config=config,
        expected_action_layout=layout,
    )
    assert observed_config == config
    assert observed_layout == layout

    with pytest.raises(ValueError, match="Topology action config mismatch"):
        require_topology_action_payload(
            payload,
            source="test payload",
            expected_action_space_config=ActionSpaceConfig(),
        )

    with pytest.raises(ValueError, match="Action layout mismatch"):
        require_topology_action_payload(
            payload,
            source="test payload",
            expected_action_layout=build_branch_action_slots((3,)),
        )
