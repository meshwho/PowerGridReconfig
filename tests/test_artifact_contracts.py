import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    physics_provenance,
    require_checkpoint_contracts,
    require_physics_provenance,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def _checkpoint() -> dict[str, object]:
    return {
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
        **physics_provenance(DEFAULT_PHYSICS_CONFIG),
    }


def test_current_checkpoint_contract_is_accepted() -> None:
    require_checkpoint_contracts(_checkpoint(), source="test checkpoint")


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_contract_version",
        "physical_objective_schema_version",
        "outcome_value_target_contract_version",
        "physics_config_contract_version",
        "physics_config",
        "physics_config_fingerprint",
    ],
)
def test_legacy_or_missing_checkpoint_contract_is_rejected(field: str) -> None:
    payload = _checkpoint()
    payload.pop(field)
    with pytest.raises(ValueError, match="legacy artifacts cannot be upgraded safely"):
        require_checkpoint_contracts(payload, source="legacy checkpoint")


def test_physics_fingerprint_mismatch_is_rejected() -> None:
    payload = _checkpoint()
    payload["physics_config_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        require_checkpoint_contracts(payload, source="damaged checkpoint")


def test_expected_physics_config_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="PhysicsConfig mismatch"):
        require_physics_provenance(
            _checkpoint(),
            source="checkpoint",
            expected_physics_config=PhysicsConfig(overload_limit_percent=115.0),
        )


def test_conflicting_legacy_pf_alg_is_rejected() -> None:
    payload = _checkpoint()
    payload["pf_alg"] = 1

    with pytest.raises(ValueError, match="conflicts with PhysicsConfig"):
        require_checkpoint_contracts(payload, source="damaged checkpoint")
