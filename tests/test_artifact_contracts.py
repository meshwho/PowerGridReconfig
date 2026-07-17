import pytest

from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    require_checkpoint_contracts,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION


def _checkpoint() -> dict[str, object]:
    return {
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "physical_objective_schema_version": PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        "outcome_value_target_contract_version": (
            OUTCOME_VALUE_TARGET_CONTRACT_VERSION
        ),
    }


def test_current_checkpoint_contract_is_accepted() -> None:
    require_checkpoint_contracts(_checkpoint(), source="test checkpoint")


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_contract_version",
        "physical_objective_schema_version",
        "outcome_value_target_contract_version",
    ],
)
def test_legacy_or_missing_checkpoint_contract_is_rejected(field: str) -> None:
    payload = _checkpoint()
    payload.pop(field)
    with pytest.raises(ValueError, match="legacy artifacts cannot be upgraded safely"):
        require_checkpoint_contracts(payload, source="legacy checkpoint")
