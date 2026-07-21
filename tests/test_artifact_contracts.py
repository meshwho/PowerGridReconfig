import json

import numpy as np
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
    PHYSICS_CONFIG_CONTRACT_VERSION,
    physics_provenance,
    require_checkpoint_contracts,
    require_exact_contract_version,
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
    observed = require_checkpoint_contracts(
        _checkpoint(),
        source="test checkpoint",
    )

    assert observed == DEFAULT_PHYSICS_CONFIG


def test_physics_provenance_accepts_mapping_and_json_payloads() -> None:
    mapping_payload = physics_provenance(DEFAULT_PHYSICS_CONFIG)
    json_payload = {
        **mapping_payload,
        "physics_config": json.dumps(
            mapping_payload["physics_config"],
            sort_keys=True,
        ),
    }

    assert require_physics_provenance(
        mapping_payload,
        source="mapping artifact",
    ) == DEFAULT_PHYSICS_CONFIG
    assert require_physics_provenance(
        json_payload,
        source="CSV artifact",
    ) == DEFAULT_PHYSICS_CONFIG


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


@pytest.mark.parametrize(
    "raw_config",
    [
        "{broken-json",
        "[]",
        [],
        {"pf_alg": 3, "unknown_setting": 1},
        {"pf_alg": 9},
    ],
)
def test_invalid_embedded_physics_config_is_rejected(
    raw_config: object,
) -> None:
    payload = _checkpoint()
    payload["physics_config"] = raw_config

    with pytest.raises(
        ValueError,
        match="Invalid physics_config|Missing or invalid physics_config",
    ):
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


@pytest.mark.parametrize("legacy", [3, np.int64(3), 3.0, "3", " 3 "])
def test_equivalent_legacy_pf_alg_is_accepted(legacy: object) -> None:
    payload = _checkpoint()
    payload["pf_alg"] = legacy

    assert require_checkpoint_contracts(
        payload,
        source="compatible checkpoint",
    ) == DEFAULT_PHYSICS_CONFIG


@pytest.mark.parametrize("legacy", [True, False, 3.5, "3.0", "x"])
def test_non_exact_legacy_pf_alg_in_artifact_is_rejected(
    legacy: object,
) -> None:
    payload = _checkpoint()
    payload["pf_alg"] = legacy

    with pytest.raises(ValueError, match="conflicts with PhysicsConfig"):
        require_checkpoint_contracts(payload, source="damaged checkpoint")


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        PHYSICS_CONFIG_CONTRACT_VERSION + 1,
        1.5,
        "1.0",
        "invalid",
    ],
)
def test_contract_version_requires_exact_current_integer(value: object) -> None:
    with pytest.raises(ValueError, match="Incompatible physics-config contract"):
        require_exact_contract_version(
            value,
            expected=PHYSICS_CONFIG_CONTRACT_VERSION,
            name="physics-config contract",
            source="test artifact",
            regeneration_command="regenerate",
        )


@pytest.mark.parametrize(
    "value",
    [
        PHYSICS_CONFIG_CONTRACT_VERSION,
        float(PHYSICS_CONFIG_CONTRACT_VERSION),
        str(PHYSICS_CONFIG_CONTRACT_VERSION),
        f" {PHYSICS_CONFIG_CONTRACT_VERSION} ",
        np.int64(PHYSICS_CONFIG_CONTRACT_VERSION),
    ],
)
def test_contract_version_accepts_lossless_storage_representations(
    value: object,
) -> None:
    require_exact_contract_version(
        value,
        expected=PHYSICS_CONFIG_CONTRACT_VERSION,
        name="physics-config contract",
        source="test artifact",
        regeneration_command="regenerate",
    )
