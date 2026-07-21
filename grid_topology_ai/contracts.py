from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid_topology_ai.config.physics import PhysicsConfig


# Version 3 binds the stored policy target to the behavior policy that actually
# produced selected_action_id. Version 2 artifacts may contain gate overrides.
OUTCOME_VALUE_TARGET_CONTRACT_VERSION = 3
# Version 4 adds paired ungated/constrained mode metrics and comparison deltas.
EVALUATION_METRICS_CONTRACT_VERSION = 4
CHECKPOINT_CONTRACT_VERSION = 4
REPLAY_BUFFER_SCHEMA_VERSION = 3
PHYSICS_CONFIG_CONTRACT_VERSION = 1


def require_exact_contract_version(
    value: object,
    *,
    expected: int,
    name: str,
    source: str,
    regeneration_command: str,
) -> None:
    if isinstance(value, bool):
        observed: int | None = None
    elif isinstance(value, Integral):
        observed = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        observed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        observed = int(text) if text.isdigit() else None
    else:
        observed = None

    if observed != int(expected):
        rendered = "missing" if value is None else repr(value)
        raise ValueError(
            f"Incompatible {name} for {source}: expected version {expected}, "
            f"observed {rendered}. The artifact semantics changed and "
            f"legacy artifacts cannot be upgraded safely. Regenerate them with: "
            f"{regeneration_command}"
        )


def physics_provenance(
    physics_config: "PhysicsConfig",
) -> dict[str, object]:
    """Build the canonical physics provenance stored in every artifact."""

    return {
        "physics_config_contract_version": PHYSICS_CONFIG_CONTRACT_VERSION,
        "physics_config": physics_config.to_dict(),
        "physics_config_fingerprint": physics_config.fingerprint(),
    }


def require_physics_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    """Validate self-contained physics provenance and optional compatibility."""

    from grid_topology_ai.config.physics import PhysicsConfig

    require_exact_contract_version(
        payload.get("physics_config_contract_version"),
        expected=PHYSICS_CONFIG_CONTRACT_VERSION,
        name="physics-config contract",
        source=source,
        regeneration_command=(
            "regenerate the artifact with the configured PhysicsConfig"
        ),
    )

    missing_fields = [
        field
        for field in ("physics_config", "physics_config_fingerprint")
        if payload.get(field) is None
    ]
    if missing_fields:
        raise ValueError(
            f"Incomplete physics provenance for {source}: missing "
            f"{missing_fields}; legacy artifacts cannot be upgraded safely. "
            "Regenerate the artifact with the configured PhysicsConfig."
        )

    raw_config = payload.get("physics_config")
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid physics_config JSON for {source}."
            ) from exc
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            f"Missing or invalid physics_config for {source}: expected an object."
        )

    try:
        observed_config = PhysicsConfig.from_mapping(raw_config)
    except ValueError as exc:
        raise ValueError(
            f"Invalid physics_config for {source}: {exc}"
        ) from exc

    observed_fingerprint = payload.get("physics_config_fingerprint")
    canonical_fingerprint = observed_config.fingerprint()
    if observed_fingerprint != canonical_fingerprint:
        rendered = (
            "missing"
            if observed_fingerprint is None
            else repr(observed_fingerprint)
        )
        raise ValueError(
            f"PhysicsConfig fingerprint mismatch for {source}: expected "
            f"{canonical_fingerprint}, observed {rendered}."
        )

    legacy_pf_alg = payload.get("pf_alg")
    if legacy_pf_alg is not None:
        if isinstance(legacy_pf_alg, bool):
            parsed_pf_alg: int | None = None
        elif isinstance(legacy_pf_alg, Integral):
            parsed_pf_alg = int(legacy_pf_alg)
        elif isinstance(legacy_pf_alg, Real) and float(legacy_pf_alg).is_integer():
            parsed_pf_alg = int(legacy_pf_alg)
        elif isinstance(legacy_pf_alg, str) and legacy_pf_alg.strip().isdigit():
            parsed_pf_alg = int(legacy_pf_alg.strip())
        else:
            parsed_pf_alg = None
        if parsed_pf_alg != observed_config.pf_alg:
            raise ValueError(
                f"PF_ALG conflicts with PhysicsConfig for {source}: observed "
                f"PF_ALG={legacy_pf_alg!r}, physics.pf_alg="
                f"{observed_config.pf_alg}."
            )

    if (
        expected_physics_config is not None
        and canonical_fingerprint != expected_physics_config.fingerprint()
    ):
        raise ValueError(
            f"PhysicsConfig mismatch for {source}: expected fingerprint "
            f"{expected_physics_config.fingerprint()}, observed "
            f"{canonical_fingerprint}. Regenerate the artifact with the configured "
            "PhysicsConfig."
        )

    return observed_config


def require_checkpoint_contracts(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    from grid_topology_ai.physical_objective import (
        PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    )

    require_exact_contract_version(
        payload.get("checkpoint_contract_version"),
        expected=CHECKPOINT_CONTRACT_VERSION,
        name="checkpoint contract",
        source=source,
        regeneration_command=(
            "regenerate self-play examples, then rerun "
            "python -m scripts.self_play.train_graph_baseline"
        ),
    )
    require_exact_contract_version(
        payload.get("physical_objective_schema_version"),
        expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        name="physical-objective contract",
        source=source,
        regeneration_command=(
            "python -m scripts.self_play.generate ... followed by "
            "python -m scripts.self_play.train_graph_baseline ..."
        ),
    )
    require_exact_contract_version(
        payload.get("outcome_value_target_contract_version"),
        expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        name="outcome/value-target contract",
        source=source,
        regeneration_command=(
            "python -m scripts.self_play.generate ... followed by "
            "python -m scripts.self_play.train_graph_baseline ..."
        ),
    )
    return require_physics_provenance(
        payload,
        source=source,
        expected_physics_config=expected_physics_config,
    )
