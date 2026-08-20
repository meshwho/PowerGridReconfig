from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid_topology_ai.config.physics import PhysicsConfig


# These versions remain only for checkpoint/replay writers that are simplified
# in later Light commits. Current example/state semantics no longer depend on
# them being present in every artifact row.
OUTCOME_VALUE_TARGET_CONTRACT_VERSION = 6
OUTCOME_OBJECTIVE_VERSION = 2
EVALUATION_METRICS_CONTRACT_VERSION = 7
CHECKPOINT_CONTRACT_VERSION = 7
REPLAY_BUFFER_SCHEMA_VERSION = 6
PHYSICS_CONFIG_CONTRACT_VERSION = 1
TOPOLOGY_ACTION_CONTRACT_VERSION = 1


def require_exact_contract_version(
    value: object,
    *,
    expected: int,
    name: str,
    source: str,
    regeneration_command: str,
) -> None:
    """Validate an explicit legacy version without requiring it in Light."""

    del regeneration_command
    if value is None:
        return

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
        raise ValueError(
            f"Incompatible {name} for {source}: expected version "
            f"{expected}, observed {value!r}."
        )


def require_outcome_objective_version(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    require_exact_contract_version(
        payload.get("outcome_objective_version"),
        expected=OUTCOME_OBJECTIVE_VERSION,
        name="outcome-objective contract",
        source=source,
        regeneration_command="",
    )


def _json_value(value: object, *, name: str, source: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {name} JSON for {source}.") from exc


def require_state_feature_schema_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    """Validate explicit state-schema metadata when it is present."""

    from grid_topology_ai.state.schema import (
        BRANCH_FEATURE_COLUMNS,
        BUS_FEATURE_COLUMNS,
        BUS_ID_SEMANTICS,
        EDGE_INDEX_SEMANTICS,
        STATE_FEATURE_SCHEMA_VERSION,
        state_feature_schema_fingerprint,
        state_feature_schema_provenance,
    )

    version = payload.get("state_feature_schema_version")
    fingerprint = payload.get("state_feature_schema_fingerprint")
    edge_semantics = payload.get("edge_index_semantics")
    bus_semantics = payload.get("bus_id_semantics")
    bus_columns = payload.get("bus_feature_columns")
    branch_columns = payload.get("branch_feature_columns")

    fields = (
        version,
        fingerprint,
        edge_semantics,
        bus_semantics,
        bus_columns,
        branch_columns,
    )
    if all(value is None for value in fields):
        return state_feature_schema_provenance()
    if any(value is None for value in fields):
        raise ValueError(
            f"Incomplete state feature metadata for {source}."
        )

    require_exact_contract_version(
        version,
        expected=STATE_FEATURE_SCHEMA_VERSION,
        name="state-feature schema",
        source=source,
        regeneration_command="",
    )
    if fingerprint != state_feature_schema_fingerprint():
        raise ValueError(
            f"State-feature schema identity mismatch for {source}."
        )
    if edge_semantics != EDGE_INDEX_SEMANTICS:
        raise ValueError(
            f"Incompatible edge_index semantics for {source}."
        )
    if bus_semantics != BUS_ID_SEMANTICS:
        raise ValueError(
            f"Incompatible bus ID semantics for {source}."
        )

    for field, raw, expected_columns in (
        ("bus_feature_columns", bus_columns, BUS_FEATURE_COLUMNS),
        ("branch_feature_columns", branch_columns, BRANCH_FEATURE_COLUMNS),
    ):
        parsed = _json_value(raw, name=field, source=source)
        if (
            not isinstance(parsed, Sequence)
            or isinstance(parsed, (str, bytes))
            or list(parsed) != list(expected_columns)
        ):
            raise ValueError(
                f"Ordered {field} mismatch for {source}."
            )

    return state_feature_schema_provenance()


def physics_provenance(
    physics_config: "PhysicsConfig",
) -> dict[str, object]:
    """Legacy replay/checkpoint metadata; example rows no longer use it."""

    from grid_topology_ai.state.schema import state_feature_schema_provenance

    return {
        **state_feature_schema_provenance(),
        "physics_config_contract_version": PHYSICS_CONFIG_CONTRACT_VERSION,
        "physics_config": physics_config.to_dict(),
        "physics_config_fingerprint": physics_config.fingerprint(),
    }


def topology_action_provenance(
    action_space_config: object,
    action_layout: object,
) -> dict[str, object]:
    """Legacy replay/checkpoint metadata; example rows store semantic data."""

    from grid_topology_ai.topology_actions import (
        action_layout_fingerprint,
        action_layout_to_list,
    )

    config_payload = action_space_config.to_contract_dict()
    layout_payload = action_layout_to_list(action_layout)
    return {
        "topology_action_contract_version": TOPOLOGY_ACTION_CONTRACT_VERSION,
        "topology_action_config": config_payload,
        "topology_action_config_fingerprint": (
            action_space_config.contract_fingerprint()
        ),
        "action_layout": layout_payload,
        "action_layout_fingerprint": action_layout_fingerprint(action_layout),
    }


def require_topology_action_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_action_space_config: object | None = None,
    expected_action_layout: object | None = None,
):
    """Validate the actual action config/layout, not provenance metadata."""

    from grid_topology_ai.topology_actions import (
        ActionSpaceConfig,
        action_layout_fingerprint,
        action_layout_from_value,
    )

    if payload.get("topology_action_config") is None:
        raise ValueError(
            f"Missing topology_action_config for {source}."
        )
    if payload.get("action_layout") is None:
        raise ValueError(
            f"Missing action_layout for {source}."
        )

    raw_config = _json_value(
        payload["topology_action_config"],
        name="topology_action_config",
        source=source,
    )
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            f"Invalid topology_action_config for {source}."
        )
    observed_config = ActionSpaceConfig.from_contract_mapping(raw_config)
    observed_layout = action_layout_from_value(payload["action_layout"])

    explicit_config_fingerprint = payload.get(
        "topology_action_config_fingerprint"
    )
    if (
        explicit_config_fingerprint is not None
        and explicit_config_fingerprint
        != observed_config.contract_fingerprint()
    ):
        raise ValueError(
            f"Topology action config identity mismatch for {source}."
        )

    canonical_layout_fingerprint = action_layout_fingerprint(
        observed_layout
    )
    explicit_layout_fingerprint = payload.get(
        "action_layout_fingerprint"
    )
    if (
        explicit_layout_fingerprint is not None
        and explicit_layout_fingerprint
        != canonical_layout_fingerprint
    ):
        raise ValueError(
            f"Action layout identity mismatch for {source}."
        )

    if expected_action_space_config is not None:
        if (
            observed_config.to_contract_dict()
            != expected_action_space_config.to_contract_dict()
        ):
            raise ValueError(
                f"Topology action config mismatch for {source}."
            )

    if (
        expected_action_layout is not None
        and tuple(observed_layout) != tuple(expected_action_layout)
    ):
        raise ValueError(
            f"Action layout mismatch for {source}."
        )

    return observed_config, observed_layout


def require_physics_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    """Validate the actual PhysicsConfig stored by the current pipeline."""

    from grid_topology_ai.config.physics import PhysicsConfig

    raw_config = _json_value(
        payload.get("physics_config"),
        name="physics_config",
        source=source,
    )
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            f"Missing or invalid physics_config for {source}."
        )
    try:
        observed_config = PhysicsConfig.from_mapping(raw_config)
    except ValueError as exc:
        raise ValueError(
            f"Invalid physics_config for {source}: {exc}"
        ) from exc

    explicit_fingerprint = payload.get("physics_config_fingerprint")
    if (
        explicit_fingerprint is not None
        and explicit_fingerprint != observed_config.fingerprint()
    ):
        raise ValueError(
            f"PhysicsConfig identity mismatch for {source}."
        )

    legacy_pf_alg = payload.get("pf_alg")
    if legacy_pf_alg is not None:
        if isinstance(legacy_pf_alg, bool):
            parsed_pf_alg: int | None = None
        elif isinstance(legacy_pf_alg, Integral):
            parsed_pf_alg = int(legacy_pf_alg)
        elif (
            isinstance(legacy_pf_alg, Real)
            and float(legacy_pf_alg).is_integer()
        ):
            parsed_pf_alg = int(legacy_pf_alg)
        elif (
            isinstance(legacy_pf_alg, str)
            and legacy_pf_alg.strip().isdigit()
        ):
            parsed_pf_alg = int(legacy_pf_alg.strip())
        else:
            parsed_pf_alg = None
        if parsed_pf_alg != observed_config.pf_alg:
            raise ValueError(
                f"PF_ALG conflicts with PhysicsConfig for {source}."
            )

    if (
        expected_physics_config is not None
        and observed_config != expected_physics_config
    ):
        raise ValueError(
            f"PhysicsConfig mismatch for {source}."
        )

    return observed_config


def require_graph_batching_checkpoint_contract(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    model_type = str(payload.get("model_type", "")).strip()
    if model_type not in {
        "graph_v2",
        "graph_policy_value_net_v2",
    }:
        return

    from grid_topology_ai.models.graph_self_play_dataset import (
        GRAPH_BATCHING_CONTRACT_VERSION,
    )

    require_exact_contract_version(
        payload.get("graph_batching_contract_version"),
        expected=GRAPH_BATCHING_CONTRACT_VERSION,
        name="graph-batching contract",
        source=source,
        regeneration_command="",
    )

    if payload.get("topology_cardinality_independent") is not True:
        raise ValueError(
            "Graph V2 checkpoint must declare "
            f"topology_cardinality_independent=True for {source}."
        )


def _require_graph_checkpoint_feature_dimensions(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    from grid_topology_ai.state.schema import (
        BRANCH_FEATURE_COLUMNS,
        BUS_FEATURE_COLUMNS,
    )

    model_type = str(payload.get("model_type", ""))
    if model_type not in {
        "graph_policy_value_net",
        "graph_policy_value_net_v2",
    }:
        return

    expected_dimensions = {
        "num_bus_features": len(BUS_FEATURE_COLUMNS),
        "num_branch_features": len(BRANCH_FEATURE_COLUMNS),
    }
    for key, expected in expected_dimensions.items():
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"Graph checkpoint {source} is missing exact integer {key}."
            )
        if int(value) != expected:
            raise ValueError(
                f"Graph checkpoint {key} mismatch for {source}: "
                f"expected {expected}, observed {int(value)}."
            )

    for key, expected in (
        ("bus_feature_mean", len(BUS_FEATURE_COLUMNS)),
        ("bus_feature_std", len(BUS_FEATURE_COLUMNS)),
        ("branch_feature_mean", len(BRANCH_FEATURE_COLUMNS)),
        ("branch_feature_std", len(BRANCH_FEATURE_COLUMNS)),
    ):
        value = payload.get(key)
        try:
            size = len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                f"Graph checkpoint is missing normalization vector "
                f"{key} for {source}."
            ) from exc
        if size != expected:
            raise ValueError(
                f"Graph checkpoint normalization vector {key} mismatch "
                f"for {source}: expected {expected}, observed {size}."
            )


def require_checkpoint_contracts(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    """Validate current checkpoint semantics while legacy versions are optional."""

    from grid_topology_ai.physics.objective import (
        PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
    )

    require_exact_contract_version(
        payload.get("checkpoint_contract_version"),
        expected=CHECKPOINT_CONTRACT_VERSION,
        name="checkpoint contract",
        source=source,
        regeneration_command="",
    )
    require_exact_contract_version(
        payload.get("physical_objective_schema_version"),
        expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
        name="physical-objective contract",
        source=source,
        regeneration_command="",
    )
    require_outcome_objective_version(payload, source=source)
    require_exact_contract_version(
        payload.get("outcome_value_target_contract_version"),
        expected=OUTCOME_VALUE_TARGET_CONTRACT_VERSION,
        name="outcome/value-target contract",
        source=source,
        regeneration_command="",
    )
    require_graph_batching_checkpoint_contract(
        payload,
        source=source,
    )

    require_topology_action_provenance(
        payload,
        source=source,
    )
    physics_config = require_physics_provenance(
        payload,
        source=source,
        expected_physics_config=expected_physics_config,
    )

    dataset_metadata = payload.get("dataset_metadata")
    model_type = str(payload.get("model_type", ""))
    if isinstance(dataset_metadata, Mapping):
        require_state_feature_schema_provenance(
            dataset_metadata,
            source=f"{source} dataset_metadata",
        )
    elif model_type in {
        "graph_policy_value_net",
        "graph_policy_value_net_v2",
    }:
        raise ValueError(
            f"Graph checkpoint is missing dataset_metadata: {source}."
        )

    _require_graph_checkpoint_feature_dimensions(
        payload,
        source=source,
    )
    return physics_config
