from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid_topology_ai.config.physics import PhysicsConfig


# Version 6 trains the value head on continuous final pre-redispatch topology
# quality. Every state in one completed episode receives the same final-state
# utility.
OUTCOME_VALUE_TARGET_CONTRACT_VERSION = 6
# Version 2 makes final pre-redispatch physical-state quality the primary
# topology objective. Redispatch remains separate terminal evidence/diagnostics.
OUTCOME_OBJECTIVE_VERSION = 2
# Version 7 adds canonical pre-redispatch topology quality (J0, Jfinal,
# delta_J, relative improvement, and final topology utility) to evaluation
# rows and aggregated metrics.
EVALUATION_METRICS_CONTRACT_VERSION = 7
# Version 7 requires exact state-feature schema provenance.
CHECKPOINT_CONTRACT_VERSION = 7
# Version 6 requires exact state-feature schema provenance in every row and
# in the replay manifest.
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
        regeneration_command=(
            "regenerate self-play examples and derived artifacts"
        ),
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
    """Require the exact ordered state representation used by this build."""

    from grid_topology_ai.state_schema import (
        BRANCH_FEATURE_COLUMNS,
        BUS_FEATURE_COLUMNS,
        BUS_ID_SEMANTICS,
        EDGE_INDEX_SEMANTICS,
        STATE_FEATURE_SCHEMA_VERSION,
        state_feature_schema_fingerprint,
        state_feature_schema_provenance,
    )

    require_exact_contract_version(
        payload.get("state_feature_schema_version"),
        expected=STATE_FEATURE_SCHEMA_VERSION,
        name="state-feature schema",
        source=source,
        regeneration_command=(
            "regenerate state NPZ files and self-play examples, then retrain "
            "the policy-value checkpoint"
        ),
    )

    expected_fingerprint = state_feature_schema_fingerprint()
    observed_fingerprint = payload.get("state_feature_schema_fingerprint")
    if observed_fingerprint != expected_fingerprint:
        rendered = (
            "missing"
            if observed_fingerprint is None
            else repr(observed_fingerprint)
        )
        raise ValueError(
            f"State-feature schema fingerprint mismatch for {source}: "
            f"expected {expected_fingerprint}, observed {rendered}. Regenerate "
            "state artifacts and retrain the checkpoint."
        )

    if payload.get("edge_index_semantics") != EDGE_INDEX_SEMANTICS:
        raise ValueError(
            f"Incompatible edge_index semantics for {source}: expected "
            f"{EDGE_INDEX_SEMANTICS!r}, observed "
            f"{payload.get('edge_index_semantics')!r}."
        )

    if payload.get("bus_id_semantics") != BUS_ID_SEMANTICS:
        raise ValueError(
            f"Incompatible bus ID semantics for {source}: expected "
            f"{BUS_ID_SEMANTICS!r}, observed "
            f"{payload.get('bus_id_semantics')!r}."
        )

    for field, expected_columns in (
        ("bus_feature_columns", BUS_FEATURE_COLUMNS),
        ("branch_feature_columns", BRANCH_FEATURE_COLUMNS),
    ):
        raw_columns = payload.get(field)
        if raw_columns is None:
            raise ValueError(
                f"Incomplete state-feature schema provenance for {source}: "
                f"missing {field}."
            )
        parsed_columns = _json_value(
            raw_columns,
            name=field,
            source=source,
        )
        if (
            not isinstance(parsed_columns, Sequence)
            or isinstance(parsed_columns, (str, bytes))
            or list(parsed_columns) != list(expected_columns)
        ):
            raise ValueError(
                f"Ordered {field} mismatch for {source}. Regenerate state "
                "artifacts and retrain the checkpoint."
            )

    return state_feature_schema_provenance()


def physics_provenance(
    physics_config: "PhysicsConfig",
) -> dict[str, object]:
    """Build canonical runtime provenance stored in pipeline artifacts."""

    from grid_topology_ai.state_schema import state_feature_schema_provenance

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
    from grid_topology_ai.topology_actions import (
        ActionSpaceConfig,
        action_layout_fingerprint,
        action_layout_from_value,
    )

    require_exact_contract_version(
        payload.get("topology_action_contract_version"),
        expected=TOPOLOGY_ACTION_CONTRACT_VERSION,
        name="topology-action contract",
        source=source,
        regeneration_command=(
            "regenerate the dataset and retrain the policy-value checkpoint"
        ),
    )

    required = (
        "topology_action_config",
        "topology_action_config_fingerprint",
        "action_layout",
        "action_layout_fingerprint",
    )
    missing = [name for name in required if payload.get(name) is None]
    if missing:
        raise ValueError(
            f"Incomplete topology action provenance for {source}: "
            f"missing {missing}."
        )

    raw_config = payload["topology_action_config"]
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid topology_action_config JSON for {source}."
            ) from exc

    observed_config = ActionSpaceConfig.from_contract_mapping(raw_config)
    observed_config_fingerprint = payload[
        "topology_action_config_fingerprint"
    ]
    canonical_config_fingerprint = observed_config.contract_fingerprint()
    if observed_config_fingerprint != canonical_config_fingerprint:
        raise ValueError(
            f"Topology action config fingerprint mismatch for {source}."
        )

    observed_layout = action_layout_from_value(payload["action_layout"])
    canonical_layout_fingerprint = action_layout_fingerprint(observed_layout)
    if (
        payload["action_layout_fingerprint"]
        != canonical_layout_fingerprint
    ):
        raise ValueError(f"Action layout fingerprint mismatch for {source}.")

    if (
        expected_action_space_config is not None
        and canonical_config_fingerprint
        != expected_action_space_config.contract_fingerprint()
    ):
        raise ValueError(f"Topology action config mismatch for {source}.")

    if expected_action_layout is not None:
        expected_layout_fingerprint = action_layout_fingerprint(
            expected_action_layout
        )
        if canonical_layout_fingerprint != expected_layout_fingerprint:
            raise ValueError(f"Action layout mismatch for {source}.")

    return observed_config, observed_layout


def require_physics_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_physics_config: "PhysicsConfig | None" = None,
) -> "PhysicsConfig":
    """Validate self-contained runtime provenance and compatibility."""

    from grid_topology_ai.config.physics import PhysicsConfig

    require_state_feature_schema_provenance(payload, source=source)
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
        raise ValueError(f"Invalid physics_config for {source}: {exc}") from exc

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

        if parsed_pf_alg is None:
            raise ValueError(
                f"PF_ALG conflicts with PhysicsConfig for {source}: expected "
                f"exact integer PF_ALG value, observed PF_ALG={legacy_pf_alg!r}, "
                f"physics.pf_alg={observed_config.pf_alg}."
            )
        if parsed_pf_alg not in {1, 2, 3, 4}:
            raise ValueError(
                f"PF_ALG conflicts with PhysicsConfig for {source}: observed "
                f"PF_ALG={parsed_pf_alg} is invalid; expected one of 1, 2, 3, "
                f"or 4, physics.pf_alg={observed_config.pf_alg}."
            )
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
            f"{canonical_fingerprint}. Regenerate the artifact with the "
            "configured PhysicsConfig."
        )

    return observed_config


def require_graph_batching_checkpoint_contract(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    """Reject legacy fixed-cardinality Graph V2 checkpoints early."""

    model_type = str(
        payload.get("model_type", "")
    ).strip()

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
        regeneration_command=(
            "retrain the Graph V2 checkpoint with "
            "python -m scripts.self_play.train_graph_baseline"
        ),
    )

    if (
        payload.get("topology_cardinality_independent")
        is not True
    ):
        raise ValueError(
            "Graph V2 checkpoint must declare "
            "topology_cardinality_independent=True for "
            f"{source}. Retrain the checkpoint with the "
            "current variable-graph architecture."
        )


def _require_graph_checkpoint_feature_dimensions(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
    from grid_topology_ai.state_schema import (
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
                f"Graph checkpoint {key} mismatch for {source}: expected "
                f"{expected}, observed {int(value)}."
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
                f"Graph checkpoint is missing normalization vector {key} "
                f"for {source}."
            ) from exc
        if size != expected:
            raise ValueError(
                f"Graph checkpoint normalization vector {key} mismatch for "
                f"{source}: expected {expected}, observed {size}."
            )


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
    require_outcome_objective_version(
        payload,
        source=source,
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
    require_graph_batching_checkpoint_contract(
        payload,
        source=source,
    )

    require_topology_action_provenance(payload, source=source)
    physics_config = require_physics_provenance(
        payload,
        source=source,
        expected_physics_config=expected_physics_config,
    )

    dataset_metadata = payload.get("dataset_metadata")
    model_type = str(payload.get("model_type", ""))
    if model_type in {
        "graph_policy_value_net",
        "graph_policy_value_net_v2",
    }:
        if not isinstance(dataset_metadata, Mapping):
            raise ValueError(
                f"Graph checkpoint is missing dataset_metadata: {source}."
            )
        require_state_feature_schema_provenance(
            dataset_metadata,
            source=f"{source} dataset_metadata",
        )
    elif isinstance(dataset_metadata, Mapping):
        require_state_feature_schema_provenance(
            dataset_metadata,
            source=f"{source} dataset_metadata",
        )

    _require_graph_checkpoint_feature_dimensions(payload, source=source)
    return physics_config
