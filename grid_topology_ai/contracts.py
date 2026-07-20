from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real


OUTCOME_VALUE_TARGET_CONTRACT_VERSION = 2
EVALUATION_METRICS_CONTRACT_VERSION = 3
CHECKPOINT_CONTRACT_VERSION = 3
REPLAY_BUFFER_SCHEMA_VERSION = 2
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
            f"observed {rendered}. The solved/outcome semantics changed and "
            f"legacy artifacts cannot be upgraded safely. Regenerate them with: "
            f"{regeneration_command}"
        )


def require_checkpoint_contracts(
    payload: Mapping[str, object],
    *,
    source: str,
) -> None:
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
