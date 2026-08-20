from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from grid_topology_ai.power_flow_errors import InvalidPhysicalState


@dataclass(frozen=True, slots=True)
class ValidatedStateTopology:
    bus_df: pd.DataFrame
    branch_df: pd.DataFrame
    gen_df: pd.DataFrame
    bus_ids: np.ndarray
    branch_ids: np.ndarray
    branch_status: np.ndarray
    edge_index: np.ndarray


def validate_state_topology(
    *,
    scenario_id: int,
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> ValidatedStateTopology:
    """Validate state identifiers and build contiguous graph indices."""

    context = f"scenario {int(scenario_id)}"
    bus = bus_df.copy()
    branch = branch_df.copy()
    gen = gen_df.copy()

    bus_ids = _unique_integral_ids(
        bus,
        column="bus",
        entity="bus",
        context=context,
    )
    branch_ids = _unique_integral_ids(
        branch,
        column="idx",
        entity="branch",
        context=context,
    )
    generator_ids = _unique_integral_ids(
        gen,
        column="idx",
        entity="generator",
        context=context,
    )

    branch_from_bus = _integral_values(
        branch,
        column="from_bus",
        label="branch from_bus",
        context=context,
    )
    branch_to_bus = _integral_values(
        branch,
        column="to_bus",
        label="branch to_bus",
        context=context,
    )
    generator_bus = _integral_values(
        gen,
        column="bus",
        label="generator bus",
        context=context,
    )

    branch_status = _binary_values(
        branch,
        column="br_status",
        label="branch status",
        context=context,
    )
    generator_status = _binary_values(
        gen,
        column="in_service",
        label="generator status",
        context=context,
    )

    _require_ordered_limits(
        bus,
        id_values=bus_ids,
        entity="bus",
        lower_column="min_vm_pu",
        upper_column="max_vm_pu",
        context=context,
    )
    _require_ordered_limits(
        gen,
        id_values=generator_ids,
        entity="generator",
        lower_column="min_p_mw",
        upper_column="max_p_mw",
        context=context,
    )
    _require_ordered_limits(
        gen,
        id_values=generator_ids,
        entity="generator",
        lower_column="min_q_mvar",
        upper_column="max_q_mvar",
        context=context,
    )
    _require_finite_columns(
        gen,
        columns=("p_mw", "q_mvar"),
        label="generator output",
        context=context,
    )

    known_bus_ids = set(int(value) for value in bus_ids)
    _require_known_branch_endpoints(
        branch_ids=branch_ids,
        from_bus=branch_from_bus,
        to_bus=branch_to_bus,
        known_bus_ids=known_bus_ids,
        context=context,
    )
    _require_known_generator_buses(
        generator_ids=generator_ids,
        generator_bus=generator_bus,
        known_bus_ids=known_bus_ids,
        context=context,
    )

    bus["bus"] = bus_ids
    branch["idx"] = branch_ids
    branch["from_bus"] = branch_from_bus
    branch["to_bus"] = branch_to_bus
    branch["br_status"] = branch_status
    gen["idx"] = generator_ids
    gen["bus"] = generator_bus
    gen["in_service"] = generator_status

    bus = bus.sort_values("bus").reset_index(drop=True)
    branch = branch.sort_values("idx").reset_index(drop=True)
    gen = gen.sort_values("idx").reset_index(drop=True)

    sorted_bus_ids = bus["bus"].to_numpy(dtype=np.int64)
    bus_position = {
        int(bus_id): position
        for position, bus_id in enumerate(sorted_bus_ids)
    }

    from_position = branch["from_bus"].map(bus_position).to_numpy(dtype=np.int64)
    to_position = branch["to_bus"].map(bus_position).to_numpy(dtype=np.int64)
    edge_index = np.vstack((from_position, to_position))

    if edge_index.size:
        if edge_index.min() < 0 or edge_index.max() >= len(sorted_bus_ids):
            raise InvalidPhysicalState(
                f"{context}: edge_index contains an invalid bus position."
            )

    return ValidatedStateTopology(
        bus_df=bus,
        branch_df=branch,
        gen_df=gen,
        bus_ids=sorted_bus_ids,
        branch_ids=branch["idx"].to_numpy(dtype=np.int64),
        branch_status=branch["br_status"].to_numpy(dtype=np.float32),
        edge_index=edge_index,
    )


def _numeric_values(
    frame: pd.DataFrame,
    *,
    column: str,
    label: str,
    context: str,
) -> np.ndarray:
    if column not in frame.columns:
        raise InvalidPhysicalState(
            f"{context}: missing required {label} column {column!r}."
        )

    try:
        values = frame[column].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidPhysicalState(
            f"{context}: {label} must be numeric."
        ) from exc

    if not np.isfinite(values).all():
        raise InvalidPhysicalState(
            f"{context}: {label} contains NaN or infinity."
        )

    return values


def _integral_values(
    frame: pd.DataFrame,
    *,
    column: str,
    label: str,
    context: str,
) -> np.ndarray:
    values = _numeric_values(
        frame,
        column=column,
        label=label,
        context=context,
    )

    if not np.equal(values, np.rint(values)).all():
        raise InvalidPhysicalState(
            f"{context}: {label} must contain integral values."
        )

    int64 = np.iinfo(np.int64)
    if np.any(values < int64.min) or np.any(values > int64.max):
        raise InvalidPhysicalState(
            f"{context}: {label} cannot be represented as int64."
        )

    return values.astype(np.int64)


def _unique_integral_ids(
    frame: pd.DataFrame,
    *,
    column: str,
    entity: str,
    context: str,
) -> np.ndarray:
    ids = _integral_values(
        frame,
        column=column,
        label=f"{entity} IDs",
        context=context,
    )

    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicates = unique_ids[counts > 1]
    if duplicates.size:
        rendered = ", ".join(str(int(value)) for value in duplicates[:10])
        raise InvalidPhysicalState(
            f"{context}: duplicate {entity} IDs: {rendered}."
        )

    return ids


def _binary_values(
    frame: pd.DataFrame,
    *,
    column: str,
    label: str,
    context: str,
) -> np.ndarray:
    values = _numeric_values(
        frame,
        column=column,
        label=label,
        context=context,
    )
    if not np.isin(values, (0.0, 1.0)).all():
        raise InvalidPhysicalState(
            f"{context}: {label} must contain only 0 or 1."
        )
    return values


def _require_ordered_limits(
    frame: pd.DataFrame,
    *,
    id_values: np.ndarray,
    entity: str,
    lower_column: str,
    upper_column: str,
    context: str,
) -> None:
    lower = _numeric_values(
        frame,
        column=lower_column,
        label=f"{entity} {lower_column}",
        context=context,
    )
    upper = _numeric_values(
        frame,
        column=upper_column,
        label=f"{entity} {upper_column}",
        context=context,
    )

    invalid = np.flatnonzero(lower > upper)
    if invalid.size:
        position = int(invalid[0])
        entity_id = int(id_values[position])
        raise InvalidPhysicalState(
            f"{context}: {entity} {entity_id} has "
            f"{lower_column}={lower[position]} greater than "
            f"{upper_column}={upper[position]}."
        )


def _require_finite_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    label: str,
    context: str,
) -> None:
    for column in columns:
        _numeric_values(
            frame,
            column=column,
            label=f"{label} {column}",
            context=context,
        )


def _require_known_branch_endpoints(
    *,
    branch_ids: np.ndarray,
    from_bus: np.ndarray,
    to_bus: np.ndarray,
    known_bus_ids: set[int],
    context: str,
) -> None:
    for position, branch_id in enumerate(branch_ids):
        from_id = int(from_bus[position])
        to_id = int(to_bus[position])

        if from_id not in known_bus_ids:
            raise InvalidPhysicalState(
                f"{context}: branch {int(branch_id)} references "
                f"unknown from_bus={from_id}."
            )
        if to_id not in known_bus_ids:
            raise InvalidPhysicalState(
                f"{context}: branch {int(branch_id)} references "
                f"unknown to_bus={to_id}."
            )


def _require_known_generator_buses(
    *,
    generator_ids: np.ndarray,
    generator_bus: np.ndarray,
    known_bus_ids: set[int],
    context: str,
) -> None:
    for position, generator_id in enumerate(generator_ids):
        bus_id = int(generator_bus[position])
        if bus_id not in known_bus_ids:
            raise InvalidPhysicalState(
                f"{context}: generator {int(generator_id)} references "
                f"unknown bus={bus_id}."
            )
