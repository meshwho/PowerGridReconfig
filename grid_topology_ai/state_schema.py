from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from grid_topology_ai.power_flow_errors import InvalidPhysicalState


STATE_FEATURE_SCHEMA_VERSION = 2

BUS_FEATURE_COLUMNS = [
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "GS",
    "BS",
    "min_vm_pu",
    "max_vm_pu",
    "gen_online_count",
    "gen_available",
    "gen_p_min_mw",
    "gen_p_max_mw",
    "gen_q_min_mvar",
    "gen_q_max_mvar",
    "gen_p_down_margin_mw",
    "gen_p_up_margin_mw",
    "gen_q_down_margin_mvar",
    "gen_q_up_margin_mvar",
]

BRANCH_FEATURE_COLUMNS = [
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "tap",
    "shift",
    "rate_a",
    "br_status",
    "s_from_mva",
    "s_to_mva",
    "s_max_mva",
    "loading_percent",
    "unlimited_rating",
]

_GENERATOR_FEATURE_COLUMNS = [
    "gen_online_count",
    "gen_available",
    "gen_p_min_mw",
    "gen_p_max_mw",
    "gen_q_min_mvar",
    "gen_q_max_mvar",
    "gen_p_down_margin_mw",
    "gen_p_up_margin_mw",
    "gen_q_down_margin_mvar",
    "gen_q_up_margin_mvar",
]


def state_feature_schema_payload() -> dict[str, object]:
    return {
        "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION,
        "bus_feature_columns": list(BUS_FEATURE_COLUMNS),
        "branch_feature_columns": list(BRANCH_FEATURE_COLUMNS),
    }


def state_feature_schema_fingerprint() -> str:
    payload = json.dumps(
        state_feature_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_feature_schema_provenance() -> dict[str, object]:
    return {
        **state_feature_schema_payload(),
        "state_feature_schema_fingerprint": (
            state_feature_schema_fingerprint()
        ),
    }


def with_bus_generator_features(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return bus rows with active-generator limits and operating margins."""

    required_bus_columns = {"bus", "min_vm_pu", "max_vm_pu"}
    required_gen_columns = {
        "bus",
        "p_mw",
        "q_mvar",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    }

    missing_bus = required_bus_columns - set(bus_df.columns)
    missing_gen = required_gen_columns - set(gen_df.columns)

    if missing_bus:
        raise InvalidPhysicalState(
            f"Bus data is missing schema columns: {sorted(missing_bus)}."
        )
    if missing_gen:
        raise InvalidPhysicalState(
            f"Generator data is missing schema columns: {sorted(missing_gen)}."
        )

    result = bus_df.copy()
    result["Pg"] = 0.0
    result["Qg"] = 0.0

    for column in _GENERATOR_FEATURE_COLUMNS:
        result[column] = 0.0

    status = gen_df["in_service"].to_numpy(dtype=np.float64)
    if not np.isfinite(status).all():
        raise InvalidPhysicalState(
            "Generator in_service contains NaN or infinity."
        )
    if not np.isin(status, (0.0, 1.0)).all():
        raise InvalidPhysicalState(
            "Generator in_service must contain only 0 or 1."
        )

    active = gen_df.loc[status > 0.0]
    if active.empty:
        return result

    numeric_columns = [
        "p_mw",
        "q_mvar",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
    ]
    numeric = active[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise InvalidPhysicalState(
            "Active generator limits and outputs must be finite."
        )

    grouped = active.groupby("bus", sort=False).agg(
        Pg=("p_mw", "sum"),
        Qg=("q_mvar", "sum"),
        gen_online_count=("bus", "size"),
        gen_p_min_mw=("min_p_mw", "sum"),
        gen_p_max_mw=("max_p_mw", "sum"),
        gen_q_min_mvar=("min_q_mvar", "sum"),
        gen_q_max_mvar=("max_q_mvar", "sum"),
    )

    mapped_columns = [
        "Pg",
        "Qg",
        "gen_online_count",
        "gen_p_min_mw",
        "gen_p_max_mw",
        "gen_q_min_mvar",
        "gen_q_max_mvar",
    ]
    bus_ids = result["bus"]

    for column in mapped_columns:
        result[column] = bus_ids.map(grouped[column]).fillna(0.0)

    result["gen_available"] = (
        result["gen_online_count"] > 0.0
    ).astype(np.float64)
    result["gen_p_down_margin_mw"] = (
        result["Pg"] - result["gen_p_min_mw"]
    )
    result["gen_p_up_margin_mw"] = (
        result["gen_p_max_mw"] - result["Pg"]
    )
    result["gen_q_down_margin_mvar"] = (
        result["Qg"] - result["gen_q_min_mvar"]
    )
    result["gen_q_up_margin_mvar"] = (
        result["gen_q_max_mvar"] - result["Qg"]
    )

    return result


def with_branch_rating_features(
    branch_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add the explicit unlimited-rating flag used by schema v2."""

    if "rate_a" not in branch_df.columns:
        raise InvalidPhysicalState(
            "Branch data is missing required column: rate_a."
        )

    result = branch_df.copy()
    rate_a = result["rate_a"].to_numpy(dtype=np.float64)

    if not np.isfinite(rate_a).all():
        raise InvalidPhysicalState(
            "Branch rate_a contains NaN or infinity."
        )
    if np.any(rate_a < 0.0):
        raise InvalidPhysicalState(
            "Branch rate_a must be non-negative."
        )

    result["unlimited_rating"] = (rate_a == 0.0).astype(np.float32)
    return result


def finite_feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise InvalidPhysicalState(
            f"{label.capitalize()} data is missing feature columns: "
            f"{sorted(missing)}."
        )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        features = frame[list(columns)].to_numpy(dtype=np.float32)

    if not np.isfinite(features).all():
        raise InvalidPhysicalState(
            f"{label.capitalize()} features cannot be represented in float32."
        )

    return features
