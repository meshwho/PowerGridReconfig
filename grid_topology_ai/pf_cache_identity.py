from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)


_NETWORK_FINGERPRINT_VERSION = b"pf-network-v1"
_TOPOLOGY_FINGERPRINT_VERSION = b"pf-topology-v1"
_EXACT_FINGERPRINT_VERSION = b"pf-problem-v1"

_NETWORK_BUS_COLUMNS = (
    BUS_I,
    GS,
    BS,
    BASE_KV,
    ZONE,
    VMAX,
    VMIN,
)
_TOPOLOGY_BUS_COLUMNS = (
    BUS_I,
    BUS_TYPE,
)
_EXACT_BUS_COLUMNS = (
    BUS_I,
    BUS_TYPE,
    PD,
    QD,
    GS,
    BS,
    BASE_KV,
    ZONE,
    VMAX,
    VMIN,
)
_NETWORK_BRANCH_COLUMNS = (
    F_BUS,
    T_BUS,
    BR_R,
    BR_X,
    BR_B,
    RATE_A,
    RATE_B,
    RATE_C,
    TAP,
    SHIFT,
    ANGMIN,
    ANGMAX,
)
_EXACT_BRANCH_COLUMNS = (*_NETWORK_BRANCH_COLUMNS, BR_STATUS)
_EXACT_GEN_COLUMNS = (
    GEN_BUS,
    PG,
    QG,
    QMAX,
    QMIN,
    VG,
    MBASE,
    GEN_STATUS,
    PMAX,
    PMIN,
)


def _matrix(ppc: dict[str, Any], name: str) -> np.ndarray:
    values = np.asarray(ppc[name], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"ppc[{name!r}] must be a two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError(f"ppc[{name!r}] contains NaN or infinity.")
    return values


def _require_columns(values: np.ndarray, columns: Sequence[int], name: str) -> None:
    if values.shape[1] <= max(columns, default=-1):
        raise ValueError(
            f"ppc[{name!r}] does not contain all required cache-key columns."
        )


def _ids(
    values: Sequence[int] | np.ndarray | None,
    rows: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or len(result) != rows:
        raise ValueError(f"{name} must contain exactly one id per matrix row.")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must contain unique ids.")
    return result


def _canonical_rows(
    values: np.ndarray,
    columns: Sequence[int],
    *,
    row_ids: Sequence[int] | np.ndarray | None = None,
    id_name: str,
) -> np.ndarray:
    _require_columns(values, columns, id_name)
    selected = np.ascontiguousarray(values[:, columns], dtype=np.float64)
    ids = _ids(row_ids, len(selected), id_name)

    if ids is not None:
        order = np.argsort(ids, kind="stable")
        id_column = ids[order].astype(np.float64, copy=False).reshape(-1, 1)
        return np.ascontiguousarray(
            np.concatenate((id_column, selected[order]), axis=1),
            dtype=np.float64,
        )

    if len(selected) <= 1:
        return selected

    keys = tuple(
        selected[:, index]
        for index in reversed(range(selected.shape[1]))
    )
    order = np.lexsort(keys)
    return np.ascontiguousarray(selected[order], dtype=np.float64)


def _update_text(digest: Any, name: str, value: str) -> None:
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(value.encode("ascii"))
    digest.update(b"\0")


def _update_float(digest: Any, name: str, value: float) -> None:
    _update_array(digest, name, np.asarray([value], dtype="<f8"))


def _update_array(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")


def network_fingerprint(
    ppc: dict[str, Any],
    *,
    branch_ids: Sequence[int] | np.ndarray | None = None,
) -> str:
    """Return a stable identity for the static electrical network."""

    bus = _matrix(ppc, "bus")
    branch = _matrix(ppc, "branch")
    base_mva = float(ppc["baseMVA"])
    if not np.isfinite(base_mva):
        raise ValueError("ppc['baseMVA'] must be finite.")

    digest = hashlib.sha256()
    digest.update(_NETWORK_FINGERPRINT_VERSION)
    digest.update(b"\0")
    _update_float(digest, "base_mva", base_mva)
    _update_array(
        digest,
        "bus",
        _canonical_rows(bus, _NETWORK_BUS_COLUMNS, id_name="bus_rows"),
    )
    _update_array(
        digest,
        "branch",
        _canonical_rows(
            branch,
            _NETWORK_BRANCH_COLUMNS,
            row_ids=branch_ids,
            id_name="branch_ids",
        ),
    )
    return digest.hexdigest()


def topology_fingerprint(
    ppc: dict[str, Any],
    *,
    branch_ids: Sequence[int] | np.ndarray | None = None,
) -> str:
    """Return a scenario-independent identity for one network topology."""

    bus = _matrix(ppc, "bus")
    branch = _matrix(ppc, "branch")
    _require_columns(branch, (BR_STATUS,), "branch")

    digest = hashlib.sha256()
    digest.update(_TOPOLOGY_FINGERPRINT_VERSION)
    digest.update(b"\0")
    _update_text(
        digest,
        "network",
        network_fingerprint(ppc, branch_ids=branch_ids),
    )
    _update_array(
        digest,
        "bus_type",
        _canonical_rows(bus, _TOPOLOGY_BUS_COLUMNS, id_name="bus_rows"),
    )
    _update_array(
        digest,
        "branch_status",
        _canonical_rows(
            branch,
            (BR_STATUS,),
            row_ids=branch_ids,
            id_name="branch_ids",
        ),
    )
    return digest.hexdigest()


def exact_pf_problem_fingerprint(
    ppc: dict[str, Any],
    *,
    physics_fingerprint: str,
    branch_ids: Sequence[int] | np.ndarray | None = None,
    generator_ids: Sequence[int] | np.ndarray | None = None,
) -> str:
    """Return the identity of a complete AC power-flow problem.

    Scenario metadata and solved output quantities are intentionally excluded.
    A matching fingerprint is therefore suitable for exact cross-scenario reuse.
    """

    if not physics_fingerprint:
        raise ValueError("physics_fingerprint must not be empty.")

    bus = _matrix(ppc, "bus")
    gen = _matrix(ppc, "gen")
    branch = _matrix(ppc, "branch")

    digest = hashlib.sha256()
    digest.update(_EXACT_FINGERPRINT_VERSION)
    digest.update(b"\0")
    _update_text(digest, "physics", str(physics_fingerprint))
    _update_text(
        digest,
        "topology",
        topology_fingerprint(ppc, branch_ids=branch_ids),
    )
    _update_array(
        digest,
        "bus_input",
        _canonical_rows(bus, _EXACT_BUS_COLUMNS, id_name="bus_rows"),
    )
    _update_array(
        digest,
        "generator_input",
        _canonical_rows(
            gen,
            _EXACT_GEN_COLUMNS,
            row_ids=generator_ids,
            id_name="generator_ids",
        ),
    )
    _update_array(
        digest,
        "branch_input",
        _canonical_rows(
            branch,
            _EXACT_BRANCH_COLUMNS,
            row_ids=branch_ids,
            id_name="branch_ids",
        ),
    )
    return digest.hexdigest()
