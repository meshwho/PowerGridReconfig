from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from grid_topology_ai.state_schema import (
    state_feature_schema_fingerprint,
)

if TYPE_CHECKING:
    from grid_topology_ai.data_adapter import GridFMState


_FINGERPRINT_VERSION = b"physical-state-v1"


def _update_array(
    digest,
    *,
    name: str,
    value: object,
    dtype: str,
) -> None:
    array = np.asarray(value, dtype=np.dtype(dtype))
    array = np.ascontiguousarray(array)

    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def physical_state_fingerprint(state: GridFMState) -> str:
    """Return a stable fingerprint of a graph state's physical inputs."""

    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_VERSION)
    digest.update(b"\0")
    digest.update(
        state_feature_schema_fingerprint().encode("ascii")
    )
    digest.update(b"\0")

    _update_array(
        digest,
        name="scenario_id",
        value=[state.scenario_id],
        dtype="<i8",
    )
    _update_array(
        digest,
        name="load_scenario_idx",
        value=[state.load_scenario_idx],
        dtype="<f8",
    )
    _update_array(
        digest,
        name="bus_features",
        value=state.bus_features,
        dtype="<f4",
    )
    _update_array(
        digest,
        name="branch_features",
        value=state.branch_features,
        dtype="<f4",
    )
    _update_array(
        digest,
        name="edge_index",
        value=state.edge_index,
        dtype="<i8",
    )

    if state.bus_ids is None:
        digest.update(b"bus_ids\0none\0")
    else:
        _update_array(
            digest,
            name="bus_ids",
            value=state.bus_ids,
            dtype="<i8",
        )

    _update_array(
        digest,
        name="branch_ids",
        value=state.branch_ids,
        dtype="<i8",
    )
    _update_array(
        digest,
        name="branch_status",
        value=state.branch_status,
        dtype="<f4",
    )
    _update_array(
        digest,
        name="outaged_branch_ids",
        value=sorted(
            int(branch_id)
            for branch_id in state.outaged_branch_ids
        ),
        dtype="<i8",
    )

    return digest.hexdigest()