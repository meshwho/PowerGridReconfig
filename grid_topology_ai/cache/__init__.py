import math
import os

from grid_topology_ai.cache.byte_lru import ByteLRUCache
from grid_topology_ai.cache.exact_power_flow import (
    DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES as _BUILTIN_DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES,
    CachedPowerFlowFailure,
    CachedPowerFlowSuccess,
    ExactPowerFlowCache,
    exact_power_flow_fingerprint,
)
from grid_topology_ai.cache.lodf_structure import (
    DEFAULT_LODF_STRUCTURE_CACHE_BYTES,
    LODFStructureCache,
    lodf_structure_fingerprint,
)
from grid_topology_ai.cache.structural_topology import (
    DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES,
    StructuralTopologyCache,
    structural_topology_fingerprint,
)


EXACT_L1_CACHE_MAX_MB_ENV = "POWERGRID_EXACT_L1_CACHE_MAX_MB"


def _configured_exact_l1_cache_bytes() -> int:
    raw_value = os.environ.get(EXACT_L1_CACHE_MAX_MB_ENV, "").strip()
    if not raw_value:
        return int(_BUILTIN_DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES)

    try:
        max_mb = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{EXACT_L1_CACHE_MAX_MB_ENV} must be a positive number of MiB, "
            f"got {raw_value!r}."
        ) from exc

    if not math.isfinite(max_mb) or max_mb <= 0.0:
        raise ValueError(
            f"{EXACT_L1_CACHE_MAX_MB_ENV} must be a positive finite number of MiB, "
            f"got {raw_value!r}."
        )

    return max(int(max_mb * 1024.0 * 1024.0), 1)


DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES = _configured_exact_l1_cache_bytes()


__all__ = [
    "ByteLRUCache",
    "CachedPowerFlowFailure",
    "CachedPowerFlowSuccess",
    "DEFAULT_EXACT_POWER_FLOW_CACHE_BYTES",
    "DEFAULT_LODF_STRUCTURE_CACHE_BYTES",
    "DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES",
    "EXACT_L1_CACHE_MAX_MB_ENV",
    "ExactPowerFlowCache",
    "LODFStructureCache",
    "StructuralTopologyCache",
    "exact_power_flow_fingerprint",
    "lodf_structure_fingerprint",
    "structural_topology_fingerprint",
]
