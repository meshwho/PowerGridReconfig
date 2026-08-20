"""Self-play pipeline components."""

import sys

from grid_topology_ai.self_play import provenance

physical_lineage = provenance
lineage_artifacts = provenance
validation_snapshot = provenance
sys.modules[f"{__name__}.physical_lineage"] = provenance
sys.modules[f"{__name__}.lineage_artifacts"] = provenance
sys.modules[f"{__name__}.validation_snapshot"] = provenance
