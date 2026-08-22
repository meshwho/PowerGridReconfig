import sys

from grid_topology_ai import outcome_record, reward


# Test-only bridge while the remaining legacy contract tests are migrated.
# Production code no longer exposes either module.
outcome_record.TERMINAL_OUTCOME_EVIDENCE_SCHEMA_VERSION = 3
sys.modules.setdefault("grid_topology_ai.outcome_contract", outcome_record)
sys.modules.setdefault("grid_topology_ai.return_contract", reward)
