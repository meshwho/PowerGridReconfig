from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from grid_topology_ai.contracts import OUTCOME_OBJECTIVE_VERSION
from tests._state_schema_fixture_core import (
    _current_state_schema_fixtures,
)
from tests._terminal_evidence_fixture_core import (
    _current_terminal_evidence_fixtures,
)


@pytest.fixture(autouse=True)
def _current_outcome_objective_checkpoint_fixtures(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = Path(str(request.node.fspath)).name
    if filename not in {
        "test_iteration.py",
        "test_typed_stage_config.py",
        "test_neural_evaluator_normalization.py",
    }:
        return

    original_save = torch.save

    def current_save(obj, file, *args, **kwargs):
        if (
            isinstance(obj, Mapping)
            and "checkpoint_contract_version" in obj
        ):
            payload = dict(obj)
            payload.setdefault(
                "outcome_objective_version",
                OUTCOME_OBJECTIVE_VERSION,
            )
            obj = payload
        return original_save(obj, file, *args, **kwargs)

    monkeypatch.setattr(torch, "save", current_save)


__all__ = (
    "_current_state_schema_fixtures",
    "_current_terminal_evidence_fixtures",
    "_current_outcome_objective_checkpoint_fixtures",
)
