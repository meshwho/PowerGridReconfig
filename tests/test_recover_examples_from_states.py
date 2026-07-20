import json
from pathlib import Path

import numpy as np
import pytest

from grid_topology_ai.physical_objective import (
    PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.self_play.recover_examples_from_states import (
    recover_examples,
)


def write_recovery_state(
    path: Path,
    metadata: dict,
) -> Path:
    np.savez(
        path,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return path


def base_metadata() -> dict:
    return {
        "physical_objective_schema_version": (
            PHYSICAL_OBJECTIVE_SCHEMA_VERSION
        ),
        "scenario_id": 1,
        "step": 0,
        "selected_action_id": 1,
        "selected_branch_id": 7,
        "step_safety_improvement": 10.0,
        "teacher_decision_reason": "test",
    }


def test_recovery_rejects_missing_exact_outcome(
    tmp_path: Path,
) -> None:
    write_recovery_state(
        tmp_path / "impact_teacher_scenario_000001_step_000.npz",
        base_metadata(),
    )

    with pytest.raises(
        ValueError,
        match="does not contain exact terminal outcome metadata",
    ):
        recover_examples(tmp_path, gamma=0.9)


def test_recovery_preserves_solved_outcome(
    tmp_path: Path,
) -> None:
    metadata = base_metadata()
    metadata.update(
        {
            "episode_done": True,
            "episode_solved": True,
            "episode_termination_reason": "solved",
        }
    )

    write_recovery_state(
        tmp_path / "impact_teacher_scenario_000001_step_000.npz",
        metadata,
    )

    recovered = recover_examples(
        tmp_path,
        gamma=0.9,
    )

    assert len(recovered) == 1
    assert bool(recovered.iloc[0]["solved"]) is True
    assert recovered.iloc[0]["termination_reason"] == "solved"
    assert recovered.iloc[0]["outcome_class"] == "solved"
    assert recovered.iloc[0]["outcome_value_target"] == pytest.approx(0.9)