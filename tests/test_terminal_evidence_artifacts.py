from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.self_play.example_validation import (
    load_and_validate_examples_csv,
)
from grid_topology_ai.self_play.examples import ExampleWriter
from grid_topology_ai.state.schema import (
    BRANCH_FEATURE_COLUMNS,
    BUS_FEATURE_COLUMNS,
)
from grid_topology_ai.termination import TerminationReason
from grid_topology_ai.value_targets import (
    add_outcome_value_targets_to_rows,
)
from tests.outcome_evidence_helpers import terminal_evidence
from tests.topology_contract_helpers import TEST_ACTION_SPACE_CONFIG


def _state() -> GridFMState:
    branch_features = np.zeros(
        (1, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[
        0,
        BRANCH_FEATURE_COLUMNS.index("br_status"),
    ] = 1.0

    return GridFMState(
        scenario_id=1,
        load_scenario_idx=0.0,
        bus_features=np.zeros(
            (2, len(BUS_FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        branch_features=branch_features,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        branch_ids=np.array([7], dtype=np.int64),
        branch_status=np.array([1.0], dtype=np.float32),
        metrics={},
        outaged_branch_ids=[],
        bus_ids=np.array([10, 20], dtype=np.int64),
    )


def _write_episode(
    tmp_path: Path,
    *,
    reason: TerminationReason = TerminationReason.SOLVED,
    topology_utility: float | None = None,
) -> tuple[Path, Path]:
    evidence = terminal_evidence(
        reason,
        topology_utility=topology_utility,
    )
    writer = ExampleWriter(
        tmp_path,
        physics_config=DEFAULT_PHYSICS_CONFIG,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    )
    writer.add_example(
        state=_state(),
        state_id="state-1",
        action_mask=np.array([True, True], dtype=bool),
        scenario_id=1,
        step=0,
        selected_action_id=0,
        selected_branch_id=None,
        step_reward=0.0,
        final_return=0.0,
        discounted_return_from_step=0.0,
        solved=evidence.solved,
        done=True,
        termination_reason=evidence.termination_reason,
        terminal_outcome_evidence=evidence,
        visit_counts={0: 1},
        mcts_policy={0: 1.0},
    )

    examples_path = writer.save()
    frame = pd.read_csv(examples_path)
    rows = frame.to_dict(orient="records")
    add_outcome_value_targets_to_rows(
        rows,
        gamma=1.0,
    )
    pd.DataFrame(rows).to_csv(examples_path, index=False)

    return examples_path, Path(rows[0]["state_path"])


def _rewrite_state_metadata(
    state_path: Path,
    mutate,
) -> None:
    with np.load(state_path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
        }

    metadata = json.loads(
        str(np.asarray(arrays["metadata_json"]).item())
    )
    mutate(metadata)
    arrays["metadata_json"] = np.array(
        json.dumps(metadata)
    )
    np.savez_compressed(state_path, **arrays)


def test_valid_terminal_evidence_artifacts_are_accepted(
    tmp_path: Path,
) -> None:
    examples_path, _ = _write_episode(tmp_path)

    examples = load_and_validate_examples_csv(examples_path)

    assert len(examples) == 1
    assert "terminal_outcome_evidence_schema_version" not in examples.columns
    assert examples.loc[0, "terminal_outcome_evidence_json"]
    assert examples.loc[0, "run_id"]
    assert examples.loc[0, "episode_id"]
    assert examples.loc[0, "iteration"] == 1


def test_validated_redispatch_preserves_topology_target(
    tmp_path: Path,
) -> None:
    topology_utility = 0.35
    examples_path, _ = _write_episode(
        tmp_path,
        reason=TerminationReason.REDISPATCH_VALIDATED,
        topology_utility=topology_utility,
    )

    examples = load_and_validate_examples_csv(examples_path)

    assert examples.loc[0, "outcome_value_target"] == pytest.approx(
        topology_utility
    )
    assert examples.loc[0, "outcome_class"] == (
        TerminationReason.REDISPATCH_VALIDATED.value
    )


def test_episode_rejects_mixed_terminal_evidence(
    tmp_path: Path,
) -> None:
    solved_path, _ = _write_episode(
        tmp_path / "solved",
        reason=TerminationReason.SOLVED,
    )
    failed_path, _ = _write_episode(
        tmp_path / "failed",
        reason=TerminationReason.MAX_STEPS_REACHED,
    )

    rows = [
        pd.read_csv(solved_path).iloc[0].to_dict(),
        pd.read_csv(failed_path).iloc[0].to_dict(),
    ]
    for step, row in enumerate(rows):
        row.update(
            run_id="run-1",
            iteration=1,
            episode_id="episode-1",
            scenario_id=1,
            step=step,
        )

    examples_path = tmp_path / "mixed_examples.csv"
    pd.DataFrame(rows).to_csv(examples_path, index=False)

    with pytest.raises(
        ValueError,
        match="mixed terminal outcomes or evidence",
    ):
        load_and_validate_examples_csv(examples_path)


def test_missing_csv_evidence_column_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, _ = _write_episode(tmp_path)
    frame = pd.read_csv(examples_path).drop(
        columns=["terminal_outcome_evidence_json"]
    )
    frame.to_csv(examples_path, index=False)

    with pytest.raises(
        ValueError,
        match="terminal_outcome_evidence_json",
    ):
        load_and_validate_examples_csv(examples_path)


def test_invalid_csv_evidence_json_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, _ = _write_episode(tmp_path)
    frame = pd.read_csv(examples_path)
    frame.loc[0, "terminal_outcome_evidence_json"] = "{"
    frame.to_csv(examples_path, index=False)

    with pytest.raises(
        ValueError,
        match="invalid terminal outcome evidence",
    ):
        load_and_validate_examples_csv(examples_path)


def test_csv_evidence_outcome_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, _ = _write_episode(tmp_path)
    frame = pd.read_csv(examples_path)
    frame.loc[0, "terminal_outcome_evidence_json"] = terminal_evidence(
        TerminationReason.MAX_STEPS_REACHED
    ).to_json()
    frame.to_csv(examples_path, index=False)

    with pytest.raises(
        ValueError,
        match="contradicts solved or termination_reason",
    ):
        load_and_validate_examples_csv(examples_path)


def test_missing_state_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, state_path = _write_episode(tmp_path)

    def remove_evidence(metadata: dict[str, object]) -> None:
        metadata.pop("terminal_outcome_evidence", None)

    _rewrite_state_metadata(state_path, remove_evidence)

    with pytest.raises(
        ValueError,
        match="missing terminal_outcome_evidence",
    ):
        load_and_validate_examples_csv(examples_path)


def test_csv_and_state_evidence_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, state_path = _write_episode(tmp_path)

    def change_evidence(metadata: dict[str, object]) -> None:
        metadata["terminal_outcome_evidence"] = terminal_evidence(
            TerminationReason.MAX_STEPS_REACHED
        ).to_dict()

    _rewrite_state_metadata(state_path, change_evidence)

    with pytest.raises(
        ValueError,
        match="does not match state metadata",
    ):
        load_and_validate_examples_csv(examples_path)


def test_csv_and_state_episode_identity_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    examples_path, state_path = _write_episode(tmp_path)

    def change_episode_id(metadata: dict[str, object]) -> None:
        metadata["episode_id"] = "other-episode"

    _rewrite_state_metadata(state_path, change_episode_id)

    with pytest.raises(ValueError, match="episode_id"):
        load_and_validate_examples_csv(examples_path)
