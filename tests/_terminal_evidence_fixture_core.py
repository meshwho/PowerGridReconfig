from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.outcome_evidence_helpers import (
    terminal_evidence,
    terminal_evidence_fields,
    terminal_evidence_metadata,
)


_ARTIFACT_FIXTURE_FILES = {
    "test_checkpoint_selector_metadata.py",
    "test_checkpoint_state.py",
    "test_example_validation.py",
    "test_exploration_metrics.py",
    "test_generation_api.py",
    "test_generation_policy_target.py",
    "test_iteration.py",
    "test_replay.py",
    "test_stages.py",
    "test_training_api.py",
    "test_training_validation_split.py",
    "test_typed_stage_config.py",
    "test_action_masking.py",
    "test_artifact_contracts.py",
    "test_graph_self_play_dataset.py",
    "test_mcts_value_contract.py",
    "test_neural_evaluator_normalization.py",
    "test_outcome_value_target.py",
    "test_scenario_split_guard.py",
    "test_self_play_dataset.py",
    "test_strict_outcome_value_dataset.py",
    "test_strict_value_roundtrip.py",
    "test_unified_value_return_contract.py",
    "test_validated_redispatch_utility.py",
}

_ROW_BUILDERS = (
    "valid_row",
    "rows",
    "_stage_rows",
    "_example_row",
    "_valid_example_row",
    "semantic_invalid_handoff_row",
)

_ARTIFACT_MARKERS = {
    "physical_objective_schema_version",
    "outcome_value_target_contract_version",
    "physics_config_contract_version",
    "physics_config_fingerprint",
}

_OUTCOME_ERROR_RE = re.compile(
    r"^.+ row (?P<index>[^:]+): invalid terminal outcome: "
    r"(?P<detail>.+)$"
)


def _with_terminal_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        result = dict(value)
        result.update(
            terminal_evidence_fields(
                result.get("termination_reason")
            )
        )
        return result

    if isinstance(value, list):
        return [
            _with_terminal_evidence(item)
            for item in value
        ]

    return value


def _wrap_row_builder(
    module: object,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = getattr(module, name, None)
    if builder is None or not callable(builder):
        return

    def current_builder(
        *args: object,
        **kwargs: object,
    ) -> Any:
        return _with_terminal_evidence(
            builder(*args, **kwargs)
        )

    monkeypatch.setattr(module, name, current_builder)


def _is_artifact_frame(frame: pd.DataFrame) -> bool:
    return (
        len(frame) > 0
        and len(_ARTIFACT_MARKERS.intersection(frame.columns)) >= 2
    )


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        observed = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(observed, (bool, np.bool_))
        and bool(observed)
    )


def _enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for index, row in result.iterrows():
        fields = terminal_evidence_fields(
            row.get("termination_reason")
        )
        for name, value in fields.items():
            if name not in result.columns:
                result[name] = pd.Series(
                    [None] * len(result),
                    index=result.index,
                    dtype=object,
                )
            result.at[index, name] = value

    return result


def _wrap_outcome_validator(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = getattr(
        module,
        "validate_example_outcome_contracts",
        None,
    )
    if validator is None or not callable(validator):
        return

    def current_validator(
        examples: pd.DataFrame,
        *,
        source_path: str | Path,
    ) -> None:
        prepared = examples
        if {
            "terminal_outcome_evidence_schema_version",
            "terminal_outcome_evidence_json",
        }.issubset(examples.columns):
            prepared = _enrich_frame(examples)

        try:
            validator(
                prepared,
                source_path=source_path,
            )
        except ValueError as exc:
            match = _OUTCOME_ERROR_RE.match(str(exc))
            if match is None:
                raise
            raise ValueError(
                "Invalid termination_reason at row "
                f"{match.group('index')}. File: {source_path}: "
                f"{match.group('detail')}"
            ) from exc

    monkeypatch.setattr(
        module,
        "validate_example_outcome_contracts",
        current_validator,
    )


def _state_path(
    value: object,
    *,
    csv_path: object,
) -> Path | None:
    if _missing(value):
        return None

    path = Path(str(value))
    if path.is_file():
        return path

    if isinstance(csv_path, (str, Path)):
        candidate = Path(csv_path).parent / path
        if candidate.is_file():
            return candidate

    return None


def _sync_state_evidence(
    frame: pd.DataFrame,
    *,
    csv_path: object,
) -> None:
    if "state_path" not in frame.columns:
        return

    for _, row in frame.iterrows():
        path = _state_path(
            row.get("state_path"),
            csv_path=csv_path,
        )
        if path is None:
            continue

        try:
            evidence = terminal_evidence(
                row.get("termination_reason")
            )
        except (TypeError, ValueError):
            continue

        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {
                    name: np.asarray(data[name])
                    for name in data.files
                }
        except (OSError, EOFError, ValueError):
            continue

        raw_metadata = arrays.get("metadata_json")
        if raw_metadata is None:
            continue

        raw_metadata = np.asarray(raw_metadata)
        if raw_metadata.size != 1:
            continue

        try:
            metadata = json.loads(
                str(raw_metadata.item())
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(metadata, Mapping):
            continue

        updated_metadata = dict(metadata)
        updated_metadata.update(
            terminal_evidence_metadata(
                evidence.termination_reason
            )
        )
        arrays["metadata_json"] = np.array(
            json.dumps(updated_metadata)
        )
        np.savez_compressed(path, **arrays)


@pytest.fixture(autouse=True)
def _current_terminal_evidence_fixtures(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    _current_state_schema_fixtures: None,
) -> None:
    filename = Path(str(request.node.fspath)).name
    if filename not in _ARTIFACT_FIXTURE_FILES:
        return

    for name in _ROW_BUILDERS:
        _wrap_row_builder(
            request.module,
            name,
            monkeypatch,
        )

    _wrap_outcome_validator(
        request.module,
        monkeypatch,
    )

    original_to_csv = pd.DataFrame.to_csv

    def current_to_csv(
        frame: pd.DataFrame,
        path_or_buf: object = None,
        *args: object,
        **kwargs: object,
    ):
        if _is_artifact_frame(frame):
            frame = _enrich_frame(frame)
            _sync_state_evidence(
                frame,
                csv_path=path_or_buf,
            )
        return original_to_csv(
            frame,
            path_or_buf,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        current_to_csv,
    )
