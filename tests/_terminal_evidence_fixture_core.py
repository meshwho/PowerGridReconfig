from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.contracts import (
    OUTCOME_OBJECTIVE_VERSION,
    REPLAY_BUFFER_SCHEMA_VERSION,
)
from grid_topology_ai.return_contract import (
    TERMINAL_UTILITY_GAMMA,
    VALUE_TARGET_MODE,
)
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
    "test_recover_examples_from_states.py",
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
    "_checkpoint",
    "_checkpoint_metadata",
    "_checkpoint_payload",
    "base_metadata",
)

_ARTIFACT_MARKERS = {
    "physical_objective_schema_version",
    "outcome_value_target_contract_version",
    "physics_config_contract_version",
    "physics_config_fingerprint",
}

_IDENTITY_COLUMNS = {
    "run_id",
    "iteration",
    "episode_id",
}

_OUTCOME_ERROR_RE = re.compile(
    r"^.+ row (?P<index>[^:]+): invalid terminal outcome: "
    r"(?P<detail>.+)$"
)


def _identity_fields(
    row: Mapping[str, object],
    *,
    episode_id: str | None = None,
) -> dict[str, object]:
    scenario_id = row.get("scenario_id", 0)
    return {
        "run_id": "test-run",
        "iteration": 1,
        "episode_id": episode_id or f"test-episode-{scenario_id}",
    }


def _current_outcome_fields(
    row: Mapping[str, object],
) -> dict[str, object]:
    result = dict(row)
    legacy_mode = (
        result.get("outcome_value_target_mode")
        == "alphazero_discounted"
    )

    gamma = result.get("outcome_gamma")
    legacy_gamma = False
    if (
        not isinstance(gamma, (bool, np.bool_))
        and isinstance(gamma, (int, float, np.integer, np.floating))
    ):
        numeric_gamma = float(gamma)
        legacy_gamma = (
            math.isfinite(numeric_gamma)
            and math.isclose(
                numeric_gamma,
                0.95,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        )

    if legacy_mode:
        result["outcome_value_target_mode"] = VALUE_TARGET_MODE
    if legacy_gamma:
        result["outcome_gamma"] = TERMINAL_UTILITY_GAMMA

    if not (legacy_mode or legacy_gamma):
        return result

    target = result.get("outcome_value_target")
    steps = result.get("outcome_steps_to_terminal")
    if (
        not isinstance(target, (bool, np.bool_))
        and isinstance(target, (int, float, np.integer, np.floating))
        and not isinstance(steps, (bool, np.bool_))
        and isinstance(steps, (int, np.integer))
        and int(steps) > 0
    ):
        numeric_target = float(target)
        expected_magnitude = 0.95 ** int(steps)
        if (
            math.isfinite(numeric_target)
            and numeric_target != 0.0
            and math.isclose(
                abs(numeric_target),
                expected_magnitude,
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
        ):
            result["outcome_value_target"] = (
                1.0 if numeric_target > 0.0 else -1.0
            )

    return result


def _with_terminal_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        result = _current_outcome_fields(value)
        result.setdefault(
            "outcome_objective_version",
            OUTCOME_OBJECTIVE_VERSION,
        )
        if "termination_reason" in result:
            result.update(
                terminal_evidence_fields(
                    result.get("termination_reason")
                )
            )
        if "state_path" not in result:
            for name, field_value in _identity_fields(result).items():
                result.setdefault(name, field_value)
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


def _wrap_target_builder(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = getattr(
        module,
        "add_outcome_value_targets_to_rows",
        None,
    )
    if builder is None or not callable(builder):
        return

    def current_builder(
        *args: object,
        **kwargs: object,
    ) -> Any:
        rows = args[0] if args else kwargs.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for name, value in _identity_fields(row).items():
                    row.setdefault(name, value)
        return builder(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "add_outcome_value_targets_to_rows",
        current_builder,
    )


def _wrap_example_constructor(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = getattr(module, "SelfPlayExample", None)
    if constructor is None or not callable(constructor):
        return

    def current_constructor(
        *args: object,
        **kwargs: object,
    ) -> Any:
        kwargs.setdefault(
            "outcome_objective_version",
            OUTCOME_OBJECTIVE_VERSION,
        )
        return constructor(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "SelfPlayExample",
        current_constructor,
    )


def _patch_fake_writer(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = getattr(module, "_FakeExampleWriter", None)
    columns = getattr(writer, "COLUMNS", None)
    if writer is None or not isinstance(columns, list):
        return
    if "outcome_objective_version" not in columns:
        updated = list(columns)
        position = updated.index("physical_objective_schema_version") + 1
        updated.insert(position, "outcome_objective_version")
        monkeypatch.setattr(writer, "COLUMNS", updated)

    add_example = getattr(writer, "add_example", None)
    if not callable(add_example):
        return

    def current_add_example(
        instance: object,
        *args: object,
        **kwargs: object,
    ) -> Any:
        result = add_example(instance, *args, **kwargs)
        rows = getattr(instance, "rows", None)
        if isinstance(rows, list) and rows:
            rows[-1].setdefault(
                "outcome_objective_version",
                OUTCOME_OBJECTIVE_VERSION,
            )
        return result

    monkeypatch.setattr(writer, "add_example", current_add_example)


def _wrap_checkpoint_writer(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = getattr(module, "_write_checkpoint", None)
    if writer is None or not callable(writer):
        return

    def current_writer(
        *args: object,
        **kwargs: object,
    ) -> Any:
        result = writer(*args, **kwargs)
        path_value = kwargs.get("path")
        if path_value is None and args:
            path_value = args[0]
        if path_value is None:
            return result

        import torch

        path = Path(path_value)
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            isinstance(payload, dict)
            and "checkpoint_contract_version" in payload
        ):
            payload.setdefault(
                "outcome_objective_version",
                OUTCOME_OBJECTIVE_VERSION,
            )
            torch.save(payload, path)
        return result

    monkeypatch.setattr(module, "_write_checkpoint", current_writer)


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


def _enrich_frame(
    frame: pd.DataFrame,
    *,
    add_objective: bool = True,
) -> pd.DataFrame:
    result = frame.copy()

    if (
        add_objective
        and "outcome_objective_version" not in result.columns
    ):
        result["outcome_objective_version"] = (
            OUTCOME_OBJECTIVE_VERSION
        )

    identity_columns = _IDENTITY_COLUMNS.intersection(result.columns)
    add_identity = not identity_columns
    if identity_columns == _IDENTITY_COLUMNS:
        add_identity = all(
            result[column].map(_missing).all()
            for column in _IDENTITY_COLUMNS
        )

    episode_counts: dict[str, int] = {}
    active_episodes: dict[str, str] = {}

    for index, row in result.iterrows():
        current = _current_outcome_fields(row.to_dict())
        for name in (
            "outcome_value_target",
            "outcome_value_target_mode",
            "outcome_gamma",
        ):
            if name in current and name in result.columns:
                result.at[index, name] = current[name]

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

        if add_identity:
            scenario_key = str(row.get("scenario_id", 0))
            try:
                step = int(row.get("step", 0))
            except (TypeError, ValueError):
                step = 0
            if step == 0 or scenario_key not in active_episodes:
                count = episode_counts.get(scenario_key, 0) + 1
                episode_counts[scenario_key] = count
                active_episodes[scenario_key] = (
                    f"test-episode-{scenario_key}-{count}"
                )
            identity = _identity_fields(
                row,
                episode_id=active_episodes[scenario_key],
            )
            for name, value in identity.items():
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
            prepared = _enrich_frame(
                examples,
                add_objective=False,
            )

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
        updated_metadata.setdefault(
            "outcome_objective_version",
            OUTCOME_OBJECTIVE_VERSION,
        )
        updated_metadata.update(
            terminal_evidence_metadata(
                evidence.termination_reason
            )
        )
        for field in _IDENTITY_COLUMNS:
            value = row.get(field)
            if _missing(value):
                continue
            updated_metadata[field] = (
                int(value) if field == "iteration" else str(value)
            )
        arrays["metadata_json"] = np.array(
            json.dumps(updated_metadata)
        )
        np.savez_compressed(path, **arrays)


def _patch_replay_manifest_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_text = Path.write_text

    def current_write_text(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        if path.name == "buffer_manifest.json":
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("schema_version")
                == REPLAY_BUFFER_SCHEMA_VERSION
            ):
                payload.setdefault(
                    "outcome_objective_version",
                    OUTCOME_OBJECTIVE_VERSION,
                )
                data = json.dumps(payload)
        return original_write_text(
            path,
            data,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(Path, "write_text", current_write_text)


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

    _wrap_target_builder(
        request.module,
        monkeypatch,
    )
    _wrap_example_constructor(
        request.module,
        monkeypatch,
    )
    _patch_fake_writer(
        request.module,
        monkeypatch,
    )
    _wrap_checkpoint_writer(
        request.module,
        monkeypatch,
    )
    if filename == "test_replay.py":
        _patch_replay_manifest_writes(monkeypatch)
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
