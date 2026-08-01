from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from grid_topology_ai.state_schema import state_feature_schema_provenance


_SCHEMA_FIXTURE_FILES = {
    "test_checkpoint_selector_metadata.py",
    "test_checkpoint_state.py",
    "test_example_validation.py",
    "test_examples.py",
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
    "test_neural_evaluator_normalization.py",
    "test_scenario_split_guard.py",
    "test_self_play_dataset.py",
    "test_strict_outcome_value_dataset.py",
    "test_strict_value_roundtrip.py",
}

_ROW_BUILDERS = (
    "valid_row",
    "rows",
    "_stage_rows",
    "_example_row",
    "semantic_invalid_handoff_row",
)


def _schema_csv_fields() -> dict[str, object]:
    provenance = state_feature_schema_provenance()
    return {
        "state_feature_schema_version": int(
            provenance["state_feature_schema_version"]
        ),
        "state_feature_schema_fingerprint": str(
            provenance["state_feature_schema_fingerprint"]
        ),
        "bus_feature_columns": json.dumps(
            provenance["bus_feature_columns"],
            separators=(",", ":"),
            allow_nan=False,
        ),
        "branch_feature_columns": json.dumps(
            provenance["branch_feature_columns"],
            separators=(",", ":"),
            allow_nan=False,
        ),
        "edge_index_semantics": str(
            provenance["edge_index_semantics"]
        ),
        "bus_id_semantics": str(provenance["bus_id_semantics"]),
    }


def _with_schema_fields(value: Any) -> Any:
    fields = _schema_csv_fields()

    if isinstance(value, dict):
        result = dict(value)
        for name, field_value in fields.items():
            result.setdefault(name, field_value)
        return result

    if isinstance(value, list):
        return [_with_schema_fields(item) for item in value]

    return value


def _wrap_row_builder(
    module: object,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = getattr(module, name, None)
    if builder is None or not callable(builder):
        return

    def current_builder(*args: object, **kwargs: object) -> Any:
        return _with_schema_fields(builder(*args, **kwargs))

    monkeypatch.setattr(module, name, current_builder)


def _is_artifact_frame(frame: pd.DataFrame) -> bool:
    markers = {
        "physical_objective_schema_version",
        "outcome_value_target_contract_version",
        "physics_config_contract_version",
        "physics_config_fingerprint",
    }
    return len(frame) > 0 and len(markers.intersection(frame.columns)) >= 2


def _enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    fields = _schema_csv_fields()

    for name, value in fields.items():
        if name not in result.columns:
            result[name] = pd.Series(
                [value] * len(result),
                index=result.index,
                dtype=object,
            )
            continue

        result[name] = result[name].astype(object)
        for index, current in result[name].items():
            missing = current is None
            if not missing:
                observed = pd.isna(current)
                missing = isinstance(observed, (bool, np.bool_)) and bool(
                    observed
                )
            if missing:
                result.at[index, name] = value
            elif name in {
                "bus_feature_columns",
                "branch_feature_columns",
            } and not isinstance(current, str):
                result.at[index, name] = json.dumps(
                    current,
                    separators=(",", ":"),
                    allow_nan=False,
                )

    return result


def _enrich_state_arrays(arrays: Mapping[str, object]) -> dict[str, object]:
    result = dict(arrays)
    bus_features = result.get("bus_features")
    metadata_json = result.get("metadata_json")

    if bus_features is None or metadata_json is None:
        return result

    bus_matrix = np.asarray(bus_features)
    if bus_matrix.ndim >= 1:
        result.setdefault(
            "bus_ids",
            np.arange(int(bus_matrix.shape[0]), dtype=np.int64),
        )

    raw_metadata = np.asarray(metadata_json)
    if raw_metadata.size != 1:
        return result

    try:
        metadata = json.loads(str(raw_metadata.item()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return result

    if not isinstance(metadata, dict):
        return result

    for name, value in state_feature_schema_provenance().items():
        metadata.setdefault(name, value)
    result["metadata_json"] = np.array(json.dumps(metadata))
    return result


def _enrich_checkpoint(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    provenance = state_feature_schema_provenance()
    for name, value in provenance.items():
        result.setdefault(name, value)

    if str(result.get("model_type", "")) in {
        "graph_policy_value_net",
        "graph_policy_value_net_v2",
    }:
        metadata = result.get("dataset_metadata")
        dataset_metadata = (
            dict(metadata) if isinstance(metadata, Mapping) else {}
        )
        for name, value in provenance.items():
            dataset_metadata.setdefault(name, value)
        result["dataset_metadata"] = dataset_metadata

    return result


@pytest.fixture(autouse=True)
def _current_state_schema_fixtures(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = Path(str(request.node.fspath)).name
    if filename not in _SCHEMA_FIXTURE_FILES:
        return

    for name in _ROW_BUILDERS:
        _wrap_row_builder(request.module, name, monkeypatch)

    original_savez = np.savez

    def current_savez(file: object, *args: object, **kwargs: object):
        if not args:
            kwargs = _enrich_state_arrays(kwargs)
        return original_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez", current_savez)

    original_to_csv = pd.DataFrame.to_csv

    def current_to_csv(
        frame: pd.DataFrame,
        path_or_buf: object = None,
        *args: object,
        **kwargs: object,
    ):
        if _is_artifact_frame(frame):
            frame = _enrich_frame(frame)
        return original_to_csv(
            frame,
            path_or_buf,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(pd.DataFrame, "to_csv", current_to_csv)

    original_torch_save = torch.save

    def current_torch_save(
        obj: object,
        f: object,
        *args: object,
        **kwargs: object,
    ):
        if (
            isinstance(obj, Mapping)
            and "checkpoint_contract_version" in obj
        ):
            obj = _enrich_checkpoint(obj)
        return original_torch_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", current_torch_save)
