from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tests.topology_contract_helpers import (
    TEST_ACTION_SPACE_CONFIG,
    branch_ids_from_state_path,
    checkpoint_topology_fields,
    enrich_state_arrays,
    fake_dataset_topology_fields,
    topology_csv_fields,
    topology_metadata,
)


_LEGACY_MCTS_FAKE_ENV_FILES = {
    "test_mcts_action_ranking.py",
    "test_mcts_dc_screening_depth.py",
    "test_mcts_off_prior_exploration.py",
}

_TOPOLOGY_FIXTURE_FILES = {
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


def _adapt_fake_dataset(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dataset = getattr(
        request.module,
        "_FakeDataset",
        None,
    )
    if fake_dataset is None:
        return

    original_init = fake_dataset.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name, value in fake_dataset_topology_fields(
            int(self.num_branches)
        ).items():
            setattr(self, name, value)

    monkeypatch.setattr(
        fake_dataset,
        "__init__",
        compatible_init,
    )


def _adapt_generation_fakes(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = Path(str(request.node.fspath)).name
    if filename not in {
        "test_generation_api.py",
        "test_generation_policy_target.py",
    }:
        return

    from grid_topology_ai.self_play import generation

    original_generate = (
        generation.generate_self_play_examples
    )

    def compatible_generate(generation_request):
        action_space_config = type(
            TEST_ACTION_SPACE_CONFIG
        )(
            require_connected_after_switch=True,
            enable_cache=bool(
                generation_request.enable_cache
            ),
        )

        action_space_cls = generation.GridFMActionSpace
        if action_space_cls is not None:
            monkeypatch.setattr(
                action_space_cls,
                "config",
                action_space_config,
                raising=False,
            )

        writer_cls = generation.ExampleWriter
        if writer_cls is not None:
            parameters = inspect.signature(
                writer_cls.__init__
            ).parameters
            if "action_space_config" not in parameters:
                original_init = writer_cls.__init__

                def compatible_init(
                    self,
                    output_dir,
                    *,
                    physics_config,
                    action_space_config=action_space_config,
                ):
                    original_init(
                        self,
                        output_dir,
                        physics_config=physics_config,
                    )
                    self.action_space_config = (
                        action_space_config
                    )

                monkeypatch.setattr(
                    writer_cls,
                    "__init__",
                    compatible_init,
                )

        return original_generate(generation_request)

    monkeypatch.setattr(
        generation,
        "generate_self_play_examples",
        compatible_generate,
    )

    if hasattr(
        request.module,
        "generate_self_play_examples",
    ):
        monkeypatch.setattr(
            request.module,
            "generate_self_play_examples",
            compatible_generate,
        )


def _adapt_example_writer(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = Path(str(request.node.fspath)).name
    if filename != "test_exploration_metrics.py":
        return

    from grid_topology_ai.self_play.examples import (
        ExampleWriter,
    )

    original_init = ExampleWriter.__init__

    def compatible_init(
        self,
        output_dir,
        *,
        physics_config,
        action_space_config=TEST_ACTION_SPACE_CONFIG,
    ):
        original_init(
            self,
            output_dir,
            physics_config=physics_config,
            action_space_config=action_space_config,
        )

    monkeypatch.setattr(
        ExampleWriter,
        "__init__",
        compatible_init,
    )

    original_add_example = ExampleWriter.add_example

    def compatible_add_example(self, *args, **kwargs):
        if args:
            state = args[0]
        else:
            state = kwargs.get("state")

        if state is not None and not hasattr(
            state,
            "branch_ids",
        ):
            selected_branch_id = kwargs.get(
                "selected_branch_id"
            )
            branch_id = (
                0
                if selected_branch_id is None
                else int(selected_branch_id)
            )
            replacement = SimpleNamespace(
                branch_ids=np.array(
                    [branch_id],
                    dtype=np.int64,
                )
            )
            if args:
                args = (replacement, *args[1:])
            else:
                kwargs["state"] = replacement

        return original_add_example(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        ExampleWriter,
        "add_example",
        compatible_add_example,
    )


def _adapt_artifact_builders(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = request.module
    filename = Path(str(request.node.fspath)).name

    checkpoint_builder = getattr(
        module,
        "_checkpoint",
        None,
    )
    if filename == "test_artifact_contracts.py" and (
        checkpoint_builder is not None
    ):
        def current_checkpoint(*args, **kwargs):
            payload = dict(
                checkpoint_builder(*args, **kwargs)
            )
            payload.update(
                checkpoint_topology_fields((0,))
            )
            return payload

        monkeypatch.setattr(
            module,
            "_checkpoint",
            current_checkpoint,
        )

    if filename == "test_replay.py":
        original_physics_provenance = (
            module.physics_provenance
        )

        def replay_provenance(config):
            payload = dict(
                original_physics_provenance(config)
            )
            payload.update(topology_metadata((0,)))
            return payload

        monkeypatch.setattr(
            module,
            "physics_provenance",
            replay_provenance,
        )

        original_rows = module.rows

        def current_rows(*args, **kwargs):
            result = original_rows(*args, **kwargs)
            for row in result:
                row.update(topology_csv_fields((0,)))
            return result

        monkeypatch.setattr(
            module,
            "rows",
            current_rows,
        )

        if (
            request.node.name
            == "test_replay_manifest_physics_config_mismatch_is_rejected"
        ):
            original_save_manifest = (
                module.RollingReplayBuffer.save_manifest
            )

            def save_non_empty_manifest(self):
                if not self.buffer:
                    self.add_examples(
                        module.rows("seed", 1),
                        iteration=0,
                    )
                return original_save_manifest(self)

            monkeypatch.setattr(
                module.RollingReplayBuffer,
                "save_manifest",
                save_non_empty_manifest,
            )

    if filename == "test_example_validation.py":
        original_load = (
            module.load_and_validate_examples_csv
        )

        def reject_empty_before_schema(path):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return original_load(path)
            if frame.empty:
                raise ValueError(
                    f"Examples CSV is empty: {path}"
                )
            return original_load(path)

        monkeypatch.setattr(
            module,
            "load_and_validate_examples_csv",
            reject_empty_before_schema,
        )


def _adapt_legacy_mcts_fake_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_filename = Path(
        str(request.node.fspath)
    ).name
    if test_filename not in _LEGACY_MCTS_FAKE_ENV_FILES:
        return

    fake_env = getattr(
        request.module,
        "_FakeEnv",
        None,
    )
    if fake_env is None:
        raise AssertionError(
            f"{test_filename} must define _FakeEnv."
        )

    if hasattr(
        fake_env,
        "operational_action_mask",
    ):
        return

    valid_action_mask = getattr(
        fake_env,
        "valid_action_mask",
        None,
    )
    if valid_action_mask is None:
        raise AssertionError(
            f"{test_filename}._FakeEnv must expose an action mask."
        )

    monkeypatch.setattr(
        fake_env,
        "operational_action_mask",
        valid_action_mask,
        raising=False,
    )


@pytest.fixture(autouse=True)
def _current_artifact_contracts(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _adapt_legacy_mcts_fake_env(
        request,
        monkeypatch,
    )

    filename = Path(str(request.node.fspath)).name
    if filename not in _TOPOLOGY_FIXTURE_FILES:
        return

    original_savez = np.savez

    def current_savez(file, *args, **kwargs):
        if not args:
            kwargs = enrich_state_arrays(kwargs)
        return original_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez", current_savez)

    original_to_csv = pd.DataFrame.to_csv

    def current_to_csv(frame, path_or_buf=None, *args, **kwargs):
        markers = {
            "state_path",
            "mcts_policy_json",
            "physical_objective_schema_version",
            "physics_config_contract_version",
        }
        if markers.issubset(frame.columns):
            frame = frame.copy()
            for index, row in frame.iterrows():
                fields = topology_csv_fields(
                    branch_ids_from_state_path(
                        row["state_path"]
                    )
                )
                for name, value in fields.items():
                    if name not in frame.columns:
                        frame[name] = None
                    current = frame.at[index, name]
                    missing = current is None
                    if not missing:
                        observed_missing = pd.isna(current)
                        missing = isinstance(
                            observed_missing,
                            (bool, np.bool_),
                        ) and bool(observed_missing)

                    if missing:
                        frame.at[index, name] = value
                    elif (
                        name
                        in {
                            "topology_action_config",
                            "action_layout",
                        }
                        and not isinstance(current, str)
                    ):
                        frame.at[index, name] = json.dumps(
                            current,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
            return original_to_csv(
                frame,
                path_or_buf,
                *args,
                **kwargs,
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

    original_torch_save = torch.save

    def current_torch_save(obj, f, *args, **kwargs):
        if (
            isinstance(obj, Mapping)
            and "checkpoint_contract_version" in obj
        ):
            payload = dict(obj)
            num_branches = int(
                payload.get("num_branches", 1)
            )
            for name, value in checkpoint_topology_fields(
                range(num_branches)
            ).items():
                payload.setdefault(name, value)
            obj = payload
        return original_torch_save(
            obj,
            f,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        torch,
        "save",
        current_torch_save,
    )

    _adapt_fake_dataset(
        request,
        monkeypatch,
    )
    _adapt_generation_fakes(
        request,
        monkeypatch,
    )
    _adapt_example_writer(
        request,
        monkeypatch,
    )
    _adapt_artifact_builders(
        request,
        monkeypatch,
    )
