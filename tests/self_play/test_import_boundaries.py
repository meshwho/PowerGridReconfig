from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )


def test_training_api_imports_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
from grid_topology_ai.training.graph_policy_value import (
    TrainingRequest,
    train_graph_policy_value_model,
)
print(TrainingRequest.__name__)
print(train_graph_policy_value_model.__name__)
"""
    )

    _assert_success(result)


def test_training_cli_help_works_in_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.self_play.train_graph_baseline",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    _assert_success(result)
    assert "Train graph/GNN policy-value baseline" in result.stdout


def test_artifact_import_does_not_eagerly_load_pipeline_or_training() -> None:
    result = _run_fresh_python(
        """
import sys

from grid_topology_ai.self_play.artifacts import save_json

assert "grid_topology_ai.self_play.pipeline" not in sys.modules
assert "grid_topology_ai.self_play.iteration" not in sys.modules
assert "grid_topology_ai.self_play.stages" not in sys.modules
assert "grid_topology_ai.training.graph_policy_value" not in sys.modules

print(save_json.__name__)
"""
    )

    _assert_success(result)


def test_pipeline_api_still_imports_explicitly() -> None:
    result = _run_fresh_python(
        """
from grid_topology_ai.self_play.pipeline import (
    PipelineRequest,
    PipelineResult,
    run_self_play_pipeline,
)
print(PipelineRequest.__name__)
print(PipelineResult.__name__)
print(run_self_play_pipeline.__name__)
"""
    )

    _assert_success(result)


def test_self_play_storage_imports_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
from grid_topology_ai.self_play.examples import (
    ExampleWriter,
    SelfPlayExample,
)
from grid_topology_ai.self_play.replay import RollingReplayBuffer
print(ExampleWriter.__name__)
print(SelfPlayExample.__name__)
print(RollingReplayBuffer.__name__)
"""
    )

    _assert_success(result)


def test_legacy_self_play_storage_modules_are_absent_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
import importlib.util

assert importlib.util.find_spec(
    "grid_topology_ai.self_play." + "replay_buffer"
) is None
assert importlib.util.find_spec(
    "grid_topology_ai.self_play." + "replay_buffer" + "_v2"
) is None
print("legacy modules absent")
"""
    )

    _assert_success(result)


def test_scenario_pool_imports_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
from grid_topology_ai.self_play.pool_state import (
    initialize_pool_metadata,
    update_and_save_pool_metadata,
)
from grid_topology_ai.self_play.pool_sampling import sample_from_pool
print(
    initialize_pool_metadata.__name__,
    update_and_save_pool_metadata.__name__,
    sample_from_pool.__name__,
)
"""
    )

    _assert_success(result)


def test_legacy_pool_metadata_module_is_absent_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
import importlib.util

assert importlib.util.find_spec(
    "grid_topology_ai.self_play." + "pool_" + "metadata"
) is None
print("legacy pool metadata module absent")
"""
    )

    _assert_success(result)


def test_scenario_pool_static_boundaries() -> None:
    sampling_source = (PROJECT_ROOT / "grid_topology_ai/self_play/pool_sampling.py").read_text(encoding="utf-8")
    state_source = (PROJECT_ROOT / "grid_topology_ai/self_play/pool_state.py").read_text(encoding="utf-8")

    assert not any(token in sampling_source for token in ("pandas", "read_csv", "write_text", "save_json", "load_json"))
    assert "def sample_from_pool" not in state_source


def test_example_validation_import_is_lightweight_in_fresh_process() -> None:
    result = _run_fresh_python(
        """
import sys
from grid_topology_ai.self_play.example_validation import (
    REQUIRED_EXAMPLE_COLUMNS,
    load_and_validate_examples_csv,
    validate_examples_dataframe,
)
assert "torch" not in sys.modules
assert "grid_topology_ai.training.graph_policy_value" not in sys.modules
assert "grid_topology_ai.self_play.pipeline" not in sys.modules
print(REQUIRED_EXAMPLE_COLUMNS)
print(load_and_validate_examples_csv.__name__, validate_examples_dataframe.__name__)
"""
    )
    _assert_success(result)


def test_example_validation_static_boundaries() -> None:
    source = (PROJECT_ROOT / "grid_topology_ai/self_play/example_validation.py").read_text(encoding="utf-8")
    forbidden = ("torch", "scripts.", "subprocess", "GridFMPowerFlowBackend", "TopologySwitchingEnv", "MCTSPlanner")
    assert all(token not in source for token in forbidden)
    replay = (PROJECT_ROOT / "grid_topology_ai/self_play/replay.py").read_text(encoding="utf-8")
    assert "load_and_validate_examples_csv" in replay
    assert "required_columns = {" not in replay
