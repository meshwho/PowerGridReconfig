from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_topology_ai.action_space import GridFMActionSpace
from grid_topology_ai.config import GenerationConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.environment import TopologySwitchingEnv
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend
from grid_topology_ai.self_play import generation
from grid_topology_ai.self_play.generation import GenerationRequest
from scripts.self_play import generate as generation_cli


_LOADING_INDEX = BRANCH_FEATURE_COLUMNS.index("loading_percent")


class _StopAfterActionSpace(RuntimeError):
    pass


class _RuntimeStub:
    def __init__(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.args = args
        self.kwargs = kwargs


class _RewardStub:
    def compute(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(reward=0.0)


def _unsafe_metrics() -> dict[str, object]:
    return {
        "power_flow_converged": True,
        "all_values_finite": True,
        "topology_connected": True,
        "max_loading_percent": 120.0,
        "num_overloaded_branches": 1,
        "num_hard_overloaded_branches": 1,
        "total_thermal_overload_mva": 1.0,
        "num_outaged_branches": 0,
        "num_low_voltage_buses": 0,
        "num_high_voltage_buses": 0,
        "total_voltage_violation": 0.0,
        "num_generator_p_violations": 0,
        "total_generator_p_violation_mw": 0.0,
        "num_generator_q_violations": 0,
        "total_generator_q_violation_mvar": 0.0,
        "num_angle_difference_violations": 0,
        "total_angle_difference_violation_degrees": 0.0,
    }


def _state(
    *,
    branch_status: tuple[int, int],
    loadings: tuple[float, float] = (50.0, 0.0),
) -> SimpleNamespace:
    branch_features = np.zeros(
        (2, len(BRANCH_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    branch_features[:, _LOADING_INDEX] = loadings

    return SimpleNamespace(
        scenario_id=1,
        outaged_branch_ids=tuple(
            branch_id
            for branch_id, status in zip(
                (10, 20),
                branch_status,
            )
            if status == 0
        ),
        branch_ids=np.asarray(
            [10, 20],
            dtype=np.int64,
        ),
        branch_status=np.asarray(
            branch_status,
            dtype=np.int64,
        ),
        branch_features=branch_features,
        bus_features=np.zeros(
            (2, 1),
            dtype=np.float32,
        ),
        edge_index=np.asarray(
            [[0, 0], [1, 1]],
            dtype=np.int64,
        ),
        metrics=_unsafe_metrics(),
    )


class _ApplyingBackend:
    def __init__(self) -> None:
        self.actions: list[object] = []

    def run_power_flow_from_state(
        self,
        *,
        state: SimpleNamespace,
        action: object,
    ) -> SimpleNamespace:
        self.actions.append(action)
        branch_frame = pd.DataFrame(
            {
                "idx": state.branch_ids,
                "br_status": state.branch_status.astype(float),
            }
        )
        GridFMPowerFlowBackend._apply_branch_status(
            branch_frame,
            branch_id=int(action.branch_id),
            target_status=int(action.target_status),
            context="configured bidirectional pipeline test",
        )
        next_status = tuple(
            int(value)
            for value in branch_frame["br_status"].tolist()
        )
        return SimpleNamespace(
            success=True,
            next_state=_state(
                branch_status=next_status,
            ),
            message="ok",
        )


def test_cli_builds_canonical_bidirectional_generation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    examples_path = tmp_path / "examples.csv"

    def fake_generate(
        request: GenerationRequest,
    ) -> Path:
        captured["request"] = request
        return examples_path

    def fake_targets(
        path: Path,
        *,
        gamma: float,
    ) -> None:
        captured["targets"] = (path, gamma)

    monkeypatch.setattr(
        generation_cli,
        "generate_self_play_examples",
        fake_generate,
    )
    monkeypatch.setattr(
        generation_cli,
        "ensure_outcome_value_targets",
        fake_targets,
    )

    result = generation_cli.main(
        [
            str(tmp_path / "raw"),
            "--transitions",
            str(tmp_path / "transitions.csv"),
            "--closeable-branch-id",
            "20",
            "--closeable-branch-id",
            "10",
            "--min-loading-for-switch-percent",
            "37.5",
            "--no-require-connected-after-switch",
        ]
    )

    request = captured["request"]
    assert isinstance(request, GenerationRequest)
    assert result == 0
    assert request.config.require_connected_after_switch is False
    assert request.config.min_loading_for_switch_percent == pytest.approx(37.5)
    assert request.config.closeable_branch_ids == (10, 20)
    targets = captured["targets"]
    assert isinstance(targets, tuple)
    assert targets[0] == examples_path
    assert targets[1] == pytest.approx(0.95)


def test_self_play_constructs_action_space_from_typed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    transitions_path = tmp_path / "transitions.csv"
    transitions_path.write_text(
        "scenario_id\n1\n",
        encoding="utf-8",
    )

    class CapturingActionSpace:
        def __init__(
            self,
            **kwargs: object,
        ) -> None:
            captured_kwargs.update(kwargs)

    class StopReward:
        def __init__(
            self,
            **kwargs: object,
        ) -> None:
            raise _StopAfterActionSpace

    monkeypatch.setattr(
        generation,
        "_ensure_runtime_dependencies",
        lambda: None,
    )
    monkeypatch.setattr(
        generation,
        "GridFMAdapter",
        _RuntimeStub,
    )
    monkeypatch.setattr(
        generation,
        "GridFMPowerFlowBackend",
        _RuntimeStub,
    )
    monkeypatch.setattr(
        generation,
        "GridFMActionSpace",
        CapturingActionSpace,
    )
    monkeypatch.setattr(
        generation,
        "GridFMReward",
        StopReward,
    )

    config = GenerationConfig(
        max_steps=1,
        require_connected_after_switch=False,
        min_loading_for_switch_percent=37.5,
        closeable_branch_ids=(20, 10),
    )
    request = GenerationRequest(
        raw_dir=tmp_path / "raw",
        transitions_csv=transitions_path,
        output_dir=tmp_path / "out",
        checkpoint=None,
        config=config,
        mcts_seed=1,
        action_seed=2,
        clear_cache_between_scenarios=False,
    )

    with pytest.raises(_StopAfterActionSpace):
        generation.generate_self_play_examples(
            request
        )

    assert captured_kwargs == {
        "require_connected_after_switch": False,
        "min_loading_for_switch_percent": 37.5,
        "closeable_branch_ids": (10, 20),
        "enable_cache": True,
    }


def test_configured_closure_runs_from_mapping_through_env_and_backend() -> None:
    generation_config = GenerationConfig.from_mapping(
        {
            "require_connected_after_switch": False,
            "min_loading_for_switch_percent": 70.0,
            "closeable_branch_ids": [20],
        }
    )
    action_config = generation_config.action_space_config
    action_space = GridFMActionSpace(
        require_connected_after_switch=(
            action_config.require_connected_after_switch
        ),
        min_loading_for_switch_percent=(
            action_config.min_loading_for_switch_percent
        ),
        closeable_branch_ids=(
            action_config.closeable_branch_ids
        ),
        enable_cache=False,
    )
    initial_state = _state(
        branch_status=(1, 0),
    )
    backend = _ApplyingBackend()
    env = TopologySwitchingEnv(
        adapter=object(),
        backend=backend,
        action_space=action_space,
        reward_fn=_RewardStub(),
        max_steps=1,
    )
    env.current_state = initial_state
    env.initial_scenario_id = 1

    assert action_space.operational_action_mask(
        initial_state
    ).tolist() == [True, False, True]

    result = env.step(2)

    assert len(backend.actions) == 1
    action = backend.actions[0]
    assert action.action_id == 2
    assert action.action_type == "switch_on_branch"
    assert action.branch_id == 20
    assert action.target_status == 1
    assert result.next_state is not None
    assert result.next_state.branch_status.tolist() == [1, 1]
    assert result.next_state.outaged_branch_ids == ()
    assert result.info["applied_actions"] == [
        {
            "action_id": 2,
            "action_type": "switch_on_branch",
            "branch_id": 20,
            "target_status": 1,
        }
    ]
