from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from grid_topology_ai.actions import (
    ActionSpaceConfig,
    GridFMActionSpace,
)
from grid_topology_ai.cache import structural_topology_fingerprint
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS


def _state(
    *,
    loadings: list[float],
    statuses: list[int] | None = None,
    scenario_id: int = 1,
    outaged_branch_ids: tuple[int, ...] = (),
) -> SimpleNamespace:
    """
    Build the minimal state required by the loading filter.

    Connectivity checks are disabled in these tests, so bus_features
    and edge_index are intentionally not needed by the action logic.
    """
    branch_count = len(loadings)

    if statuses is None:
        statuses = [1] * branch_count

    if len(statuses) != branch_count:
        raise ValueError(
            "statuses and loadings must have the same length"
        )

    branch_features = np.zeros(
        (
            branch_count,
            len(BRANCH_FEATURE_COLUMNS),
        ),
        dtype=np.float32,
    )

    loading_column_idx = (
        BRANCH_FEATURE_COLUMNS.index(
            "loading_percent"
        )
    )

    branch_features[
        :,
        loading_column_idx,
    ] = np.asarray(
        loadings,
        dtype=np.float32,
    )

    return SimpleNamespace(
        scenario_id=int(scenario_id),
        outaged_branch_ids=tuple(
            int(branch_id)
            for branch_id in outaged_branch_ids
        ),
        branch_ids=np.arange(
            100,
            100 + branch_count,
            dtype=np.int64,
        ),
        branch_status=np.asarray(
            statuses,
            dtype=np.int8,
        ),
        branch_features=branch_features,
    )


@pytest.mark.parametrize(
    ("loading", "expected_switch_allowed"),
    [
        (69.9, False),
        (70.0, True),
        (120.0, True),
    ],
)
def test_nonzero_loading_threshold_filters_branch_actions(
    loading: float,
    expected_switch_allowed: bool,
) -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=False,
    )

    mask = action_space.operational_action_mask(
        _state(
            loadings=[loading],
        )
    )

    assert mask.tolist() == [
        True,
        expected_switch_allowed,
    ]


def test_inactive_branch_is_rejected_above_loading_threshold() -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=False,
    )

    mask = action_space.operational_action_mask(
        _state(
            loadings=[120.0],
            statuses=[0],
        )
    )

    assert mask.tolist() == [
        True,
        False,
    ]


def test_loading_column_comes_from_central_feature_schema() -> None:
    action_space = GridFMActionSpace()

    assert action_space._loading_column_idx == (
        BRANCH_FEATURE_COLUMNS.index(
            "loading_percent"
        )
    )


def test_action_space_config_is_frozen_and_hashable() -> None:
    config = ActionSpaceConfig(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70,
        enable_cache=True,
    )

    assert config.min_loading_for_switch_percent == 70.0
    assert isinstance(hash(config), int)

    with pytest.raises(FrozenInstanceError):
        config.min_loading_for_switch_percent = 10.0  # type: ignore[misc]


def test_action_space_properties_are_read_only() -> None:
    action_space = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )

    assert action_space.config == ActionSpaceConfig(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )
    assert action_space.require_connected_after_switch is False
    assert action_space.min_loading_for_switch_percent == 70.0
    assert action_space.enable_cache is True

    with pytest.raises(AttributeError):
        action_space.min_loading_for_switch_percent = 10.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "threshold",
    [
        True,
        -1.0,
        float("nan"),
        float("inf"),
        "not-a-number",
        None,
    ],
)
def test_invalid_loading_threshold_is_rejected(
    threshold: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "min_loading_for_switch_percent must be "
            "a finite non-negative number"
        ),
    ):
        GridFMActionSpace(
            min_loading_for_switch_percent=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "require_connected_after_switch",
            1,
        ),
        (
            "enable_cache",
            1,
        ),
    ],
)
def test_boolean_config_fields_reject_integer_values(
    field_name: str,
    field_value: object,
) -> None:
    kwargs: dict[str, object] = {
        field_name: field_value,
    }

    with pytest.raises(
        ValueError,
        match="must be a boolean",
    ):
        GridFMActionSpace(
            **kwargs,  # type: ignore[arg-type]
        )


def test_structural_cache_key_excludes_operational_loading_filter() -> None:
    state = _state(
        loadings=[60.0],
        scenario_id=5,
        outaged_branch_ids=(7, 2),
    )
    state.bus_features = np.zeros((2, 1), dtype=np.float32)
    state.edge_index = np.asarray([[0], [1]], dtype=np.int64)

    unfiltered = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=0.0,
        enable_cache=True,
    )
    filtered = GridFMActionSpace(
        require_connected_after_switch=False,
        min_loading_for_switch_percent=70.0,
        enable_cache=True,
    )

    unfiltered_key = structural_topology_fingerprint(
        state,
        require_connected_after_switch=unfiltered.require_connected_after_switch,
        closeable_branch_ids=unfiltered.closeable_branch_ids,
    )
    filtered_key = structural_topology_fingerprint(
        state,
        require_connected_after_switch=filtered.require_connected_after_switch,
        closeable_branch_ids=filtered.closeable_branch_ids,
    )

    assert unfiltered_key == filtered_key
    assert unfiltered.operational_action_mask(state).tolist() == [True, True]
    assert filtered.operational_action_mask(state).tolist() == [True, False]
