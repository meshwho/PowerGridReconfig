from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.search.root_policy import (
    constrain_policy,
    normalize_policy,
    require_action_in_policy_support,
    select_action_from_policy,
)


def test_normalize_policy_keeps_only_positive_support() -> None:
    assert normalize_policy({0: 0.0, 1: 2.0, 2: 1.0}) == pytest.approx(
        {1: 2.0 / 3.0, 2: 1.0 / 3.0}
    )


@pytest.mark.parametrize(
    ("policy", "error_type"),
    [
        ({}, ValueError),
        ({-1: 1.0}, ValueError),
        ({True: 1.0}, ValueError),
        ({1: -0.1}, ValueError),
        ({1: float("nan")}, ValueError),
        ({1: "1.0"}, TypeError),
        ({1: 0.0}, ValueError),
    ],
)
def test_normalize_policy_rejects_invalid_inputs(
    policy: dict[object, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        normalize_policy(policy)  # type: ignore[arg-type]


def test_constrain_policy_renormalizes_allowed_mass() -> None:
    assert constrain_policy(
        {0: 0.2, 1: 0.3, 2: 0.5},
        {0, 2},
    ) == pytest.approx({0: 2.0 / 7.0, 2: 5.0 / 7.0})


def test_constrain_policy_returns_empty_when_support_is_removed() -> None:
    assert constrain_policy({1: 0.7, 2: 0.3}, {0}) == {}


def test_zero_temperature_selects_policy_argmax() -> None:
    selected = select_action_from_policy(
        {1: 0.25, 2: 0.75},
        temperature=0.0,
        rng=np.random.default_rng(1),
    )
    assert selected == 2


def test_sampling_never_leaves_positive_policy_support() -> None:
    rng = np.random.default_rng(2)
    selected = {
        select_action_from_policy(
            {1: 0.6, 2: 0.4, 3: 0.0},
            temperature=1.0,
            rng=rng,
        )
        for _ in range(200)
    }
    assert selected <= {1, 2}
    assert selected


def test_require_action_in_policy_support_rejects_external_action() -> None:
    require_action_in_policy_support(1, {1: 0.7, 2: 0.3})

    with pytest.raises(ValueError, match="outside the support"):
        require_action_in_policy_support(0, {1: 0.7, 2: 0.3})


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_action_selection_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        select_action_from_policy(
            {1: 1.0},
            temperature=temperature,
            rng=np.random.default_rng(3),
        )
