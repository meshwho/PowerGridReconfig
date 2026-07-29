from __future__ import annotations

from pathlib import Path

import pytest


_LEGACY_MCTS_FAKE_ENV_FILES = {
    "test_mcts_action_ranking.py",
    "test_mcts_dc_screening_depth.py",
    "test_mcts_off_prior_exploration.py",
}


@pytest.fixture(autouse=True)
def _adapt_legacy_mcts_fake_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep focused legacy MCTS fixtures aligned with the environment API.

    These three modules define minimal local ``_FakeEnv`` classes for
    ranking and exploration tests. Production MCTS now explicitly asks
    for the operational action mask, while the fixtures still expose the
    previous compatibility name only.
    """

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
