from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pypower.idx_brch import BR_STATUS, PF, PT, RATE_A

import grid_topology_ai.search.dc_action_screener as dc_module
from grid_topology_ai.search.dc_action_screener import DCActionScreener
from grid_topology_ai.topology_actions import GridFMAction


def _branch_action(
    *,
    branch_id: int,
    branch_pos: int,
    target_status: int,
) -> GridFMAction:
    return GridFMAction(
        action_id=1 + branch_pos,
        action_type=(
            "switch_off_branch"
            if target_status == 0
            else "switch_on_branch"
        ),
        branch_id=branch_id,
        branch_pos=branch_pos,
        target_status=target_status,
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.seen_actions: list[GridFMAction] = []

    def _make_cache_key_from_state(
        self,
        state: object,
        *,
        action: GridFMAction,
    ) -> tuple:
        return (
            int(getattr(state, "scenario_id")),
            int(action.branch_id),
            int(action.target_status),
        )

    def _build_ppc_from_state(
        self,
        *,
        state: object,
        action: GridFMAction,
    ) -> tuple[dict[str, np.ndarray], dict]:
        del state
        self.seen_actions.append(action)

        width = max(BR_STATUS, RATE_A, PF, PT) + 1
        branch = np.zeros((2, width), dtype=float)
        branch[:, BR_STATUS] = 1.0
        branch[:, RATE_A] = 100.0
        branch[:, PF] = [50.0, 40.0]
        branch[:, PT] = [-50.0, -40.0]

        return {"branch": branch}, {}


def test_dc_screener_supports_both_branch_status_directions() -> None:
    screener = DCActionScreener()

    opening = _branch_action(
        branch_id=10,
        branch_pos=0,
        target_status=0,
    )
    closing = _branch_action(
        branch_id=20,
        branch_pos=1,
        target_status=1,
    )
    stop = GridFMAction(
        action_id=0,
        action_type="do_nothing",
    )

    assert screener.supports(opening) is True
    assert screener.supports(closing) is True
    assert screener.supports(stop) is False


def test_dc_screener_passes_full_closing_action_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    screener = DCActionScreener(enable_cache=False)
    state = SimpleNamespace(
        scenario_id=7,
        outaged_branch_ids=[20],
    )
    closing = _branch_action(
        branch_id=20,
        branch_pos=1,
        target_status=1,
    )

    monkeypatch.setattr(
        dc_module,
        "rundcpf",
        lambda ppc, options: (ppc, True),
    )

    score = screener.score_action(
        state=state,  # type: ignore[arg-type]
        action=closing,
        backend=backend,  # type: ignore[arg-type]
    )

    assert score.success is True
    assert backend.seen_actions == [closing]


def test_dc_cache_key_distinguishes_opening_and_closing() -> None:
    backend = _FakeBackend()
    state = SimpleNamespace(
        scenario_id=7,
        outaged_branch_ids=[20],
    )
    opening = _branch_action(
        branch_id=10,
        branch_pos=0,
        target_status=0,
    )
    closing = _branch_action(
        branch_id=20,
        branch_pos=1,
        target_status=1,
    )

    opening_key = DCActionScreener._make_cache_key(
        state=state,  # type: ignore[arg-type]
        action=opening,
        backend=backend,  # type: ignore[arg-type]
        policy_prior=0.1,
    )
    closing_key = DCActionScreener._make_cache_key(
        state=state,  # type: ignore[arg-type]
        action=closing,
        backend=backend,  # type: ignore[arg-type]
        policy_prior=0.9,
    )

    assert opening_key != closing_key
    assert opening_key[-1] == 0
    assert closing_key[-1] == 1


def test_dc_rank_actions_drops_only_unsupported_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    screener = DCActionScreener(enable_cache=False)
    state = SimpleNamespace(
        scenario_id=7,
        outaged_branch_ids=[20],
    )
    opening = _branch_action(
        branch_id=10,
        branch_pos=0,
        target_status=0,
    )
    closing = _branch_action(
        branch_id=20,
        branch_pos=1,
        target_status=1,
    )
    stop = GridFMAction(
        action_id=0,
        action_type="do_nothing",
    )

    monkeypatch.setattr(
        dc_module,
        "rundcpf",
        lambda ppc, options: (ppc, True),
    )

    ranked = screener.rank_actions(
        state=state,  # type: ignore[arg-type]
        actions=[stop, opening, closing],
        backend=backend,  # type: ignore[arg-type]
    )

    assert {action.action_id for action in ranked} == {1, 2}
    assert stop not in ranked
