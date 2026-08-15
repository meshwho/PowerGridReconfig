from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pypower.idx_brch import BR_STATUS, PF, PT, RATE_A

import grid_topology_ai.search.dc_action_screener as dc_module
from grid_topology_ai.cache import DCScreeningCache, dc_screening_fingerprint
from grid_topology_ai.power_flow_problem import CanonicalPowerFlowProblem
from grid_topology_ai.search.dc_action_screener import DCActionScreener
from grid_topology_ai.topology_actions import GridFMAction


def _problem(marker: float = 0.0) -> CanonicalPowerFlowProblem:
    bus = np.zeros((2, 13), dtype=np.float64)
    branch = np.zeros((1, 13), dtype=np.float64)
    gen = np.zeros((1, 21), dtype=np.float64)
    bus[0, 2] = 20.0 + marker
    branch[0, 3] = 0.1
    branch[0, BR_STATUS] = 1.0
    gen[0, 1] = 20.0
    return CanonicalPowerFlowProblem(
        base_mva=100.0,
        bus=bus,
        branch=branch,
        gen=gen,
    )


def _action() -> GridFMAction:
    return GridFMAction(
        action_id=1,
        action_type="switch_off_branch",
        branch_id=10,
        branch_pos=0,
        target_status=0,
    )


class _Backend:
    def __init__(self, problem: CanonicalPowerFlowProblem) -> None:
        self.problem = problem

    def _build_ppc_from_state(self, *, state, action):
        del state, action
        return self.problem.to_ppc(), {}


def test_dc_cache_fingerprint_is_exact() -> None:
    original = _problem()
    changed = _problem(1e-12)

    original_key = dc_screening_fingerprint(
        original,
        physics_fingerprint="physics-a",
    )
    assert dc_screening_fingerprint(
        _problem(),
        physics_fingerprint="physics-a",
    ) == original_key
    assert dc_screening_fingerprint(
        changed,
        physics_fingerprint="physics-a",
    ) != original_key
    assert dc_screening_fingerprint(
        original,
        physics_fingerprint="physics-b",
    ) != original_key


def test_dc_screening_cache_obeys_byte_budget() -> None:
    cache = DCScreeningCache(max_bytes=256)
    problem = _problem()
    key, cached = cache.lookup(problem, physics_fingerprint="physics-a")
    assert cached is None

    result = dc_module.CachedDCScreeningResult(
        success=True,
        max_loading_percent=80.0,
        num_overloaded=0,
        num_hard_overloaded=0,
        total_overload=0.0,
        hard_overload=0.0,
    )
    assert cache.store(key, result)
    assert cache.info()["bytes"] <= cache.info()["max_bytes"]


def test_dc_cache_does_not_cache_policy_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()
    backend = _Backend(problem)
    screener = DCActionScreener(
        enable_cache=True,
        policy_weight=10.0,
    )
    action = _action()
    calls = 0

    def fake_rundcpf(ppc, options):
        nonlocal calls
        del options
        calls += 1
        result = dict(ppc)
        branch = np.zeros((1, max(BR_STATUS, RATE_A, PF, PT) + 1), dtype=float)
        branch[:, BR_STATUS] = 1.0
        branch[:, RATE_A] = 100.0
        branch[:, PF] = 80.0
        branch[:, PT] = -80.0
        result["branch"] = branch
        return result, True

    monkeypatch.setattr(dc_module, "rundcpf", fake_rundcpf)

    first_policy = np.array([0.0, 0.2], dtype=np.float64)
    second_policy = np.array([0.0, 0.8], dtype=np.float64)
    first = screener.score_action(
        state=SimpleNamespace(),
        action=action,
        backend=backend,  # type: ignore[arg-type]
        neural_policy=first_policy,
    )
    second = screener.score_action(
        state=SimpleNamespace(),
        action=action,
        backend=backend,  # type: ignore[arg-type]
        neural_policy=second_policy,
    )

    assert calls == 1
    assert screener.cache_hits == 1
    assert first.policy_prior == 0.2
    assert second.policy_prior == 0.8
    assert first.penalty != second.penalty


def test_cached_and_uncached_dc_scores_match(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()
    backend = _Backend(problem)
    action = _action()

    def fake_rundcpf(ppc, options):
        del options
        result = dict(ppc)
        branch = np.zeros((1, max(BR_STATUS, RATE_A, PF, PT) + 1), dtype=float)
        branch[:, BR_STATUS] = 1.0
        branch[:, RATE_A] = 100.0
        branch[:, PF] = 125.0
        branch[:, PT] = -125.0
        result["branch"] = branch
        return result, True

    monkeypatch.setattr(dc_module, "rundcpf", fake_rundcpf)

    cached = DCActionScreener(enable_cache=True, policy_weight=2.0)
    uncached = DCActionScreener(enable_cache=False, policy_weight=2.0)
    policy = np.array([0.0, 0.4], dtype=np.float64)

    cached_score = cached.score_action(
        state=SimpleNamespace(),
        action=action,
        backend=backend,  # type: ignore[arg-type]
        neural_policy=policy,
    )
    uncached_score = uncached.score_action(
        state=SimpleNamespace(),
        action=action,
        backend=backend,  # type: ignore[arg-type]
        neural_policy=policy,
    )

    assert cached_score == uncached_score
