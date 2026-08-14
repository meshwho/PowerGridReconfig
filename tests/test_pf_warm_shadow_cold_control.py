from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pypower.idx_bus import BUS_I, VA, VM

from grid_topology_ai import pf_warm_shadow_cold_control as cold_control
from grid_topology_ai.pf_warm_shadow import WarmCandidate
from grid_topology_ai.pf_warm_shadow_cold_control import (
    ColdControlWarmStartShadow,
)
from grid_topology_ai.pf_warm_shadow_runtime import BoundedWarmStartShadow
from scripts.self_play import generate_impact_teacher_redispatch_staged as staged


def test_cold_control_keeps_request_voltage_seed_unchanged(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}
    result_state = object()

    class FakeBackend:
        physics_config = object()

        @staticmethod
        def _build_pp_options():
            return {}

        @staticmethod
        def _build_state_from_pypower_result_fast(**kwargs):
            return result_state

    def fake_runpf(ppc, options):
        captured["bus"] = np.asarray(ppc["bus"]).copy()
        return ppc, True

    monkeypatch.setattr(cold_control, "runpf", fake_runpf)
    monkeypatch.setattr(cold_control, "validate_ppc_input", lambda *a, **k: None)
    monkeypatch.setattr(cold_control, "validate_pypower_result", lambda *a, **k: None)
    monkeypatch.setattr(
        cold_control,
        "calculate_physical_metrics_from_result",
        lambda *a, **k: {"power_flow_converged": True},
    )

    bus = np.zeros((2, 13), dtype=np.float64)
    bus[:, BUS_I] = [1, 2]
    bus[:, VM] = [1.015, 0.987]
    bus[:, VA] = [2.5, -7.25]
    original = bus.copy()

    shadow = ColdControlWarmStartShadow(
        FakeBackend(),
        SimpleNamespace(close=lambda: None),
        sample_rate=1.0,
    )
    state = SimpleNamespace(scenario_id=17)

    result = shadow._cold_state(state, {"bus": bus}, {})

    assert result is result_state
    np.testing.assert_array_equal(bus, original)
    np.testing.assert_allclose(captured["bus"][:, VM], original[:, VM])
    np.testing.assert_allclose(captured["bus"][:, VA], original[:, VA])
    assert shadow._cold_diagnostics["comparison_reference"] == "canonical_cold"


def test_cold_control_compares_cold_reference_with_global_warm(
    monkeypatch,
) -> None:
    cold_state = object()
    warm_state = object()
    authoritative_state = object()
    seen: dict[str, object] = {}

    shadow = ColdControlWarmStartShadow(
        SimpleNamespace(),
        SimpleNamespace(close=lambda: None),
        sample_rate=1.0,
    )
    shadow._cold_reference_state = cold_state
    shadow._cold_diagnostics = {
        "comparison_reference": "canonical_cold",
        "cold_path_seconds": 0.12,
    }

    def fake_compare(self, left, right):
        seen["left"] = left
        seen["right"] = right
        return {"shadow_success": True}

    monkeypatch.setattr(BoundedWarmStartShadow, "_compare", fake_compare)

    record = shadow._compare(authoritative_state, warm_state)

    assert seen["left"] is cold_state
    assert seen["right"] is warm_state
    assert record["comparison_reference"] == "canonical_cold"
    assert record["cold_path_seconds"] == 0.12
    assert record["authoritative_used_for_comparison"] is False


def test_cold_control_runs_reference_before_global_warm(monkeypatch) -> None:
    events: list[str] = []
    cold_state = object()
    warm_state = object()

    shadow = ColdControlWarmStartShadow(
        SimpleNamespace(),
        SimpleNamespace(close=lambda: None),
        sample_rate=1.0,
    )

    def fake_cold(state, ppc, frames):
        events.append("cold")
        return cold_state

    def fake_warm(self, state, ppc, frames, candidate):
        events.append("warm")
        return warm_state

    monkeypatch.setattr(shadow, "_cold_state", fake_cold)
    monkeypatch.setattr(BoundedWarmStartShadow, "_shadow_state", fake_warm)

    result = shadow._shadow_state(
        SimpleNamespace(),
        {},
        {},
        WarmCandidate(
            exact_key="1" * 64,
            topology_key="a" * 64,
            distance=0.1,
            state_payload=b"unused",
        ),
    )

    assert events == ["cold", "warm"]
    assert shadow._cold_reference_state is cold_state
    assert result is warm_state


def test_staged_worker_selects_cold_control_only_when_env_is_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeBackend:
        enable_cache = True
        _persistent_exact_cache = object()
        _warm_start_shadow = None

    class FakeShadow:
        def close(self) -> None:
            return None

    backend = FakeBackend()
    installed = FakeShadow()
    captured: dict[str, object] = {}

    def fake_cold_installer(backend_arg, cache_root, **kwargs):
        captured["backend"] = backend_arg
        captured["cache_root"] = cache_root
        captured.update(kwargs)
        backend_arg._warm_start_shadow = installed
        return installed

    def normal_installer_must_not_run(*args, **kwargs):
        raise AssertionError("normal warm-shadow installer was used")

    cache_root = tmp_path / "pf-cache"
    monkeypatch.setenv(staged._PF_CACHE_DIR_ENV, str(cache_root))
    monkeypatch.setenv(staged._PF_WARM_SHADOW_ENV, "1")
    monkeypatch.setenv(staged._PF_WARM_SHADOW_RATE_ENV, "0.5")
    monkeypatch.setenv(staged._PF_WARM_SHADOW_MAX_PAIRS_ENV, "200")
    monkeypatch.setenv(staged._PF_WARM_MAX_CANDIDATES_ENV, "8")
    monkeypatch.setenv(staged._PF_WARM_SHADOW_COLD_CONTROL_ENV, "1")
    monkeypatch.setattr(
        staged.redispatch.base.teacher,
        "_require_worker_context",
        lambda: {"backend": backend},
    )
    monkeypatch.setattr(
        staged,
        "install_runtime_warm_shadow",
        normal_installer_must_not_run,
    )
    monkeypatch.setattr(
        cold_control,
        "install_cold_control_warm_shadow",
        fake_cold_installer,
    )
    registered: list[object] = []
    monkeypatch.setattr(staged.atexit, "register", registered.append)

    staged._attach_persistent_pf_cache()

    assert captured["backend"] is backend
    assert captured["cache_root"] == str(cache_root)
    assert captured["sample_rate"] == 0.5
    assert captured["max_pairs"] == 200
    assert captured["max_candidates_per_topology"] == 8
    assert registered == [installed.close]
