from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pypower.idx_cost import COST
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.physics import redispatch
from grid_topology_ai.termination import TerminationReason
from tests.outcome_evidence_helpers import terminal_evidence


def _case() -> tuple[dict[str, object], dict[str, object]]:
    bus = np.zeros((1, 13), dtype=np.float64)
    branch = np.zeros((0, 13), dtype=np.float64)
    gen = np.zeros((2, 21), dtype=np.float64)
    gen[:, GEN_STATUS] = 1.0
    gen[:, PG] = [50.0, 30.0]
    gen[:, QG] = [5.0, -2.0]

    ppc = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus.copy(),
        "branch": branch.copy(),
        "gen": gen.copy(),
    }
    baseline = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus.copy(),
        "branch": branch.copy(),
        "gen": gen.copy(),
    }
    return ppc, baseline


class _Backend:
    physics_config = object()

    def __init__(self) -> None:
        self.ppc, self.baseline = _case()

    def _build_ppc_from_state(self, state):
        del state
        return self.ppc, {}

    def _solve_ppc(self, ppc, *, context: str):
        del ppc, context
        return self.baseline, {}

    def _build_pp_options(self):
        return {}


def _patch_validation(monkeypatch: pytest.MonkeyPatch, assessment) -> None:
    monkeypatch.setattr(redispatch, "validate_ppc_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        redispatch,
        "validate_pypower_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        redispatch,
        "calculate_physical_metrics_from_result",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        redispatch,
        "assess_physical_state",
        lambda metrics: assessment,
    )


def test_minimal_redispatch_uses_handoff_dispatch_and_reports_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure = terminal_evidence(TerminationReason.SOLVED).assessment
    assert secure is not None
    _patch_validation(monkeypatch, secure)

    captured: dict[str, object] = {}

    def fake_runopf(case, options):
        del options
        captured.update(case)
        result = {
            "version": "2",
            "baseMVA": 100.0,
            "bus": np.array(case["bus"], copy=True),
            "branch": np.array(case["branch"], copy=True),
            "gen": np.array(case["gen"], copy=True),
            "success": True,
        }
        result["gen"][:, PG] = [40.0, 40.0]
        return result

    monkeypatch.setattr(redispatch, "runopf", fake_runopf)

    result = redispatch.run_minimal_ac_redispatch(
        _Backend(),
        SimpleNamespace(scenario_id=5),
    )

    assert result.validated is True
    assert result.redispatch_l1_mw == pytest.approx(20.0)
    assert result.redispatch_up_mw == pytest.approx(10.0)
    assert result.redispatch_down_mw == pytest.approx(10.0)
    assert result.redispatch_max_generator_delta_mw == pytest.approx(10.0)

    gencost = np.asarray(captured["gencost"])
    assert gencost[0, COST : COST + 3] == pytest.approx([1.0, -100.0, 2500.0])
    assert gencost[1, COST : COST + 3] == pytest.approx([1.0, -60.0, 900.0])


def test_minimal_redispatch_keeps_infeasible_opf_unvalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure = terminal_evidence(TerminationReason.SOLVED).assessment
    assert secure is not None
    _patch_validation(monkeypatch, secure)
    monkeypatch.setattr(redispatch, "runopf", lambda case, options: {"success": False})

    result = redispatch.run_minimal_ac_redispatch(
        _Backend(),
        SimpleNamespace(scenario_id=5),
    )

    assert result.opf_success is False
    assert result.validated is False
    assert result.assessment is None


def test_minimal_redispatch_requires_strict_physical_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = terminal_evidence(
        TerminationReason.HANDOFF_TO_REDISPATCH_TEACHER
    ).assessment
    assert unsafe is not None
    _patch_validation(monkeypatch, unsafe)

    def fake_runopf(case, options):
        del options
        return {
            "version": "2",
            "baseMVA": 100.0,
            "bus": np.array(case["bus"], copy=True),
            "branch": np.array(case["branch"], copy=True),
            "gen": np.array(case["gen"], copy=True),
            "success": True,
        }

    monkeypatch.setattr(redispatch, "runopf", fake_runopf)

    result = redispatch.run_minimal_ac_redispatch(
        _Backend(),
        SimpleNamespace(scenario_id=5),
    )

    assert result.opf_success is True
    assert result.validated is False
    assert result.assessment is unsafe
