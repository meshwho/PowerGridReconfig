from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pypower.api import runopf
from pypower.idx_bus import VA, VM
from pypower.idx_cost import COST, MODEL, NCOST, SHUTDOWN, STARTUP
from pypower.idx_gen import GEN_STATUS, PG, QG

from grid_topology_ai.cache import solver_invocation_fingerprint
from grid_topology_ai.state import GridFMState
from grid_topology_ai.physics.constraints import (
    calculate_physical_metrics_from_result,
    validate_ppc_input,
)
from grid_topology_ai.physics.objective import (
    PhysicalStateAssessment,
    assess_physical_state,
)
from grid_topology_ai.power_flow import (
    InvalidPhysicalState,
    PowerFlowNotConverged,
)

if TYPE_CHECKING:
    from grid_topology_ai.power_flow.backend import GridFMPowerFlowBackend


@dataclass(frozen=True, slots=True)
class MinimalRedispatchResult:
    opf_success: bool
    assessment: PhysicalStateAssessment | None
    message: str
    redispatch_l1_mw: float | None = None
    redispatch_up_mw: float | None = None
    redispatch_down_mw: float | None = None
    redispatch_max_generator_delta_mw: float | None = None

    @property
    def validated(self) -> bool:
        return bool(
            self.opf_success
            and self.assessment is not None
            and self.assessment.physically_secure
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "redispatch_attempted": True,
            "redispatch_opf_success": bool(self.opf_success),
            "redispatch_validated": self.validated,
            "redispatch_l1_mw": self.redispatch_l1_mw,
            "redispatch_up_mw": self.redispatch_up_mw,
            "redispatch_down_mw": self.redispatch_down_mw,
            "redispatch_max_generator_delta_mw": (
                self.redispatch_max_generator_delta_mw
            ),
            "redispatch_message": self.message,
        }


_REDISPATCH_MEMO: dict[bytes, MinimalRedispatchResult] = {}
_REDISPATCH_MEMO_BACKEND: object | None = None
_REDISPATCH_MEMO_SCENARIO_ID: int | None = None


def empty_redispatch_diagnostics() -> dict[str, object]:
    return {
        "redispatch_attempted": False,
        "redispatch_opf_success": False,
        "redispatch_validated": False,
        "redispatch_l1_mw": None,
        "redispatch_up_mw": None,
        "redispatch_down_mw": None,
        "redispatch_max_generator_delta_mw": None,
        "redispatch_message": None,
    }


def _quadratic_redispatch_cost(
    reference_pg: np.ndarray,
    generator_status: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_pg, dtype=np.float64)
    status = np.asarray(generator_status, dtype=np.float64)

    if reference.ndim != 1 or status.shape != reference.shape:
        raise ValueError("Generator dispatch and status must be matching vectors.")
    if not np.isfinite(reference).all() or not np.isfinite(status).all():
        raise ValueError("Generator dispatch and status must be finite.")

    gencost = np.zeros((len(reference), COST + 3), dtype=np.float64)
    gencost[:, MODEL] = 2.0
    gencost[:, STARTUP] = 0.0
    gencost[:, SHUTDOWN] = 0.0
    gencost[:, NCOST] = 3.0

    active = status > 0.0
    gencost[active, COST] = 1.0
    gencost[active, COST + 1] = -2.0 * reference[active]
    gencost[active, COST + 2] = reference[active] ** 2
    return gencost


def _opf_case_from_baseline(
    ppc: dict[str, object],
    baseline_result: dict[str, object],
) -> dict[str, object]:
    opf_case: dict[str, object] = {
        "version": ppc["version"],
        "baseMVA": ppc["baseMVA"],
        "bus": np.array(ppc["bus"], dtype=np.float64, copy=True),
        "branch": np.array(ppc["branch"], dtype=np.float64, copy=True),
        "gen": np.array(ppc["gen"], dtype=np.float64, copy=True),
    }

    baseline_bus = np.asarray(baseline_result["bus"], dtype=np.float64)
    baseline_gen = np.asarray(baseline_result["gen"], dtype=np.float64)
    bus = np.asarray(opf_case["bus"])
    gen = np.asarray(opf_case["gen"])

    bus[:, VM] = baseline_bus[:, VM]
    bus[:, VA] = baseline_bus[:, VA]
    gen[:, PG] = baseline_gen[:, PG]
    gen[:, QG] = baseline_gen[:, QG]
    opf_case["gencost"] = _quadratic_redispatch_cost(
        baseline_gen[:, PG],
        baseline_gen[:, GEN_STATUS],
    )
    return opf_case


def _redispatch_magnitudes(
    baseline_gen: np.ndarray,
    redispatched_gen: np.ndarray,
) -> tuple[float, float, float, float]:
    baseline = np.asarray(baseline_gen, dtype=np.float64)
    redispatched = np.asarray(redispatched_gen, dtype=np.float64)
    active = baseline[:, GEN_STATUS] > 0.0
    delta = redispatched[active, PG] - baseline[active, PG]

    if delta.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    absolute = np.abs(delta)
    return (
        float(np.sum(absolute)),
        float(np.sum(np.maximum(delta, 0.0))),
        float(np.sum(np.maximum(-delta, 0.0))),
        float(np.max(absolute)),
    )


def run_minimal_ac_redispatch(
    backend: GridFMPowerFlowBackend,
    state: GridFMState,
) -> MinimalRedispatchResult:
    """Run one terminal AC OPF that minimizes deviation from handoff dispatch."""

    global _REDISPATCH_MEMO_BACKEND, _REDISPATCH_MEMO_SCENARIO_ID

    scenario_id = int(state.scenario_id)
    context = f"minimal redispatch scenario={scenario_id}"
    trusted_checker = getattr(backend, "_is_trusted_repeated_state", None)
    trusted_state = bool(trusted_checker(state)) if callable(trusted_checker) else False

    try:
        ppc, _ = backend._build_ppc_from_state(state)
    except (InvalidPhysicalState, PowerFlowNotConverged) as exc:
        return MinimalRedispatchResult(
            opf_success=False,
            assessment=None,
            message=f"Could not reconstruct handoff dispatch: {exc}",
        )

    cache_key: bytes | None = None
    if bool(getattr(backend, "enable_cache", False)):
        if (
            backend is not _REDISPATCH_MEMO_BACKEND
            or scenario_id != _REDISPATCH_MEMO_SCENARIO_ID
        ):
            _REDISPATCH_MEMO.clear()
            _REDISPATCH_MEMO_BACKEND = backend
            _REDISPATCH_MEMO_SCENARIO_ID = scenario_id

        cache_key = solver_invocation_fingerprint(
            backend._problem_from_ppc(ppc),
            physics_fingerprint=backend.physics_config.fingerprint(),
        )
        cached = _REDISPATCH_MEMO.get(cache_key)
        if cached is not None:
            return cached

    def remember(result: MinimalRedispatchResult) -> MinimalRedispatchResult:
        if cache_key is not None:
            _REDISPATCH_MEMO[cache_key] = result
        return result

    if getattr(state, "generator_ids", None) is not None:
        baseline_result = ppc
    else:
        try:
            if trusted_state:
                baseline_result, _ = backend._solve_ppc(
                    ppc,
                    context=f"{context} baseline",
                    validate_input=False,
                )
            else:
                baseline_result, _ = backend._solve_ppc(
                    ppc,
                    context=f"{context} baseline",
                )
        except (InvalidPhysicalState, PowerFlowNotConverged) as exc:
            return remember(
                MinimalRedispatchResult(
                    opf_success=False,
                    assessment=None,
                    message=f"Could not reconstruct handoff dispatch: {exc}",
                )
            )

    opf_case = _opf_case_from_baseline(ppc, baseline_result)
    if not trusted_state:
        validate_ppc_input(
            opf_case,
            backend.physics_config,
            context=context,
        )

    try:
        result_ppc = runopf(opf_case, backend._build_pp_options())
    except Exception as exc:
        return remember(
            MinimalRedispatchResult(
                opf_success=False,
                assessment=None,
                message=f"PYPOWER AC OPF failed: {type(exc).__name__}: {exc}",
            )
        )

    if not isinstance(result_ppc, dict) or not bool(result_ppc.get("success", False)):
        return remember(
            MinimalRedispatchResult(
                opf_success=False,
                assessment=None,
                message="PYPOWER AC OPF did not find a feasible solution.",
            )
        )

    metrics = calculate_physical_metrics_from_result(
        result_ppc,
        power_flow_converged=True,
        physics_config=backend.physics_config,
    )
    assessment = assess_physical_state(metrics)

    l1_mw, up_mw, down_mw, max_delta_mw = _redispatch_magnitudes(
        np.asarray(baseline_result["gen"], dtype=np.float64),
        np.asarray(result_ppc["gen"], dtype=np.float64),
    )

    message = (
        "AC redispatch satisfies the strict physical contract."
        if assessment.physically_secure
        else "AC OPF converged but the strict physical contract is not satisfied."
    )
    return remember(
        MinimalRedispatchResult(
            opf_success=True,
            assessment=assessment,
            message=message,
            redispatch_l1_mw=l1_mw,
            redispatch_up_mw=up_mw,
            redispatch_down_mw=down_mw,
            redispatch_max_generator_delta_mw=max_delta_mw,
        )
    )
