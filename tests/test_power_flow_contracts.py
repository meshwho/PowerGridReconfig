from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_STATUS,
    F_BUS,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import BUS_I, VM, VMAX, VMIN
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PMAX,
    PMIN,
    QMAX,
    QMIN,
)

from grid_topology_ai.config.physics import (
    PhysicsConfig,
    QLimitPolicy,
    ZeroRateAPolicy,
)
from grid_topology_ai.physical_constraints import (
    validate_ppc_input,
    validate_pypower_result,
)
from grid_topology_ai.power_flow_errors import InvalidPhysicalState
from grid_topology_ai.pypower_backend import GridFMPowerFlowBackend


def _valid_ppc() -> dict[str, object]:
    bus = np.zeros((2, VMIN + 1), dtype=float)
    bus[:, BUS_I] = [10, 20]
    bus[:, VM] = 1.0
    bus[:, VMIN] = 0.95
    bus[:, VMAX] = 1.05

    branch = np.zeros((1, ANGMAX + 1), dtype=float)
    branch[0, F_BUS] = 10
    branch[0, T_BUS] = 20
    branch[0, RATE_A] = 100.0
    branch[0, BR_STATUS] = 1.0
    branch[0, ANGMIN] = -30.0
    branch[0, ANGMAX] = 30.0

    gen = np.zeros((1, PMIN + 1), dtype=float)
    gen[0, GEN_BUS] = 10
    gen[0, GEN_STATUS] = 1.0
    gen[0, PMIN] = 0.0
    gen[0, PMAX] = 100.0
    gen[0, QMIN] = -50.0
    gen[0, QMAX] = 50.0

    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus,
        "branch": branch,
        "gen": gen,
    }


def _valid_result(input_ppc: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(input_ppc)
    result["branch"] = np.pad(
        np.asarray(result["branch"]),
        ((0, 0), (0, QT + 1 - np.asarray(result["branch"]).shape[1])),
    )
    return result


def test_valid_input_and_result_satisfy_strict_contract() -> None:
    config = PhysicsConfig()
    ppc = _valid_ppc()
    result = _valid_result(ppc)

    validate_ppc_input(ppc, config, context="test input")
    validate_pypower_result(
        result,
        config,
        input_ppc=ppc,
        context="test result",
    )


def test_result_contract_requires_a_mapping() -> None:
    with pytest.raises(InvalidPhysicalState, match="result must be a mapping"):
        validate_pypower_result(
            [],  # type: ignore[arg-type]
            PhysicsConfig(),
            input_ppc=_valid_ppc(),
        )


def test_input_contract_requires_a_mapping() -> None:
    with pytest.raises(InvalidPhysicalState, match="ppc must be a mapping"):
        validate_ppc_input([], PhysicsConfig())  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [None, 1, "1", 3, "3"])
def test_input_contract_rejects_unsupported_matpower_version(
    version: object,
) -> None:
    ppc = _valid_ppc()
    ppc["version"] = version

    with pytest.raises(InvalidPhysicalState, match="MATPOWER version"):
        validate_ppc_input(ppc, PhysicsConfig())


@pytest.mark.parametrize(
    "base_mva",
    [None, "invalid", 0.0, -1.0, float("nan"), float("inf")],
)
def test_input_contract_rejects_invalid_base_mva(base_mva: object) -> None:
    ppc = _valid_ppc()
    ppc["baseMVA"] = base_mva

    with pytest.raises(InvalidPhysicalState, match="baseMVA"):
        validate_ppc_input(ppc, PhysicsConfig())


def test_input_contract_rejects_base_mva_config_mismatch() -> None:
    with pytest.raises(InvalidPhysicalState, match="disagrees with PhysicsConfig"):
        validate_ppc_input(_valid_ppc(), PhysicsConfig(base_mva=110.0))


@pytest.mark.parametrize("matrix_name", ["bus", "branch", "gen"])
def test_input_contract_requires_every_matrix(matrix_name: str) -> None:
    ppc = _valid_ppc()
    ppc.pop(matrix_name)

    with pytest.raises(InvalidPhysicalState, match=f"required {matrix_name}"):
        validate_ppc_input(ppc, PhysicsConfig())


@pytest.mark.parametrize(
    ("matrix_name", "columns"),
    [("bus", VMIN), ("branch", ANGMAX), ("gen", PMIN)],
)
def test_input_contract_rejects_malformed_matrices(
    matrix_name: str,
    columns: int,
) -> None:
    ppc = _valid_ppc()
    ppc[matrix_name] = np.zeros((1, columns), dtype=float)

    with pytest.raises(InvalidPhysicalState, match="must be 2D"):
        validate_ppc_input(ppc, PhysicsConfig())


def test_input_contract_rejects_non_numeric_matrix() -> None:
    ppc = _valid_ppc()
    ppc["bus"] = [["not-a-number"] * (VMIN + 1)]

    with pytest.raises(InvalidPhysicalState, match="bus is not numeric"):
        validate_ppc_input(ppc, PhysicsConfig())


@pytest.mark.parametrize(
    ("matrix_name", "row", "column", "value", "message"),
    [
        ("bus", 1, BUS_I, 10.0, "unique integral IDs"),
        ("bus", 1, BUS_I, 20.5, "unique integral IDs"),
        ("bus", 0, VM, float("nan"), "NaN or infinity"),
        ("branch", 0, F_BUS, 999.0, "unknown bus"),
        ("branch", 0, T_BUS, 20.5, "unknown bus"),
        ("branch", 0, BR_STATUS, 0.5, "status must be 0 or 1"),
        ("branch", 0, RATE_A, -1.0, "RATE_A is invalid"),
        ("branch", 0, ANGMIN, 31.0, "ANGMIN > ANGMAX"),
        ("gen", 0, GEN_BUS, 999.0, "unknown bus"),
        ("gen", 0, GEN_BUS, 10.5, "unknown bus"),
        ("gen", 0, GEN_STATUS, -1.0, "status must be 0 or 1"),
        ("gen", 0, PMIN, 101.0, "limits are inverted"),
        ("gen", 0, QMIN, 51.0, "limits are inverted"),
        ("bus", 0, VMIN, 1.06, "limits are inverted"),
    ],
)
def test_input_contract_rejects_invalid_matrix_semantics(
    matrix_name: str,
    row: int,
    column: int,
    value: float,
    message: str,
) -> None:
    ppc = _valid_ppc()
    matrix = np.asarray(ppc[matrix_name])
    matrix[row, column] = value

    with pytest.raises(InvalidPhysicalState, match=message):
        validate_ppc_input(ppc, PhysicsConfig())


def test_input_contract_rejects_disconnected_active_topology() -> None:
    ppc = _valid_ppc()
    np.asarray(ppc["branch"])[0, BR_STATUS] = 0.0

    with pytest.raises(InvalidPhysicalState, match="topology is disconnected"):
        validate_ppc_input(ppc, PhysicsConfig())


def test_zero_rate_a_policy_is_enforced_at_input_boundary() -> None:
    ppc = _valid_ppc()
    np.asarray(ppc["branch"])[0, RATE_A] = 0.0

    validate_ppc_input(
        ppc,
        PhysicsConfig(zero_rate_a_policy=ZeroRateAPolicy.UNLIMITED),
    )
    with pytest.raises(InvalidPhysicalState, match="RATE_A is invalid"):
        validate_ppc_input(
            ppc,
            PhysicsConfig(zero_rate_a_policy=ZeroRateAPolicy.ERROR),
        )


@pytest.mark.parametrize("matrix_name", ["bus", "branch", "gen"])
def test_result_contract_rejects_missing_matrix(matrix_name: str) -> None:
    ppc = _valid_ppc()
    result = _valid_result(ppc)
    result.pop(matrix_name)

    with pytest.raises(InvalidPhysicalState, match=f"{matrix_name} result"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


@pytest.mark.parametrize("matrix_name", ["bus", "branch", "gen"])
def test_result_contract_rejects_non_finite_matrix(matrix_name: str) -> None:
    ppc = _valid_ppc()
    result = _valid_result(ppc)
    np.asarray(result[matrix_name])[0, 0] = np.nan

    with pytest.raises(InvalidPhysicalState, match=f"{matrix_name} result"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


def test_result_contract_rejects_non_numeric_matrix() -> None:
    ppc = _valid_ppc()
    result = _valid_result(ppc)
    result["gen"] = [["not-a-number"] * (PMIN + 1)]

    with pytest.raises(InvalidPhysicalState, match="gen result is not numeric"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


def test_result_contract_rejects_missing_flow_columns() -> None:
    ppc = _valid_ppc()
    result = copy.deepcopy(ppc)

    with pytest.raises(InvalidPhysicalState, match="lacks flow columns"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


@pytest.mark.parametrize("matrix_name", ["branch", "gen"])
def test_result_contract_rejects_changed_row_count(matrix_name: str) -> None:
    ppc = _valid_ppc()
    result = _valid_result(ppc)
    matrix = np.asarray(result[matrix_name])
    extra_row = matrix[0].copy()
    if matrix_name == "branch":
        extra_row[BR_STATUS] = 0.0
    else:
        extra_row[GEN_STATUS] = 0.0
    result[matrix_name] = np.vstack([matrix, extra_row])

    with pytest.raises(InvalidPhysicalState, match="row count differs"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


def test_result_contract_rejects_changed_bus_row_count() -> None:
    ppc = _valid_ppc()
    result = _valid_result(ppc)

    extra_bus = np.asarray(result["bus"])[0].copy()
    extra_bus[BUS_I] = 30.0
    result["bus"] = np.vstack([np.asarray(result["bus"]), extra_bus])

    extra_branch = np.asarray(result["branch"])[0].copy()
    extra_branch[F_BUS] = 20.0
    extra_branch[T_BUS] = 30.0
    result["branch"] = np.vstack(
        [np.asarray(result["branch"]), extra_branch]
    )

    with pytest.raises(InvalidPhysicalState, match="bus row count differs"):
        validate_pypower_result(result, PhysicsConfig(), input_ppc=ppc)


def test_backend_builds_solver_options_from_physics_config() -> None:
    config = PhysicsConfig(
        pf_alg=4,
        pf_tolerance=1e-10,
        max_iterations=47,
        q_limit_policy=QLimitPolicy.VALIDATE_ONLY,
    )
    backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=config,
    )

    options = backend._build_pp_options()

    assert options["PF_DC"] is False
    assert options["PF_ALG"] == 4
    assert options["PF_TOL"] == 1e-10
    assert options["PF_MAX_IT"] == 47
    assert options["PF_MAX_IT_FD"] == 47
    assert options["PF_MAX_IT_GS"] == 47
    assert options["ENFORCE_Q_LIMS"] == 0


def test_backend_rejects_non_physics_config() -> None:
    with pytest.raises(TypeError, match="physics_config"):
        GridFMPowerFlowBackend(
            adapter=object(),  # type: ignore[arg-type]
            physics_config={"pf_alg": 3},  # type: ignore[arg-type]
        )


def test_backend_cache_key_includes_physics_fingerprint() -> None:
    state = SimpleNamespace(scenario_id=7, outaged_branch_ids=[5, 2])
    default_backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(),
    )
    custom_backend = GridFMPowerFlowBackend(
        adapter=object(),  # type: ignore[arg-type]
        physics_config=PhysicsConfig(overload_limit_percent=115.0),
    )

    default_key = default_backend._make_cache_key_from_state(state, 3)
    custom_key = custom_backend._make_cache_key_from_state(state, 3)

    assert default_key == (
        7,
        default_backend.physics_config.fingerprint(),
        (2, 3, 5),
    )
    assert custom_key != default_key
