from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.idx_brch import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import BS, BUS_I, GS, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG
from pypower.int2ext import int2ext
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.pfsoln import pfsoln

from grid_topology_ai.pypower_newton_workspace import (
    PowerDerivativeWorkspace,
    newton_power_flow,
)


_BUS_NETWORK_COLUMNS = (BUS_I, GS, BS)
_BRANCH_NETWORK_COLUMNS = (
    F_BUS,
    T_BUS,
    BR_R,
    BR_X,
    BR_B,
    TAP,
    SHIFT,
    BR_STATUS,
)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=float, copy=True)
    copied.flags.writeable = False
    return copied


def _network_bus_view(bus: np.ndarray) -> np.ndarray:
    return np.asarray(bus[:, _BUS_NETWORK_COLUMNS], dtype=float)


def _network_branch_view(branch: np.ndarray) -> np.ndarray:
    return np.asarray(branch[:, _BRANCH_NETWORK_COLUMNS], dtype=float)


@dataclass(frozen=True)
class PreparedACNetwork:
    """Admittance matrices valid for one unchanged AC network structure."""

    base_mva: float
    bus_network: np.ndarray
    branch_network: np.ndarray
    ybus: Any
    yf: Any
    yt: Any
    derivatives: PowerDerivativeWorkspace

    @classmethod
    def build(
        cls,
        *,
        base_mva: float,
        bus: np.ndarray,
        branch: np.ndarray,
    ) -> "PreparedACNetwork":
        ybus, yf, yt = makeYbus(float(base_mva), bus, branch)
        return cls(
            base_mva=float(base_mva),
            bus_network=_readonly_copy(_network_bus_view(bus)),
            branch_network=_readonly_copy(_network_branch_view(branch)),
            ybus=ybus,
            yf=yf,
            yt=yt,
            derivatives=PowerDerivativeWorkspace.from_ybus(ybus),
        )

    def require_matches(
        self,
        *,
        base_mva: float,
        bus: np.ndarray,
        branch: np.ndarray,
    ) -> None:
        if float(base_mva) != self.base_mva:
            raise ValueError("Prepared AC network baseMVA does not match the case.")
        if not np.array_equal(
            _network_bus_view(bus),
            self.bus_network,
            equal_nan=True,
        ):
            raise ValueError("Prepared AC network bus admittance data changed.")
        if not np.array_equal(
            _network_branch_view(branch),
            self.branch_network,
            equal_nan=True,
        ):
            raise ValueError("Prepared AC network branch data changed.")


def _prepare_internal_case(casedata: Any) -> dict[str, Any]:
    ppc = loadcase(deepcopy(casedata))
    if not isinstance(ppc, dict):
        raise ValueError("PYPOWER could not load the supplied case.")

    branch = np.asarray(ppc["branch"])
    if branch.shape[1] < QT:
        missing = QT - branch.shape[1] + 1
        ppc["branch"] = np.c_[
            branch,
            np.zeros((branch.shape[0], missing), dtype=float),
        ]

    return ext2int(ppc)


def solve_newton_power_flow(
    casedata: Any,
    options: dict[str, Any],
    *,
    prepared_network: PreparedACNetwork | None = None,
) -> tuple[dict[str, Any], bool, PreparedACNetwork]:
    """Run one stock-equivalent Newton AC solve with optional admittance reuse."""

    if bool(options["PF_DC"]) or int(options["PF_ALG"]) != 1:
        raise ValueError("Prepared AC network solve supports Newton AC power flow only.")

    started = perf_counter()
    ppc = _prepare_internal_case(casedata)
    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)

    ref, pv, pq = bustypes(bus, gen)
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    v0 = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])
    voltage_controlled = np.ones(v0.shape)
    voltage_controlled[pq] = 0
    controlled_generators = np.flatnonzero(voltage_controlled[gbus])
    if len(controlled_generators):
        controlled_buses = gbus[controlled_generators]
        v0[controlled_buses] = (
            gen[on[controlled_generators], VG]
            / np.abs(v0[controlled_buses])
            * v0[controlled_buses]
        )

    if prepared_network is None:
        prepared_network = PreparedACNetwork.build(
            base_mva=base_mva,
            bus=bus,
            branch=branch,
        )
    else:
        prepared_network.require_matches(
            base_mva=base_mva,
            bus=bus,
            branch=branch,
        )

    sbus = makeSbus(base_mva, bus, gen)
    voltage, success, _iterations = newton_power_flow(
        prepared_network.ybus,
        sbus,
        v0,
        ref,
        pv,
        pq,
        options,
        derivative_workspace=prepared_network.derivatives,
    )
    bus, gen, branch = pfsoln(
        base_mva,
        bus,
        gen,
        branch,
        prepared_network.ybus,
        prepared_network.yf,
        prepared_network.yt,
        voltage,
        ref,
        pv,
        pq,
    )

    ppc["bus"] = bus
    ppc["gen"] = gen
    ppc["branch"] = branch
    ppc["success"] = bool(success)
    ppc["et"] = perf_counter() - started

    results = int2ext(ppc)
    off_gen = results["order"]["gen"]["status"]["off"]
    if len(off_gen):
        results["gen"][np.ix_(off_gen, [PG, QG])] = 0

    off_branch = results["order"]["branch"]["status"]["off"]
    if len(off_branch):
        results["branch"][np.ix_(off_branch, [PF, QF, PT, QT])] = 0

    return results, bool(success), prepared_network
