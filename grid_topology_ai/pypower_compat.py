from __future__ import annotations

from typing import Any

import numpy as np
from pypower.api import runpf as _runpf
from pypower.idx_gen import GEN_BUS


class _IntegralGenBusMatrix(np.ndarray):
    """Keep PYPOWER's generator bus column usable as a NumPy index."""

    def __new__(cls, values: Any):
        return np.asarray(values, dtype=float).view(cls)

    def __array_finalize__(self, obj: object) -> None:
        del obj

    def __getitem__(self, key: object):
        value = super().__getitem__(key)

        if (
            isinstance(key, tuple)
            and len(key) >= 2
            and isinstance(key[1], (int, np.integer))
            and int(key[1]) == GEN_BUS
        ):
            if np.isscalar(value):
                return np.int64(value)
            return np.asarray(value, dtype=np.int64)

        return value


def _prepare_case(casedata: Any, ppopt: Any) -> Any:
    if not isinstance(casedata, dict):
        return casedata

    if not isinstance(ppopt, dict) or not bool(ppopt.get("ENFORCE_Q_LIMS", 0)):
        return casedata

    if "gen" not in casedata:
        return casedata

    prepared = dict(casedata)
    prepared["gen"] = _IntegralGenBusMatrix(casedata["gen"])
    return prepared


def runpf(
    casedata: Any = None,
    ppopt: Any = None,
    fname: str = "",
    solvedcase: str = "",
):
    """Run PYPOWER with integer generator-bus reads in Q-limit enforcement."""

    prepared = _prepare_case(casedata, ppopt)
    result, success = _runpf(prepared, ppopt, fname, solvedcase)

    if isinstance(result, dict) and isinstance(result.get("gen"), np.ndarray):
        result["gen"] = np.asarray(result["gen"], dtype=float)

    return result, success
