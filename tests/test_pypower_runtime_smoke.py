from __future__ import annotations

import numpy as np
from pypower.api import case9, ppoption, runpf


def _run_case9(pf_alg: int) -> None:
    result, success = runpf(
        case9(),
        ppoption(PF_ALG=pf_alg, VERBOSE=0, OUT_ALL=0),
    )

    assert success
    assert result["bus"].size > 0
    assert np.isfinite(result["bus"]).all()
    assert np.isfinite(result["branch"]).all()
    assert np.isfinite(result["gen"]).all()


def test_pypower_case9_runtime_with_numpy_1x() -> None:
    assert int(np.__version__.split(".")[0]) == 1

    _run_case9(pf_alg=1)
    _run_case9(pf_alg=3)
