from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_topology_ai.config import GridConfig


@dataclass(frozen=True)
class LineLimitCalibrationReport:
    """
    Report describing how line limits were calibrated.

    This is useful for debugging and for scientific transparency.
    We should always be able to explain how the artificial operational
    limits were created.
    """

    target_base_loading_percent: float
    num_lines: int
    num_lines_changed: int
    min_old_max_i_ka: float
    max_old_max_i_ka: float
    min_new_max_i_ka: float
    max_new_max_i_ka: float


def calibrate_line_limits_from_base_case(
    net,
    config: GridConfig,
) -> LineLimitCalibrationReport:
    """
    Calibrate line current limits based on base-case line currents.

    The goal:
    Make the base-case loading of most lines approximately equal to
    config.target_base_line_loading_percent.

    Why this is needed:
    In many IEEE test cases, the original line thermal limits are very high.
    For example, max loading may be only 4-5%. Such a system is not useful
    for emergency topology switching dataset generation.

    Important:
    This function does NOT change the physical power flow itself.
    It only changes line current limits max_i_ka, which affects
    loading_percent and overload detection.

    Requirements:
    Run AC power flow before calling this function, because we need net.res_line.i_ka.
    """

    if not hasattr(net, "res_line") or "i_ka" not in net.res_line.columns:
        raise ValueError(
            "Line currents are not available. Run power flow before calibration."
        )

    if "max_i_ka" not in net.line.columns:
        raise ValueError("net.line does not contain max_i_ka column.")

    old_max_i_ka = net.line["max_i_ka"].to_numpy(dtype=float).copy()

    line_currents_ka = net.res_line["i_ka"].to_numpy(dtype=float)
    line_currents_ka = np.abs(np.nan_to_num(line_currents_ka, nan=0.0))

    target = config.target_base_line_loading_percent

    if target <= 0 or target >= 100:
        raise ValueError(
            "target_base_line_loading_percent must be between 0 and 100."
        )

    # Some lines may have almost zero current in the base case.
    # If we used their current directly, their new limit would become almost zero.
    # That would create artificial overloads too easily.
    #
    # Therefore, for near-zero-flow lines we use a reference current.
    positive_currents = line_currents_ka[line_currents_ka > 1e-6]

    if len(positive_currents) == 0:
        reference_current = config.min_line_max_i_ka * target / 100.0
    else:
        # 20th percentile is a conservative low-current reference.
        reference_current = float(np.percentile(positive_currents, 20))

    effective_currents = np.where(
        line_currents_ka > 1e-6,
        line_currents_ka,
        reference_current,
    )

    # Formula:
    # loading_percent = current / max_current * 100
    #
    # So:
    # max_current = current * 100 / target_loading
    new_max_i_ka = effective_currents * 100.0 / target

    # Safety clipping.
    new_max_i_ka = np.clip(
        new_max_i_ka,
        config.min_line_max_i_ka,
        config.max_line_max_i_ka,
    )

    net.line["max_i_ka"] = new_max_i_ka

    num_lines_changed = int(np.sum(np.abs(old_max_i_ka - new_max_i_ka) > 1e-12))

    return LineLimitCalibrationReport(
        target_base_loading_percent=target,
        num_lines=len(net.line),
        num_lines_changed=num_lines_changed,
        min_old_max_i_ka=float(np.nanmin(old_max_i_ka)),
        max_old_max_i_ka=float(np.nanmax(old_max_i_ka)),
        min_new_max_i_ka=float(np.nanmin(new_max_i_ka)),
        max_new_max_i_ka=float(np.nanmax(new_max_i_ka)),
    )