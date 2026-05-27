from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from grid_topology_ai.config import GridConfig


@dataclass(frozen=True)
class GridMetrics:


    converged: bool

    # Line loading metrics
    max_line_loading_percent: float
    total_line_overload_percent: float
    num_overloaded_lines: int
    num_hard_overloaded_lines: int

    # Voltage metrics
    min_vm_pu: float
    max_vm_pu: float
    total_voltage_violation_pu: float
    num_voltage_violations: int
    num_hard_voltage_violations: int

    # General flags
    has_soft_violations: bool
    has_hard_violations: bool
    is_acceptable: bool


def compute_grid_metrics(net, config: GridConfig, converged: bool) -> GridMetrics:
    """
    Compute metrics for the current pandapower network state.

    Parameters
    ----------
    net:
        pandapower network after power flow.

    config:
        Project configuration with voltage and loading limits.

    converged:
        Whether AC power flow converged.

    Returns
    -------
    GridMetrics
        A compact numerical description of the grid state.

    Important:
    If power flow did not converge, returns safe placeholder values
    that clearly indicate a bad state.
    """

    if not converged:
        return GridMetrics(
            converged=False,
            max_line_loading_percent=1e6,
            total_line_overload_percent=1e6,
            num_overloaded_lines=10**6,
            num_hard_overloaded_lines=10**6,
            min_vm_pu=0.0,
            max_vm_pu=1e6,
            total_voltage_violation_pu=1e6,
            num_voltage_violations=10**6,
            num_hard_voltage_violations=10**6,
            has_soft_violations=True,
            has_hard_violations=True,
            is_acceptable=False,
        )

    # -----------------------------
    # Line loading metrics
    # -----------------------------

    if len(net.res_line) > 0:
        line_loading = net.res_line["loading_percent"].to_numpy(dtype=float)
    else:
        line_loading = np.array([], dtype=float)

    if line_loading.size > 0:
        max_line_loading = float(np.nanmax(line_loading))
    else:
        max_line_loading = 0.0

    # Soft overload means loading above 100%.
    soft_line_overload = np.maximum(
        line_loading - config.line_loading_soft_limit_percent,
        0.0,
    )

    # Hard overload means loading above 120%.
    hard_line_overload = line_loading > config.line_loading_hard_limit_percent

    total_line_overload = float(np.nansum(soft_line_overload))
    num_overloaded_lines = int(np.sum(line_loading > config.line_loading_soft_limit_percent))
    num_hard_overloaded_lines = int(np.sum(hard_line_overload))

    # -----------------------------
    # Voltage metrics
    # -----------------------------

    if len(net.res_bus) > 0:
        vm = net.res_bus["vm_pu"].to_numpy(dtype=float)
    else:
        vm = np.array([], dtype=float)

    if vm.size > 0:
        min_vm = float(np.nanmin(vm))
        max_vm = float(np.nanmax(vm))
    else:
        min_vm = 0.0
        max_vm = 0.0

    # Soft voltage violation:
    # below 0.95 or above 1.05.
    low_voltage_violation = np.maximum(config.vm_min_soft_pu - vm, 0.0)
    high_voltage_violation = np.maximum(vm - config.vm_max_soft_pu, 0.0)

    total_voltage_violation = float(
        np.nansum(low_voltage_violation + high_voltage_violation)
    )

    soft_voltage_mask = (vm < config.vm_min_soft_pu) | (vm > config.vm_max_soft_pu)
    hard_voltage_mask = (vm < config.vm_min_hard_pu) | (vm > config.vm_max_hard_pu)

    num_voltage_violations = int(np.sum(soft_voltage_mask))
    num_hard_voltage_violations = int(np.sum(hard_voltage_mask))

    # -----------------------------
    # Global flags
    # -----------------------------

    has_soft_violations = (
        num_overloaded_lines > 0
        or num_voltage_violations > 0
    )

    has_hard_violations = (
        num_hard_overloaded_lines > 0
        or num_hard_voltage_violations > 0
    )

    # Acceptable means:
    # - power flow converged;
    # - no hard violations.
    #
    # Soft violations are allowed but penalized.
    is_acceptable = converged and not has_hard_violations

    return GridMetrics(
        converged=True,
        max_line_loading_percent=max_line_loading,
        total_line_overload_percent=total_line_overload,
        num_overloaded_lines=num_overloaded_lines,
        num_hard_overloaded_lines=num_hard_overloaded_lines,
        min_vm_pu=min_vm,
        max_vm_pu=max_vm,
        total_voltage_violation_pu=total_voltage_violation,
        num_voltage_violations=num_voltage_violations,
        num_hard_voltage_violations=num_hard_voltage_violations,
        has_soft_violations=has_soft_violations,
        has_hard_violations=has_hard_violations,
        is_acceptable=is_acceptable,
    )