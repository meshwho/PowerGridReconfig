from grid_topology_ai.config import GridConfig
from grid_topology_ai.legacy_pandapower.limit_calibrator import calibrate_line_limits_from_base_case
from grid_topology_ai.metrics import compute_grid_metrics
from grid_topology_ai.network_factory import create_network, run_power_flow


def print_metrics(title, net, metrics):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Max line loading:              {metrics.max_line_loading_percent:.2f} %")
    print(f"Total line overload:           {metrics.total_line_overload_percent:.2f} %")
    print(f"Number of overloaded lines:    {metrics.num_overloaded_lines}")
    print(f"Number of hard overloads:      {metrics.num_hard_overloaded_lines}")
    print(f"Min bus voltage:               {metrics.min_vm_pu:.4f} pu")
    print(f"Max bus voltage:               {metrics.max_vm_pu:.4f} pu")
    print(f"Total voltage violation:       {metrics.total_voltage_violation_pu:.4f} pu")
    print(f"Number of voltage violations:  {metrics.num_voltage_violations}")
    print(f"Hard voltage violations:       {metrics.num_hard_voltage_violations}")
    print(f"Has soft violations:           {metrics.has_soft_violations}")
    print(f"Has hard violations:           {metrics.has_hard_violations}")
    print(f"Is acceptable:                 {metrics.is_acceptable}")

    print("\nWorst loaded lines:")
    print(
        net.res_line[["loading_percent", "i_ka", "p_from_mw", "q_from_mvar"]]
        .sort_values("loading_percent", ascending=False)
        .head(10)
    )

    print("\nWorst voltage buses:")
    print(
        net.res_bus[["vm_pu", "va_degree"]]
        .sort_values("vm_pu", ascending=True)
        .head(10)
    )


def main():
    config = GridConfig(network_name="case118")

    print("=" * 70)
    print("Grid topology AI - network check")
    print("=" * 70)

    print(f"Loading network: {config.network_name}")

    net = create_network(config.network_name)

    print("Network loaded successfully.")
    print(f"Number of buses: {len(net.bus)}")
    print(f"Number of lines: {len(net.line)}")
    print(f"Number of loads: {len(net.load)}")
    print(f"Number of generators: {len(net.gen)}")
    print(f"Number of external grids: {len(net.ext_grid)}")

    print("\nRunning raw AC power flow...")

    converged = run_power_flow(net)
    raw_metrics = compute_grid_metrics(net, config, converged)

    if not converged:
        print("Raw power flow did NOT converge.")
        return

    print_metrics("Raw base case metrics", net, raw_metrics)

    if config.calibrate_line_limits:
        print("\nCalibrating line limits...")

        report = calibrate_line_limits_from_base_case(net, config)

        print("\nCalibration report:")
        print(f"Target base loading:  {report.target_base_loading_percent:.2f} %")
        print(f"Lines changed:        {report.num_lines_changed}/{report.num_lines}")
        print(f"Old max_i_ka range:   {report.min_old_max_i_ka:.6f} - {report.max_old_max_i_ka:.6f}")
        print(f"New max_i_ka range:   {report.min_new_max_i_ka:.6f} - {report.max_new_max_i_ka:.6f}")

        print("\nRunning calibrated AC power flow...")

        converged = run_power_flow(net)
        calibrated_metrics = compute_grid_metrics(net, config, converged)

        if not converged:
            print("Calibrated power flow did NOT converge.")
            return

        print_metrics("Calibrated base case metrics", net, calibrated_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()