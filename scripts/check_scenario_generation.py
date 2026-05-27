from grid_topology_ai.config import GridConfig
from grid_topology_ai.scenario_generator import ScenarioGenerator


def main():
    config = GridConfig(
        network_name="case118",
        min_load_scale=1.05,
        max_load_scale=1.60,
        max_scenario_attempts=200,
        useful_scenario_min_loading_percent=100.0,
    )

    print("=" * 70)
    print("Grid topology AI - scenario generation check")
    print("=" * 70)

    generator = ScenarioGenerator(config)

    scenarios = generator.generate_many(num_scenarios=10)

    print("\nGenerated scenarios:")
    print(f"Count: {len(scenarios)}")

    if len(scenarios) == 0:
        print("\nNo useful emergency scenarios were generated.")
        print("Possible fixes:")
        print("1. Increase max_load_scale.")
        print("2. Increase target_base_line_loading_percent.")
        print("3. Lower useful_scenario_min_loading_percent.")
        return

    max_loadings = [
        scenario.metrics.max_line_loading_percent
        for scenario in scenarios
    ]

    overloaded_counts = [
        scenario.metrics.num_overloaded_lines
        for scenario in scenarios
    ]

    print(f"Min max-loading: {min(max_loadings):.2f} %")
    print(f"Max max-loading: {max(max_loadings):.2f} %")
    print(f"Avg max-loading: {sum(max_loadings) / len(max_loadings):.2f} %")

    print(f"Min overloaded lines: {min(overloaded_counts)}")
    print(f"Max overloaded lines: {max(overloaded_counts)}")
    print(f"Avg overloaded lines: {sum(overloaded_counts) / len(overloaded_counts):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()