from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_branch_loading_percent(branch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute approximate branch loading from gridfm branch data.

    gridfm branch data gives:
    - pf, qf: active/reactive power at from-side
    - pt, qt: active/reactive power at to-side
    - rate_a: branch MVA rating

    We estimate loading as:
        max(S_from, S_to) / rate_a * 100

    where:
        S_from = sqrt(pf^2 + qf^2)
        S_to   = sqrt(pt^2 + qt^2)
    """

    df = branch_df.copy()

    s_from = np.sqrt(df["pf"] ** 2 + df["qf"] ** 2)
    s_to = np.sqrt(df["pt"] ** 2 + df["qt"] ** 2)

    s_max = np.maximum(s_from, s_to)

    rate_a = df["rate_a"].replace(0, np.nan)

    df["s_from_mva"] = s_from
    df["s_to_mva"] = s_to
    df["s_max_mva"] = s_max
    df["loading_percent"] = s_max / rate_a * 100.0

    # If branch is out of service, its loading should not be treated as overload.
    if "br_status" in df.columns:
        df.loc[df["br_status"] <= 0, "loading_percent"] = 0.0

    return df


def analyze_scenarios(raw_dir: Path) -> pd.DataFrame:
    bus_path = raw_dir / "bus_data.parquet"
    branch_path = raw_dir / "branch_data.parquet"
    gen_path = raw_dir / "gen_data.parquet"

    if not bus_path.exists():
        raise FileNotFoundError(f"Missing file: {bus_path}")

    if not branch_path.exists():
        raise FileNotFoundError(f"Missing file: {branch_path}")

    if not gen_path.exists():
        raise FileNotFoundError(f"Missing file: {gen_path}")

    bus_df = pd.read_parquet(bus_path)
    branch_df = pd.read_parquet(branch_path)
    gen_df = pd.read_parquet(gen_path)

    branch_df = compute_branch_loading_percent(branch_df)

    scenario_ids = sorted(bus_df["scenario"].unique())

    rows = []

    for scenario in scenario_ids:
        b = bus_df[bus_df["scenario"] == scenario]
        br = branch_df[branch_df["scenario"] == scenario]
        g = gen_df[gen_df["scenario"] == scenario]

        in_service_branches = br[br["br_status"] > 0]
        out_of_service_branches = br[br["br_status"] <= 0]

        max_loading = float(in_service_branches["loading_percent"].max())
        mean_loading = float(in_service_branches["loading_percent"].mean())

        overloaded = in_service_branches[in_service_branches["loading_percent"] > 100.0]
        hard_overloaded = in_service_branches[in_service_branches["loading_percent"] > 120.0]

        min_vm = float(b["Vm"].min())
        max_vm = float(b["Vm"].max())

        # Use limits from the dataset itself.
        low_voltage = b[b["Vm"] < b["min_vm_pu"]]
        high_voltage = b[b["Vm"] > b["max_vm_pu"]]

        total_load_p = float(b["Pd"].sum())
        total_load_q = float(b["Qd"].sum())
        total_gen_p = float(g[g["in_service"] > 0]["p_mw"].sum())

        inactive_branch_ids = list(out_of_service_branches["idx"].astype(int).values)

        rows.append(
            {
                "scenario": int(scenario),
                "load_scenario_idx": b["load_scenario_idx"].iloc[0],
                "total_load_p_mw": total_load_p,
                "total_load_q_mvar": total_load_q,
                "total_gen_p_mw": total_gen_p,
                "max_loading_percent": max_loading,
                "mean_loading_percent": mean_loading,
                "num_overloaded_branches": int(len(overloaded)),
                "num_hard_overloaded_branches": int(len(hard_overloaded)),
                "min_vm_pu": min_vm,
                "max_vm_pu": max_vm,
                "num_low_voltage_buses": int(len(low_voltage)),
                "num_high_voltage_buses": int(len(high_voltage)),
                "num_outaged_branches": int(len(out_of_service_branches)),
                "outaged_branch_ids": inactive_branch_ids,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze gridfm-datakit scenarios for emergency topology switching."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to gridfm raw output directory.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save scenario summary CSV.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    summary = analyze_scenarios(raw_dir)

    print("=" * 100)
    print("GridFM scenario analysis")
    print("=" * 100)

    print(f"Raw directory: {raw_dir.resolve()}")
    print(f"Number of scenarios: {len(summary)}")

    print("\nScenario summary:")
    columns_to_show = [
        "scenario",
        "load_scenario_idx",
        "total_load_p_mw",
        "max_loading_percent",
        "num_overloaded_branches",
        "num_hard_overloaded_branches",
        "min_vm_pu",
        "max_vm_pu",
        "num_low_voltage_buses",
        "num_high_voltage_buses",
        "num_outaged_branches",
        "outaged_branch_ids",
    ]

    print(summary[columns_to_show].to_string(index=False))

    print("\nGlobal statistics:")
    print(f"Max loading overall:       {summary['max_loading_percent'].max():.2f} %")
    print(f"Average max loading:       {summary['max_loading_percent'].mean():.2f} %")
    print(f"Scenarios with overloads:  {(summary['num_overloaded_branches'] > 0).sum()}")
    print(f"Scenarios with hard overloads >120%: {(summary['num_hard_overloaded_branches'] > 0).sum()}")
    print(f"Scenarios with low voltage: {(summary['num_low_voltage_buses'] > 0).sum()}")
    print(f"Scenarios with high voltage: {(summary['num_high_voltage_buses'] > 0).sum()}")

    useful = summary[
        (summary["num_overloaded_branches"] > 0)
        & (summary["max_loading_percent"] >= 100.0)
        & (summary["max_loading_percent"] <= 250.0)
    ]

    print("\nUseful emergency scenarios for first MVP:")
    print(f"Count: {len(useful)}")

    if len(useful) > 0:
        print(
            useful[
                [
                    "scenario",
                    "max_loading_percent",
                    "num_overloaded_branches",
                    "min_vm_pu",
                    "num_outaged_branches",
                    "outaged_branch_ids",
                ]
            ].to_string(index=False)
        )

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path, index=False)
        print(f"\nSaved summary to: {save_path}")


if __name__ == "__main__":
    main()