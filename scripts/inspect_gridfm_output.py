from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def print_separator(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def inspect_parquet_file(path: Path) -> None:
    print_separator(path.name)

    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)

    print(f"Path:    {path}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nHead:")
    print(df.head())

    # Try to detect scenario/sample columns.
    possible_scenario_columns = [
        "scenario",
        "scenario_id",
        "sample",
        "sample_id",
        "idx",
        "grid_id",
    ]

    found = [col for col in possible_scenario_columns if col in df.columns]

    if found:
        print("\nPossible scenario columns:")
        for col in found:
            print(f"  {col}: unique values = {df[col].nunique()}")
            print(f"  first values: {list(df[col].drop_duplicates().head(10))}")
    else:
        print("\nNo obvious scenario column found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect gridfm-datakit parquet output files."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to gridfm raw output directory, for example data/gridfm_smoke/case118_ieee/raw",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print_separator("Inspecting gridfm-datakit output")
    print(f"Raw directory: {raw_dir.resolve()}")

    files = [
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
        "stats.parquet",
        "y_bus_data.parquet",
    ]

    for file_name in files:
        inspect_parquet_file(raw_dir / file_name)


if __name__ == "__main__":
    main()