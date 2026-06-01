from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def read_scenario_ids(transitions_path: Path) -> set[int]:
    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    df = pd.read_csv(transitions_path)

    if "scenario_id" not in df.columns:
        raise ValueError(f"{transitions_path} must contain scenario_id column.")

    return set(int(x) for x in df["scenario_id"].unique())


def try_filter_with_pyarrow(
    parquet_path: Path,
    output_path: Path,
    scenario_ids: set[int],
) -> bool:
    """
    Fast parquet filtering with pyarrow.

    Returns True if pyarrow filtering was used successfully.
    Returns False if pyarrow is unavailable or the file cannot be filtered this way.
    """

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception:
        return False

    try:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = parquet_file.schema_arrow.names

        if "scenario" not in schema_names:
            shutil.copy2(parquet_path, output_path)
            return True

        scenario_array = pa.array(sorted(int(x) for x in scenario_ids), type=pa.int64())

        tables = []

        for batch in parquet_file.iter_batches(batch_size=250_000):
            table = pa.Table.from_batches([batch])
            scenario_col = table["scenario"]

            if scenario_col.type != pa.int64():
                scenario_col = pc.cast(scenario_col, pa.int64())

            mask = pc.is_in(scenario_col, value_set=scenario_array)
            filtered = table.filter(mask)

            if filtered.num_rows > 0:
                tables.append(filtered)

        if tables:
            result = pa.concat_tables(tables, promote_options="default")
        else:
            result = parquet_file.schema_arrow.empty_table()

        pq.write_table(result, output_path, compression="snappy")
        return True

    except Exception as exc:
        print(f"pyarrow filtering failed for {parquet_path.name}: {exc}")
        return False


def filter_with_pandas(
    parquet_path: Path,
    output_path: Path,
    scenario_ids: set[int],
) -> None:
    """
    Fallback filtering with pandas.

    This reads the whole parquet file, so it is less memory efficient,
    but still only runs once per parquet file, not once per worker.
    """

    df = pd.read_parquet(parquet_path)

    if "scenario" in df.columns:
        before = len(df)
        df = df[df["scenario"].astype(int).isin(scenario_ids)].copy()
        after = len(df)
        print(f"  {parquet_path.name}: filtered {before} -> {after} rows")
    else:
        print(f"  {parquet_path.name}: no scenario column, copying full file")

    df.to_parquet(output_path, index=False)


def filter_parquet_file(
    parquet_path: Path,
    output_path: Path,
    scenario_ids: set[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {parquet_path.name}...")

    used_pyarrow = try_filter_with_pyarrow(
        parquet_path=parquet_path,
        output_path=output_path,
        scenario_ids=scenario_ids,
    )

    if not used_pyarrow:
        filter_with_pandas(
            parquet_path=parquet_path,
            output_path=output_path,
            scenario_ids=scenario_ids,
        )

    print(f"  saved: {output_path}")


def copy_non_parquet_files(
    raw_dir: Path,
    output_dir: Path,
) -> None:
    for src in raw_dir.iterdir():
        if src.is_file() and src.suffix.lower() != ".parquet":
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            print(f"Copied non-parquet file: {src.name}")


def verify_required_files(output_dir: Path) -> None:
    required = [
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
    ]

    missing = [
        name for name in required
        if not (output_dir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Subset raw directory is missing required files: {missing}"
        )


def print_subset_summary(output_dir: Path) -> None:
    bus_path = output_dir / "bus_data.parquet"
    branch_path = output_dir / "branch_data.parquet"

    bus = pd.read_parquet(bus_path, columns=["scenario"])
    branch = pd.read_parquet(branch_path, columns=["scenario"])

    print("\nSubset summary:")
    print(f"  bus rows:        {len(bus)}")
    print(f"  branch rows:     {len(branch)}")
    print(f"  bus scenarios:   {bus['scenario'].nunique()}")
    print(f"  branch scenarios:{branch['scenario'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a smaller GridFM raw subset by scenario_id."
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Original GridFM raw directory with parquet files.",
    )

    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV containing scenario_id column.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output raw subset directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    transitions_path = Path(args.transitions)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = read_scenario_ids(transitions_path)

    print("=" * 100)
    print("Extracting GridFM raw subset")
    print("=" * 100)
    print(f"Raw dir:        {raw_dir.resolve()}")
    print(f"Transitions:    {transitions_path.resolve()}")
    print(f"Output dir:     {output_dir.resolve()}")
    print(f"Scenarios:      {len(scenario_ids)}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found in raw_dir: {raw_dir}")

    for parquet_path in parquet_files:
        filter_parquet_file(
            parquet_path=parquet_path,
            output_path=output_dir / parquet_path.name,
            scenario_ids=scenario_ids,
        )

    copy_non_parquet_files(
        raw_dir=raw_dir,
        output_dir=output_dir,
    )

    verify_required_files(output_dir)
    print_subset_summary(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()