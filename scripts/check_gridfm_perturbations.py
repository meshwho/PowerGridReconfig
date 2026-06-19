from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ENTITY_ID_NAMES = {
    "branch",
    "branch_id",
    "branch_idx",
    "line",
    "line_id",
    "line_idx",
    "gen",
    "gen_id",
    "gen_idx",
    "generator",
    "generator_id",
    "generator_idx",
    "element",
    "element_id",
    "element_idx",
    "bus",
    "bus_id",
    "bus_idx",
    "index",
}

# Electrical branch parameters that may change with admittance perturbation.
ADMITTANCE_PARAMETER_NAMES = {
    "r",
    "x",
    "b",
    "g",
    "br_r",
    "br_x",
    "br_b",
    "branch_r",
    "branch_x",
    "branch_b",
    "tap",
    "ratio",
    "shift",
    "admittance",
    "susceptance",
    "conductance",
}

# Generator parameters that may expose generation/cost perturbation.
GENERATION_PARAMETER_NAMES = {
    "pg",
    "p_g",
    "pmin",
    "pmax",
    "qmin",
    "qmax",
    "cost",
    "cost0",
    "cost1",
    "cost2",
    "c0",
    "c1",
    "c2",
}


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_column(
    columns: list[str],
    accepted_names: set[str],
) -> str | None:
    for column in columns:
        if normalized_name(column) in accepted_names:
            return column

    return None


def is_candidate_parameter(column: str) -> tuple[bool, str]:
    name = normalized_name(column)

    if name in ADMITTANCE_PARAMETER_NAMES:
        return True, "admittance"

    if name in GENERATION_PARAMETER_NAMES or "cost" in name:
        return True, "generation"

    return False, ""


def check_file(path: Path) -> list[dict[str, object]]:
    df = pd.read_parquet(path)

    print("\n" + "=" * 100)
    print(f"FILE: {path.name}")
    print(f"ROWS: {len(df)}")
    print(f"COLUMNS: {list(df.columns)}")

    if df.empty:
        print("Result: empty file")
        return []

    scenario_column = find_column(
        list(df.columns),
        {"scenario", "scenario_id"},
    )

    if scenario_column is None:
        print("Result: no scenario column - probably static data")
        return []

    entity_column = find_column(
        list(df.columns),
        ENTITY_ID_NAMES,
    )

    numeric_columns = list(
        df.select_dtypes(include="number").columns
    )

    results: list[dict[str, object]] = []

    for column in numeric_columns:
        if column == scenario_column:
            continue

        is_candidate, category = is_candidate_parameter(column)

        if not is_candidate:
            continue

        if entity_column is not None and entity_column != column:
            value_counts = (
                df.groupby(entity_column, dropna=False)[column]
                .nunique(dropna=False)
            )

            changed_entities = int((value_counts > 1).sum())
            total_entities = int(len(value_counts))
            max_unique_values = int(value_counts.max())
        else:
            scenario_values = (
                df.groupby(scenario_column, dropna=False)[column]
                .mean()
            )

            changed_entities = int(scenario_values.nunique(dropna=False) > 1)
            total_entities = 1
            max_unique_values = int(
                scenario_values.nunique(dropna=False)
            )

        result = {
            "file": path.name,
            "category": category,
            "column": column,
            "entity_column": entity_column,
            "changed_entities": changed_entities,
            "total_entities": total_entities,
            "max_unique_values": max_unique_values,
        }

        results.append(result)

        status = "CHANGED" if changed_entities > 0 else "constant"

        print(
            f"{status:8s} "
            f"category={category:10s} "
            f"column={column!r} "
            f"changed={changed_entities}/{total_entities} "
            f"max_unique={max_unique_values}"
        )

    if not results:
        print(
            "No recognized perturbation parameter columns found "
            "in this parquet."
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect GridFM raw parquet files and check whether "
            "generation or admittance parameters vary between scenarios."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Merged GridFM raw directory.",
    )

    args = parser.parse_args()
    raw_dir: Path = args.raw_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in: {raw_dir}"
        )

    all_results: list[dict[str, object]] = []

    for path in parquet_files:
        all_results.extend(check_file(path))

    print("\n" + "=" * 100)
    print("PERTURBATION CHECK SUMMARY")
    print("=" * 100)

    for category in ("generation", "admittance"):
        category_results = [
            result
            for result in all_results
            if result["category"] == category
        ]

        changed_results = [
            result
            for result in category_results
            if int(result["changed_entities"]) > 0
        ]

        if changed_results:
            print(
                f"{category}: variation detected in "
                f"{len(changed_results)} parameter columns"
            )

            for result in changed_results:
                print(
                    f"  {result['file']} -> {result['column']}: "
                    f"{result['changed_entities']}/"
                    f"{result['total_entities']} entities changed"
                )
        elif category_results:
            print(
                f"{category}: relevant columns found, "
                "but no variation detected"
            )
        else:
            print(
                f"{category}: no suitable scenario-dependent "
                "parameter columns found"
            )


if __name__ == "__main__":
    main()