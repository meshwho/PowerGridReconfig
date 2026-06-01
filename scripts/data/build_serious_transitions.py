from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

def now() -> float:
    return time.perf_counter()


def print_time(label: str, start: float) -> None:
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f} s")


def read_parquet_columns(
    path: Path,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read only required/optional columns from parquet.

    This is faster and uses less RAM than reading the whole file.
    """

    optional_columns = optional_columns or []

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # First read only metadata if possible.
    # pandas itself will validate requested columns.
    columns = list(dict.fromkeys(required_columns + optional_columns))

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        # Fallback: read full parquet only if column projection failed.
        df = pd.read_parquet(path)

        missing_required = set(required_columns) - set(df.columns)

        if missing_required:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_required)}"
            )

        available = [col for col in columns if col in df.columns]

        return df[available].copy()


def compute_branch_summary(branch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized branch summary per scenario.

    Produces:
        max_loading_percent
        mean_loading_percent
        num_overloaded_branches
        num_hard_overloaded_branches
        num_outaged_branches
        outaged_branch_ids
    """

    df = branch_df.copy()

    required = {
        "scenario",
        "idx",
        "br_status",
        "pf",
        "qf",
        "pt",
        "qt",
        "rate_a",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"branch_data.parquet missing columns: {sorted(missing)}")

    df["scenario"] = df["scenario"].astype(np.int64)
    df["idx"] = df["idx"].astype(np.int64)

    br_status = df["br_status"].to_numpy(dtype=np.float32)
    in_service = br_status > 0.0

    pf = df["pf"].to_numpy(dtype=np.float64)
    qf = df["qf"].to_numpy(dtype=np.float64)
    pt = df["pt"].to_numpy(dtype=np.float64)
    qt = df["qt"].to_numpy(dtype=np.float64)

    rate_a = df["rate_a"].replace(0.0, np.nan).to_numpy(dtype=np.float64)

    s_from = np.sqrt(pf * pf + qf * qf)
    s_to = np.sqrt(pt * pt + qt * qt)
    s_max = np.maximum(s_from, s_to)

    loading = s_max / rate_a * 100.0
    loading = np.where(in_service, loading, 0.0)
    loading = np.nan_to_num(loading, nan=0.0, posinf=0.0, neginf=0.0)

    df["loading_percent"] = loading.astype(np.float32)
    df["is_in_service"] = in_service
    df["is_outaged"] = ~in_service
    df["is_overloaded"] = in_service & (df["loading_percent"].to_numpy() > 100.0)
    df["is_hard_overloaded"] = in_service & (
        df["loading_percent"].to_numpy() > 120.0
    )

    active = df[df["is_in_service"]].copy()

    # If some scenario has no active branches, it will be filled with 0 later.
    active_loading_summary = active.groupby("scenario", sort=False).agg(
        max_loading_percent=("loading_percent", "max"),
        mean_loading_percent=("loading_percent", "mean"),
    )

    count_summary = df.groupby("scenario", sort=False).agg(
        num_overloaded_branches=("is_overloaded", "sum"),
        num_hard_overloaded_branches=("is_hard_overloaded", "sum"),
        num_outaged_branches=("is_outaged", "sum"),
    )

    outaged = df[df["is_outaged"]][["scenario", "idx"]].copy()

    if outaged.empty:
        outaged_ids = pd.Series(
            data=[[] for _ in range(len(count_summary))],
            index=count_summary.index,
            name="outaged_branch_ids",
        )
    else:
        outaged_ids = outaged.groupby("scenario", sort=False)["idx"].apply(
            lambda s: [int(x) for x in s.to_numpy()]
        )
        outaged_ids.name = "outaged_branch_ids"

    summary = count_summary.join(active_loading_summary, how="left")
    summary = summary.join(outaged_ids, how="left")

    summary["max_loading_percent"] = summary["max_loading_percent"].fillna(0.0)
    summary["mean_loading_percent"] = summary["mean_loading_percent"].fillna(0.0)

    summary["num_overloaded_branches"] = summary[
        "num_overloaded_branches"
    ].astype(int)

    summary["num_hard_overloaded_branches"] = summary[
        "num_hard_overloaded_branches"
    ].astype(int)

    summary["num_outaged_branches"] = summary["num_outaged_branches"].astype(int)

    summary["outaged_branch_ids"] = summary["outaged_branch_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    summary = summary.reset_index()

    return summary


def compute_bus_summary(bus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized bus summary per scenario.

    Produces:
        load_scenario_idx
        min_vm_pu
        max_vm_pu
        total_voltage_violation
        num_low_voltage_buses
        num_high_voltage_buses
    """

    df = bus_df.copy()

    required = {
        "scenario",
        "load_scenario_idx",
        "Vm",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"bus_data.parquet missing columns: {sorted(missing)}")

    df["scenario"] = df["scenario"].astype(np.int64)

    if "min_vm_pu" not in df.columns:
        df["min_vm_pu"] = 0.94

    if "max_vm_pu" not in df.columns:
        df["max_vm_pu"] = 1.06

    vm = df["Vm"].to_numpy(dtype=np.float64)
    vmin = df["min_vm_pu"].to_numpy(dtype=np.float64)
    vmax = df["max_vm_pu"].to_numpy(dtype=np.float64)

    low_violation = np.maximum(vmin - vm, 0.0)
    high_violation = np.maximum(vm - vmax, 0.0)

    df["low_voltage_violation"] = low_violation.astype(np.float32)
    df["high_voltage_violation"] = high_violation.astype(np.float32)
    df["total_voltage_violation_part"] = (
        low_violation + high_violation
    ).astype(np.float32)

    df["is_low_voltage"] = low_violation > 0.0
    df["is_high_voltage"] = high_violation > 0.0

    summary = df.groupby("scenario", sort=False).agg(
        load_scenario_idx=("load_scenario_idx", "first"),
        min_vm_pu=("Vm", "min"),
        max_vm_pu=("Vm", "max"),
        total_low_voltage_violation=("low_voltage_violation", "sum"),
        total_high_voltage_violation=("high_voltage_violation", "sum"),
        total_voltage_violation=("total_voltage_violation_part", "sum"),
        num_low_voltage_buses=("is_low_voltage", "sum"),
        num_high_voltage_buses=("is_high_voltage", "sum"),
    )

    summary["num_low_voltage_buses"] = summary["num_low_voltage_buses"].astype(int)
    summary["num_high_voltage_buses"] = summary["num_high_voltage_buses"].astype(int)

    summary = summary.reset_index()

    return summary


def add_seriousness_score(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranking score for selecting the most useful emergency scenarios.
    """

    df = summary.copy()

    df["seriousness_score"] = (
        2000.0 * df["num_hard_overloaded_branches"].astype(float)
        + 200.0 * df["num_overloaded_branches"].astype(float)
        + 5.0 * df["max_loading_percent"].astype(float)
        + 20.0 * df["num_outaged_branches"].astype(float)
        + 1000.0 * df["total_voltage_violation"].astype(float)
    )

    return df


def build_fast_summary(
    raw_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build scenario summary using vectorized groupby operations.

    This replaces the slow adapter.build_summary() loop.

    tqdm shows progress by major stages:
        1. read bus parquet
        2. read branch parquet
        3. compute bus summary
        4. compute branch summary
        5. merge summaries
        6. add seriousness score
    """

    bus_path = raw_dir / "bus_data.parquet"
    branch_path = raw_dir / "branch_data.parquet"

    progress = None

    if show_progress and tqdm is not None:
        progress = tqdm(
            total=6,
            desc="Building summary",
            unit="stage",
            dynamic_ncols=True,
        )

    def update_progress(stage_name: str) -> None:
        if progress is not None:
            progress.set_postfix_str(stage_name)
            progress.update(1)

    try:
        t0 = now()

        bus_df = read_parquet_columns(
            path=bus_path,
            required_columns=[
                "scenario",
                "load_scenario_idx",
                "Vm",
            ],
            optional_columns=[
                "min_vm_pu",
                "max_vm_pu",
            ],
        )

        print_time("Read bus_data.parquet", t0)
        update_progress("bus parquet read")

        t0 = now()

        branch_df = read_parquet_columns(
            path=branch_path,
            required_columns=[
                "scenario",
                "idx",
                "from_bus",
                "to_bus",
                "br_status",
                "pf",
                "qf",
                "pt",
                "qt",
                "rate_a",
            ],
        )

        print_time("Read branch_data.parquet", t0)
        update_progress("branch parquet read")

        t0 = now()
        bus_summary = compute_bus_summary(bus_df)
        print_time("Compute bus summary", t0)
        update_progress("bus summary")

        # Free memory earlier on large raw datasets.
        del bus_df

        t0 = now()
        branch_summary = compute_branch_summary(branch_df)
        print_time("Compute branch summary", t0)
        update_progress("branch summary")

        # Free memory earlier on large raw datasets.
        del branch_df

        t0 = now()

        summary = pd.merge(
            bus_summary,
            branch_summary,
            on="scenario",
            how="inner",
            validate="one_to_one",
        )

        print_time("Merge summaries", t0)
        update_progress("merge")

        del bus_summary
        del branch_summary

        t0 = now()
        summary = add_seriousness_score(summary)
        print_time("Add seriousness score", t0)
        update_progress("score")

        return summary

    finally:
        if progress is not None:
            progress.close()


def select_candidates(
    summary: pd.DataFrame,
    min_loading: float,
    max_loading: float,
    require_outage: bool,
) -> pd.DataFrame:
    mask = (
        (summary["max_loading_percent"] >= float(min_loading))
        & (summary["max_loading_percent"] <= float(max_loading))
        & (summary["num_overloaded_branches"] > 0)
    )

    if require_outage:
        mask = mask & (summary["num_outaged_branches"] > 0)

    return summary.loc[mask].copy()


def sort_candidates(
    candidates: pd.DataFrame,
    prefer_hard_overloads: bool,
) -> pd.DataFrame:
    sort_columns: list[str] = []

    if prefer_hard_overloads:
        sort_columns.append("num_hard_overloaded_branches")

    sort_columns.extend(
        [
            "seriousness_score",
            "max_loading_percent",
            "num_overloaded_branches",
            "num_outaged_branches",
        ]
    )

    return candidates.sort_values(
        sort_columns,
        ascending=[False for _ in sort_columns],
    )


def make_output(selected: pd.DataFrame) -> pd.DataFrame:
    """
    Convert selected summary rows into transitions CSV format.
    """

    output = pd.DataFrame(
        {
            "scenario_id": selected["scenario"].astype(int),
            "load_scenario_idx": selected["load_scenario_idx"],
            "max_loading_percent": selected["max_loading_percent"],
            "mean_loading_percent": selected["mean_loading_percent"],
            "num_overloaded_branches": selected["num_overloaded_branches"].astype(int),
            "num_hard_overloaded_branches": selected[
                "num_hard_overloaded_branches"
            ].astype(int),
            "num_outaged_branches": selected["num_outaged_branches"].astype(int),
            "total_voltage_violation": selected["total_voltage_violation"],
            "seriousness_score": selected["seriousness_score"],
            "outaged_branch_ids": selected["outaged_branch_ids"].astype(str),
        }
    )

    return output


def print_selected_preview(
    output: pd.DataFrame,
    print_top: int,
) -> None:
    """
    Print only a preview, not thousands of selected rows.
    """

    if output.empty:
        return

    n = min(int(print_top), len(output))

    if n <= 0:
        return

    print(f"\nSelected scenarios preview, top {n} of {len(output)}:")
    print(
        output[
            [
                "scenario_id",
                "max_loading_percent",
                "num_overloaded_branches",
                "num_hard_overloaded_branches",
                "num_outaged_branches",
                "seriousness_score",
            ]
        ]
        .head(n)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a transitions CSV with serious overloaded GridFM scenarios. "
            "Optimized groupby version for large raw datasets."
        )
    )

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to GridFM raw directory containing bus_data.parquet and branch_data.parquet.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output transitions CSV path.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of selected scenarios to save.",
    )

    parser.add_argument(
        "--min-loading",
        type=float,
        default=105.0,
        help="Minimum max branch loading percent.",
    )

    parser.add_argument(
        "--max-loading",
        type=float,
        default=250.0,
        help="Maximum max branch loading percent.",
    )

    parser.add_argument(
        "--require-outage",
        action="store_true",
        help="First selection requires at least one initially outaged branch.",
    )

    parser.add_argument(
        "--fill-with-no-outage",
        action="store_true",
        help=(
            "If --require-outage gives fewer than --top-n rows, fill remaining rows "
            "with serious scenarios without requiring outage."
        ),
    )

    parser.add_argument(
        "--prefer-hard-overloads",
        action="store_true",
        help="Sort primarily by number of hard overloaded branches.",
    )

    parser.add_argument(
        "--print-top",
        type=int,
        default=20,
        help="How many selected rows to print as preview. Use 0 to disable.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    args = parser.parse_args()

    total_start = now()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Building serious transitions, optimized")
    print("=" * 100)
    print(f"Raw directory:          {raw_dir.resolve()}")
    print(f"Output:                 {output_path}")
    print(f"Top N requested:         {args.top_n}")
    print(f"Min loading:             {args.min_loading}")
    print(f"Max loading:             {args.max_loading}")
    print(f"Require outage first:    {args.require_outage}")
    print(f"Fill with no outage:     {args.fill_with_no_outage}")
    print(f"Prefer hard overloads:   {args.prefer_hard_overloads}")
    print(f"Print top:               {args.print_top}")
    print(f"Progress bar:            {not args.no_progress}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    t0 = now()
    summary = build_fast_summary(
        raw_dir=raw_dir,
        show_progress=not args.no_progress,
    )
    print_time("Build full summary", t0)

    if summary.empty:
        raise RuntimeError("GridFM summary is empty. Check raw_dir.")

    t0 = now()

    primary = select_candidates(
        summary=summary,
        min_loading=args.min_loading,
        max_loading=args.max_loading,
        require_outage=args.require_outage,
    )

    primary = sort_candidates(
        candidates=primary,
        prefer_hard_overloads=args.prefer_hard_overloads,
    )

    selected = primary.copy()

    if args.require_outage and args.fill_with_no_outage and len(selected) < args.top_n:
        selected_ids = set(int(x) for x in selected["scenario"].values)

        filler = select_candidates(
            summary=summary,
            min_loading=args.min_loading,
            max_loading=args.max_loading,
            require_outage=False,
        )

        filler = filler[~filler["scenario"].astype(int).isin(selected_ids)].copy()

        filler = sort_candidates(
            candidates=filler,
            prefer_hard_overloads=args.prefer_hard_overloads,
        )

        selected = pd.concat([selected, filler], ignore_index=True)

    selected = selected.head(int(args.top_n)).copy()

    print_time("Select and sort candidates", t0)

    if selected.empty:
        raise RuntimeError(
            "No scenarios matched the filters. "
            "Try lowering --min-loading or disabling --require-outage."
        )

    output = make_output(selected)
    output.to_csv(output_path, index=False)

    print_selected_preview(
        output=output,
        print_top=args.print_top,
    )

    print("\nSummary:")
    print(f"  Total scenarios in raw data:      {len(summary)}")
    print(f"  Primary matched scenarios:        {len(primary)}")
    print(f"  Saved rows:                       {len(output)}")
    print(f"  Requested top-n:                  {args.top_n}")
    print(f"  Output:                           {output_path}")

    if len(output) < args.top_n:
        print("\nWARNING:")
        print(
            f"  Requested {args.top_n} scenarios, but only {len(output)} matched the filters."
        )
        print("  This is not a script error.")
        print("  The current raw dataset is too small or not serious enough.")
        print("  To get more examples, generate/load a larger GridFM raw dataset.")

    weak_count = int((output["max_loading_percent"] < float(args.min_loading)).sum())

    if weak_count > 0:
        print("\nWARNING:")
        print(
            f"  Output still contains {weak_count} scenarios below "
            f"min loading {args.min_loading}%."
        )

    print_time("\nTotal runtime", total_start)
    print("\nDone.")


if __name__ == "__main__":
    main()