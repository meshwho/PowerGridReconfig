from __future__ import annotations

import time
from pathlib import Path

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

    columns = list(dict.fromkeys(required_columns + optional_columns))

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        df = pd.read_parquet(path)

        missing_required = set(required_columns) - set(df.columns)

        if missing_required:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_required)}"
            )

        available = [col for col in columns if col in df.columns]

        return df[available].copy()


def compute_branch_summary(branch_df: pd.DataFrame) -> pd.DataFrame:
    """Build the vectorized branch summary for each scenario."""

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

    return summary.reset_index()


def compute_bus_summary(bus_df: pd.DataFrame) -> pd.DataFrame:
    """Build the vectorized bus summary for each scenario."""

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

    return summary.reset_index()


def add_seriousness_score(summary: pd.DataFrame) -> pd.DataFrame:
    """Add the existing emergency-scenario ranking score."""

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
    """Build the existing vectorized GridFM scenario summary."""

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

        del bus_df

        t0 = now()
        branch_summary = compute_branch_summary(branch_df)
        print_time("Compute branch summary", t0)
        update_progress("branch summary")

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


def make_output(selected: pd.DataFrame) -> pd.DataFrame:
    """Convert selected summary rows into the existing transitions format."""

    return pd.DataFrame(
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
