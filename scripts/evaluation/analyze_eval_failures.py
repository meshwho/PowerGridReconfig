from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def safe_parse_list(value: Any) -> list[Any]:
    """
    Safely parse list-like strings from evaluation CSV.

    Expected examples:
        "[40, 32, None]"
        "[154, None]"
        "[]"

    If parsing fails, return an empty list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []

    if isinstance(parsed, list):
        return parsed

    return []


def normalize_branch_sequence(
    branches_value: Any,
    drop_none: bool = False,
) -> tuple[Any, ...]:
    """
    Convert a branch sequence string to a hashable tuple.

    If drop_none=True, handoff markers are removed.
    """

    branches = safe_parse_list(branches_value)

    normalized = []

    for item in branches:
        if item is None:
            if not drop_none:
                normalized.append(None)
            continue

        if isinstance(item, float) and pd.isna(item):
            if not drop_none:
                normalized.append(None)
            continue

        try:
            normalized.append(int(item))
        except Exception:
            normalized.append(str(item))

    return tuple(normalized)


def normalize_action_sequence(
    actions_value: Any,
) -> tuple[Any, ...]:
    """
    Convert action sequence string to a hashable tuple.
    """

    actions = safe_parse_list(actions_value)

    normalized = []

    for item in actions:
        if item is None:
            normalized.append(None)
            continue

        if isinstance(item, float) and pd.isna(item):
            normalized.append(None)
            continue

        try:
            normalized.append(int(item))
        except Exception:
            normalized.append(str(item))

    return tuple(normalized)


def parse_bad_reasons(text: str) -> set[str]:
    """
    Parse comma-separated termination reasons.
    """

    return {
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    }


def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
    """
    Validate that evaluation CSV has expected columns.
    """

    required = {
        "scenario_id",
        "steps",
        "termination_reason",
        "solved",
        "branches",
        "actions",
        "discounted_return",
        "safety_score",
        "final_max_loading_percent",
        "final_num_overloaded_branches",
        "final_num_hard_overloaded_branches",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"CSV file is missing required columns: {missing}\n"
            f"File: {path}"
        )


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper columns for analysis.
    """

    df = df.copy()

    df["branch_sequence"] = df["branches"].apply(
        lambda x: normalize_branch_sequence(x, drop_none=False)
    )

    df["branch_sequence_no_none"] = df["branches"].apply(
        lambda x: normalize_branch_sequence(x, drop_none=True)
    )

    df["action_sequence"] = df["actions"].apply(normalize_action_sequence)

    df["num_switch_actions"] = df["branch_sequence_no_none"].apply(len)

    df["has_hard_overload"] = (
        df["final_num_hard_overloaded_branches"].astype(float) > 0
    )

    df["has_soft_or_hard_overload"] = (
        df["final_num_overloaded_branches"].astype(float) > 0
    )

    df["is_max_steps"] = df["termination_reason"].astype(str).eq(
        "max_steps_reached"
    )

    df["is_handoff"] = df["termination_reason"].astype(str).str.contains(
        "handoff",
        case=False,
        na=False,
    )

    return df


def summarize_basic(df: pd.DataFrame, title: str) -> None:
    """
    Print general summary for evaluation CSV.
    """

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(f"Rows:              {len(df)}")
    print(f"Scenarios:         {df['scenario_id'].nunique()}")
    print(f"Solved:            {int(df['solved'].astype(bool).sum())}")
    print(f"Unsolved:          {int((~df['solved'].astype(bool)).sum())}")

    print("\nTermination reasons:")
    print(df["termination_reason"].value_counts(dropna=False).to_string())

    print("\nAverage metrics:")
    print(f"  Avg discounted return: {df['discounted_return'].mean():.4f}")
    print(f"  Avg final loading:     {df['final_max_loading_percent'].mean():.4f}%")
    print(f"  Avg overloaded:        {df['final_num_overloaded_branches'].mean():.4f}")
    print(f"  Avg hard overloaded:   {df['final_num_hard_overloaded_branches'].mean():.4f}")
    print(f"  Avg safety score:      {df['safety_score'].mean():.4f}")
    print(f"  Total safety score:    {df['safety_score'].sum():.4f}")


def print_top_sequences(
    df: pd.DataFrame,
    title: str,
    top_n: int,
) -> pd.DataFrame:
    """
    Print most frequent branch sequences.
    """

    sequence_counts = (
        df["branch_sequence"]
        .value_counts(dropna=False)
        .head(top_n)
        .reset_index()
    )

    sequence_counts.columns = ["branch_sequence", "count"]

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    if sequence_counts.empty:
        print("No sequences found.")
    else:
        print(sequence_counts.to_string(index=False))

    return sequence_counts


def branch_frequency(
    df: pd.DataFrame,
    title: str,
    top_n: int,
) -> pd.DataFrame:
    """
    Count how often each branch appears in the selected trajectories.
    """

    counter: Counter[int] = Counter()

    for branches in df["branch_sequence_no_none"]:
        for branch_id in branches:
            try:
                counter[int(branch_id)] += 1
            except Exception:
                pass

    rows = [
        {"branch_id": branch_id, "count": count}
        for branch_id, count in counter.most_common(top_n)
    ]

    out = pd.DataFrame(rows)

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    if out.empty:
        print("No branches found.")
    else:
        print(out.to_string(index=False))

    return out


def worst_cases_table(
    df: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool],
    top_n: int,
) -> pd.DataFrame:
    """
    Return table with worst cases according to selected columns.
    """

    columns = [
        "scenario_id",
        "termination_reason",
        "solved",
        "steps",
        "branches",
        "discounted_return",
        "safety_score",
        "final_max_loading_percent",
        "final_num_overloaded_branches",
        "final_num_hard_overloaded_branches",
        "num_switch_actions",
    ]

    available_columns = [
        col for col in columns
        if col in df.columns
    ]

    return (
        df.sort_values(
            sort_columns,
            ascending=ascending,
        )
        .head(top_n)[available_columns]
        .copy()
    )


def print_table(
    title: str,
    table: pd.DataFrame,
) -> None:
    """
    Print a table safely.
    """

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    if table.empty:
        print("No rows.")
    else:
        print(table.to_string(index=False))


def save_table(
    table: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Path:
    """
    Save table to CSV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    table.to_csv(path, index=False)

    return path


def analyze_one_file(
    eval_csv: Path,
    output_dir: Path,
    bad_reasons: set[str],
    top_n: int,
    hard_case_min_loading: float,
    hard_case_min_overloaded: int,
    hard_case_min_hard: int,
) -> None:
    """
    Analyze one evaluation CSV.
    """

    df = pd.read_csv(eval_csv)
    ensure_required_columns(df, eval_csv)

    df = add_derived_columns(df)

    summarize_basic(
        df=df,
        title=f"Evaluation analysis: {eval_csv}",
    )

    bad_df = df[df["termination_reason"].isin(bad_reasons)].copy()

    hard_remaining_df = df[
        df["final_num_hard_overloaded_branches"].astype(float) > 0
    ].copy()

    severe_df = df[
        (
            df["final_max_loading_percent"].astype(float)
            >= float(hard_case_min_loading)
        )
        | (
            df["final_num_overloaded_branches"].astype(float)
            >= int(hard_case_min_overloaded)
        )
        | (
            df["final_num_hard_overloaded_branches"].astype(float)
            >= int(hard_case_min_hard)
        )
    ].copy()

    print("\n" + "=" * 100)
    print("Failure subset sizes")
    print("=" * 100)

    print(f"Bad reasons selected:       {sorted(bad_reasons)}")
    print(f"Bad rows:                   {len(bad_df)}")
    print(f"Rows with hard remaining:   {len(hard_remaining_df)}")
    print(f"Severe rows selected:       {len(severe_df)}")

    # Top repeated bad sequences
    top_bad_sequences = print_top_sequences(
        df=bad_df,
        title="Top repeated branch sequences among selected bad cases",
        top_n=top_n,
    )

    top_max_step_sequences = print_top_sequences(
        df=df[df["termination_reason"].eq("max_steps_reached")],
        title="Top repeated branch sequences among max_steps_reached cases",
        top_n=top_n,
    )

    top_hard_sequences = print_top_sequences(
        df=hard_remaining_df,
        title="Top repeated branch sequences among cases with hard overload remaining",
        top_n=top_n,
    )

    # Branch frequencies
    bad_branch_freq = branch_frequency(
        df=bad_df,
        title="Most frequent branches in selected bad trajectories",
        top_n=top_n,
    )

    max_step_branch_freq = branch_frequency(
        df=df[df["termination_reason"].eq("max_steps_reached")],
        title="Most frequent branches in max_steps_reached trajectories",
        top_n=top_n,
    )

    hard_branch_freq = branch_frequency(
        df=hard_remaining_df,
        title="Most frequent branches in trajectories with hard overload remaining",
        top_n=top_n,
    )

    # Worst cases
    worst_by_loading = worst_cases_table(
        df=df,
        sort_columns=[
            "final_max_loading_percent",
            "final_num_hard_overloaded_branches",
            "final_num_overloaded_branches",
        ],
        ascending=[False, False, False],
        top_n=top_n,
    )

    print_table(
        title="Worst cases by final max loading",
        table=worst_by_loading,
    )

    worst_by_hard = worst_cases_table(
        df=df,
        sort_columns=[
            "final_num_hard_overloaded_branches",
            "final_max_loading_percent",
            "final_num_overloaded_branches",
        ],
        ascending=[False, False, False],
        top_n=top_n,
    )

    print_table(
        title="Worst cases by remaining hard overloads",
        table=worst_by_hard,
    )

    worst_by_score = worst_cases_table(
        df=df,
        sort_columns=[
            "safety_score",
            "final_num_hard_overloaded_branches",
            "final_max_loading_percent",
        ],
        ascending=[True, False, False],
        top_n=top_n,
    )

    print_table(
        title="Worst cases by safety score",
        table=worst_by_score,
    )

    severe_scenarios = (
        severe_df[["scenario_id"]]
        .drop_duplicates()
        .sort_values("scenario_id")
        .reset_index(drop=True)
    )

    bad_scenarios = (
        bad_df[["scenario_id"]]
        .drop_duplicates()
        .sort_values("scenario_id")
        .reset_index(drop=True)
    )

    hard_remaining_scenarios = (
        hard_remaining_df[["scenario_id"]]
        .drop_duplicates()
        .sort_values("scenario_id")
        .reset_index(drop=True)
    )

    # Save all outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    saved_paths.append(
        save_table(
            df,
            output_dir,
            "all_eval_rows_with_derived_columns.csv",
        )
    )

    saved_paths.append(
        save_table(
            bad_df,
            output_dir,
            "bad_cases.csv",
        )
    )

    saved_paths.append(
        save_table(
            hard_remaining_df,
            output_dir,
            "hard_remaining_cases.csv",
        )
    )

    saved_paths.append(
        save_table(
            severe_df,
            output_dir,
            "severe_cases.csv",
        )
    )

    saved_paths.append(
        save_table(
            bad_scenarios,
            output_dir,
            "bad_case_scenario_ids.csv",
        )
    )

    saved_paths.append(
        save_table(
            hard_remaining_scenarios,
            output_dir,
            "hard_remaining_scenario_ids.csv",
        )
    )

    saved_paths.append(
        save_table(
            severe_scenarios,
            output_dir,
            "severe_case_scenario_ids.csv",
        )
    )

    saved_paths.append(
        save_table(
            top_bad_sequences,
            output_dir,
            "top_bad_branch_sequences.csv",
        )
    )

    saved_paths.append(
        save_table(
            top_max_step_sequences,
            output_dir,
            "top_max_steps_branch_sequences.csv",
        )
    )

    saved_paths.append(
        save_table(
            top_hard_sequences,
            output_dir,
            "top_hard_remaining_branch_sequences.csv",
        )
    )

    saved_paths.append(
        save_table(
            bad_branch_freq,
            output_dir,
            "bad_branch_frequency.csv",
        )
    )

    saved_paths.append(
        save_table(
            max_step_branch_freq,
            output_dir,
            "max_steps_branch_frequency.csv",
        )
    )

    saved_paths.append(
        save_table(
            hard_branch_freq,
            output_dir,
            "hard_remaining_branch_frequency.csv",
        )
    )

    saved_paths.append(
        save_table(
            worst_by_loading,
            output_dir,
            "worst_by_final_loading.csv",
        )
    )

    saved_paths.append(
        save_table(
            worst_by_hard,
            output_dir,
            "worst_by_hard_overloads.csv",
        )
    )

    saved_paths.append(
        save_table(
            worst_by_score,
            output_dir,
            "worst_by_safety_score.csv",
        )
    )

    print("\n" + "=" * 100)
    print("Saved analysis files")
    print("=" * 100)

    for path in saved_paths:
        print(path)


def compare_eval_files(
    eval_csvs: list[Path],
) -> pd.DataFrame:
    """
    Compare several evaluation CSV files.
    """

    rows = []

    for path in eval_csvs:
        df = pd.read_csv(path)
        ensure_required_columns(df, path)

        row = {
            "file": str(path),
            "rows": len(df),
            "scenarios": df["scenario_id"].nunique(),
            "solved": int(df["solved"].astype(bool).sum()),
            "unsolved": int((~df["solved"].astype(bool)).sum()),
            "avg_discounted_return": float(df["discounted_return"].mean()),
            "avg_final_loading_percent": float(
                df["final_max_loading_percent"].mean()
            ),
            "avg_overloaded": float(
                df["final_num_overloaded_branches"].mean()
            ),
            "avg_hard_overloaded": float(
                df["final_num_hard_overloaded_branches"].mean()
            ),
            "avg_safety_score": float(df["safety_score"].mean()),
            "total_safety_score": float(df["safety_score"].sum()),
        }

        reason_counts = df["termination_reason"].value_counts(dropna=False)

        for reason, count in reason_counts.items():
            row[f"reason_{reason}"] = int(count)

        rows.append(row)

    comparison = pd.DataFrame(rows).fillna(0)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze evaluation failures from evaluate_checkpoint output CSV."
        )
    )

    parser.add_argument(
        "eval_csv",
        nargs="+",
        type=str,
        help="One or more evaluation CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save analysis files. "
            "If omitted, a folder is created near the first eval CSV."
        ),
    )

    parser.add_argument(
        "--bad-reasons",
        type=str,
        default="max_steps_reached",
        help=(
            "Comma-separated termination reasons considered bad. "
            "Default: max_steps_reached"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top cases/sequences/branches to print and save.",
    )

    parser.add_argument(
        "--hard-case-min-loading",
        type=float,
        default=120.0,
        help=(
            "Select severe cases with final max loading above this threshold."
        ),
    )

    parser.add_argument(
        "--hard-case-min-overloaded",
        type=int,
        default=3,
        help=(
            "Select severe cases with at least this many overloaded branches."
        ),
    )

    parser.add_argument(
        "--hard-case-min-hard",
        type=int,
        default=1,
        help=(
            "Select severe cases with at least this many hard overloaded branches."
        ),
    )

    args = parser.parse_args()

    eval_csvs = [
        Path(path)
        for path in args.eval_csv
    ]

    for path in eval_csvs:
        if not path.exists():
            raise FileNotFoundError(f"Evaluation CSV not found: {path}")

    bad_reasons = parse_bad_reasons(args.bad_reasons)

    first_csv = eval_csvs[0]

    if args.output_dir is None:
        output_dir = first_csv.parent / f"{first_csv.stem}_analysis"
    else:
        output_dir = Path(args.output_dir)

    if len(eval_csvs) > 1:
        comparison = compare_eval_files(eval_csvs)

        print("\n" + "=" * 100)
        print("Evaluation comparison")
        print("=" * 100)
        print(comparison.to_string(index=False))

        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = output_dir / "eval_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        print(f"\nSaved comparison: {comparison_path}")

    # Detailed analysis only for the first file.
    analyze_one_file(
        eval_csv=first_csv,
        output_dir=output_dir,
        bad_reasons=bad_reasons,
        top_n=int(args.top_n),
        hard_case_min_loading=float(args.hard_case_min_loading),
        hard_case_min_overloaded=int(args.hard_case_min_overloaded),
        hard_case_min_hard=int(args.hard_case_min_hard),
    )


if __name__ == "__main__":
    main()