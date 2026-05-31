from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_topology_ai.data_adapter import GridFMAdapter


def add_seriousness_score(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    df["seriousness_score"] = (
        2000.0 * df["num_hard_overloaded_branches"].astype(float)
        + 200.0 * df["num_overloaded_branches"].astype(float)
        + 5.0 * df["max_loading_percent"].astype(float)
        + 20.0 * df["num_outaged_branches"].astype(float)
        + 1000.0 * df["total_voltage_violation"].astype(float)
    )

    return df


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a transitions CSV with serious overloaded GridFM scenarios."
    )

    parser.add_argument("raw_dir", type=str)

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-loading",
        type=float,
        default=105.0,
        help="Minimum max branch loading percent. Use 105 or 110 to remove weak ~100% cases.",
    )

    parser.add_argument(
        "--max-loading",
        type=float,
        default=250.0,
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
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Building serious transitions")
    print("=" * 100)
    print(f"Raw directory:          {raw_dir.resolve()}")
    print(f"Output:                 {output_path}")
    print(f"Top N requested:         {args.top_n}")
    print(f"Min loading:             {args.min_loading}")
    print(f"Max loading:             {args.max_loading}")
    print(f"Require outage first:    {args.require_outage}")
    print(f"Fill with no outage:     {args.fill_with_no_outage}")
    print(f"Prefer hard overloads:   {args.prefer_hard_overloads}")

    adapter = GridFMAdapter(raw_dir)
    summary = adapter.build_summary()

    if summary.empty:
        raise RuntimeError("GridFM summary is empty. Check raw_dir.")

    summary = add_seriousness_score(summary)

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

    if selected.empty:
        raise RuntimeError(
            "No scenarios matched the filters. Try lowering --min-loading or disabling --require-outage."
        )

    output = pd.DataFrame(
        {
            "scenario_id": selected["scenario"].astype(int),
            "load_scenario_idx": selected["load_scenario_idx"],
            "max_loading_percent": selected["max_loading_percent"],
            "mean_loading_percent": selected["mean_loading_percent"],
            "num_overloaded_branches": selected["num_overloaded_branches"],
            "num_hard_overloaded_branches": selected["num_hard_overloaded_branches"],
            "num_outaged_branches": selected["num_outaged_branches"],
            "total_voltage_violation": selected["total_voltage_violation"],
            "seriousness_score": selected["seriousness_score"],
            "outaged_branch_ids": selected["outaged_branch_ids"].astype(str),
        }
    )

    output.to_csv(output_path, index=False)

    print("\nSelected scenarios:")
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
        ].to_string(index=False)
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

    weak_count = int((output["max_loading_percent"] < 105.0).sum())

    if weak_count > 0:
        print("\nWARNING:")
        print(f"  Output still contains {weak_count} scenarios with max_loading < 105%.")
        print("  Consider using --min-loading 105 or --min-loading 110.")

    print("\nDone.")


if __name__ == "__main__":
    main()