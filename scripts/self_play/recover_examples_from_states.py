from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_RE = re.compile(r"impact_teacher_scenario_(\d+)_step_(\d+)\.npz$")


def read_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"].item()

    return json.loads(str(metadata_raw))


def make_policy_json(action_id: int) -> str:
    return json.dumps({str(int(action_id)): 1.0})


def make_visit_counts_json(action_id: int) -> str:
    return json.dumps({str(int(action_id)): 1})


def recover_examples(states_dir: Path, gamma: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(states_dir.glob("*.npz")):
        match = STATE_RE.search(path.name)

        if match is None:
            continue

        state_id = path.stem

        scenario_id_from_name = int(match.group(1))
        step_from_name = int(match.group(2))

        try:
            meta = read_npz_metadata(path)
        except Exception as exc:
            print(f"WARNING: failed to read {path}: {exc}")
            continue

        scenario_id = int(meta.get("scenario_id", scenario_id_from_name))
        step = int(meta.get("step", step_from_name))

        selected_action_id = int(meta["selected_action_id"])
        selected_branch_id = meta.get("selected_branch_id", None)

        if selected_branch_id is not None:
            selected_branch_id = int(selected_branch_id)

        step_reward = float(meta.get("step_safety_improvement", 0.0))

        teacher_reason = str(meta.get("teacher_decision_reason", ""))

        # This is approximate for recovered CSV.
        # Training mainly needs state_path, selected_action_id, policy json and returns.
        if selected_action_id == 0:
            termination_reason = "handoff_to_redispatch_teacher"
            solved = False
            done = True
        else:
            handoff_added = bool(meta.get("handoff_added", False))
            if handoff_added:
                termination_reason = "handoff_to_redispatch_teacher"
            else:
                termination_reason = "max_steps_reached"

            solved = False
            done = False

        rows.append(
            {
                "state_id": state_id,
                "state_path": str(path),
                "scenario_id": scenario_id,
                "step": step,
                "selected_action_id": selected_action_id,
                "selected_branch_id": selected_branch_id,
                "step_reward": step_reward,
                "final_return": 0.0,
                "discounted_return_from_step": 0.0,
                "solved": bool(solved),
                "done": bool(done),
                "termination_reason": termination_reason,
                "visit_counts_json": make_visit_counts_json(selected_action_id),
                "mcts_policy_json": make_policy_json(selected_action_id),
                "teacher_decision_reason": teacher_reason,
            }
        )

    if not rows:
        raise RuntimeError(f"No recoverable .npz states found in: {states_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["scenario_id", "step"]).reset_index(drop=True)

    # Recompute discounted returns per scenario from saved step_reward.
    recovered_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("scenario_id", sort=False):
        group = group.sort_values("step").copy()

        rewards = group["step_reward"].astype(float).tolist()
        returns = [0.0 for _ in rewards]

        running = 0.0
        for i in reversed(range(len(rewards))):
            running = float(rewards[i]) + float(gamma) * running
            returns[i] = float(running)

        group["discounted_return_from_step"] = returns
        group["final_return"] = float(returns[0]) if returns else 0.0

        recovered_parts.append(group)

    recovered = pd.concat(recovered_parts, ignore_index=True)
    recovered = recovered.sort_values(["scenario_id", "step"]).reset_index(drop=True)

    return recovered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover teacher examples.csv from saved state .npz files."
    )

    parser.add_argument(
        "--states-dir",
        type=str,
        required=True,
        help="Directory with saved .npz states.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Recovered examples CSV path.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    states_dir = Path(args.states_dir)
    output_csv = Path(args.output_csv)

    if not states_dir.exists():
        raise FileNotFoundError(f"States directory not found: {states_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = recover_examples(
        states_dir=states_dir,
        gamma=float(args.gamma),
    )

    df.to_csv(output_csv, index=False)

    print("Recovered examples CSV")
    print(f"states dir: {states_dir}")
    print(f"output:     {output_csv}")
    print(f"examples:   {len(df)}")
    print(f"scenarios:  {df.scenario_id.nunique()}")

    print("\nStep distribution:")
    print(df.step.value_counts().sort_index().to_string())

    print("\nAction 0 / handoff examples:")
    print(int((df.selected_action_id == 0).sum()))

    print("\nTermination reasons:")
    print(df.termination_reason.value_counts(dropna=False).to_string())

    print("\nStep reward:")
    print(df.step_reward.describe().to_string())

    print("\nNegative rewards:")
    print(int((df.step_reward < 0).sum()))


if __name__ == "__main__":
    main()