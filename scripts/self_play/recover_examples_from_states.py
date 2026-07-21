from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.contracts import (
    physics_provenance,
    require_exact_contract_version,
    require_physics_provenance,
)
from grid_topology_ai.physical_objective import PHYSICAL_OBJECTIVE_SCHEMA_VERSION
from grid_topology_ai.termination import (
    parse_termination_reason,
    validate_outcome_invariants,
)
from grid_topology_ai.value_targets import add_outcome_value_targets_to_rows

STATE_RE = re.compile(r"impact_teacher_scenario_(\d+)_step_(\d+)\.npz$")


def read_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"].item()

    return json.loads(str(metadata_raw))


def make_policy_json(action_id: int) -> str:
    return json.dumps({str(int(action_id)): 1.0})


def make_visit_counts_json(action_id: int) -> str:
    return json.dumps({str(int(action_id)): 1})

def require_metadata_bool(
    metadata: dict[str, Any],
    key: str,
    *,
    source: Path,
) -> bool:
    value = metadata.get(key)

    if not isinstance(value, bool):
        raise ValueError(
            f"{source} metadata field {key!r} must be a boolean."
        )

    return value

def recover_examples(states_dir: Path, gamma: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_physics_config: PhysicsConfig | None = None

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

        require_exact_contract_version(
            meta.get("physical_objective_schema_version"),
            expected=PHYSICAL_OBJECTIVE_SCHEMA_VERSION,
            name="physical-objective contract",
            source=str(path),
            regeneration_command="python -m scripts.self_play.generate ...",
        )
        state_physics_config = require_physics_provenance(
            meta,
            source=str(path),
            expected_physics_config=observed_physics_config,
        )
        if observed_physics_config is None:
            observed_physics_config = state_physics_config
        provenance = physics_provenance(state_physics_config)

        scenario_id = int(meta.get("scenario_id", scenario_id_from_name))
        step = int(meta.get("step", step_from_name))

        selected_action_id = int(meta["selected_action_id"])
        selected_branch_id = meta.get("selected_branch_id", None)

        if selected_branch_id is not None:
            selected_branch_id = int(selected_branch_id)

        step_reward = float(meta.get("step_safety_improvement", 0.0))

        teacher_reason = str(meta.get("teacher_decision_reason", ""))

        required_outcome_fields = {
            "episode_done",
            "episode_solved",
            "episode_termination_reason",
        }

        missing_outcome_fields = (
            required_outcome_fields - set(meta)
        )

        if missing_outcome_fields:
            raise ValueError(
                f"{path} does not contain exact terminal outcome metadata: "
                f"{sorted(missing_outcome_fields)}. Regenerate the episode; "
                f"approximate recovery cannot produce current-contract "
                f"value targets."
            )

        done = require_metadata_bool(
            meta,
            "episode_done",
            source=path,
        )

        solved = require_metadata_bool(
            meta,
            "episode_solved",
            source=path,
        )

        termination_reason = parse_termination_reason(
            meta["episode_termination_reason"],
            allow_none=False,
        )

        if not done:
            raise ValueError(
                f"{path} does not contain a terminal episode outcome."
            )

        validate_outcome_invariants(
            solved=solved,
            termination_reason=termination_reason,
        )

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
                "termination_reason": termination_reason.value,
                "physical_objective_schema_version": (
                    PHYSICAL_OBJECTIVE_SCHEMA_VERSION
                ),
                "physics_config_contract_version": provenance[
                    "physics_config_contract_version"
                ],
                "physics_config": json.dumps(
                    provenance["physics_config"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "physics_config_fingerprint": provenance[
                    "physics_config_fingerprint"
                ],
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

        episode_outcomes = group[
            ["solved", "done", "termination_reason"]
        ].drop_duplicates()

        if len(episode_outcomes) != 1:
            raise ValueError(
                "Recovered state files for one scenario contain "
                "contradictory terminal outcomes."
            )

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

    rows = recovered.to_dict(orient="records")
    add_outcome_value_targets_to_rows(
        rows=rows,
        gamma=float(gamma),
        group_keys=("scenario_id",),
    )
    recovered = pd.DataFrame(rows)

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
