from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from grid_topology_ai.evaluation.checkpoint import load_scenario_ids
from grid_topology_ai.self_play.lineage_artifacts import (
    build_scenario_lineages,
)


@dataclass(frozen=True, slots=True)
class PhysicalDatasetLineages:
    label: str
    transitions_csv: Path
    raw_dir: Path
    scenarios_by_fingerprint: dict[str, tuple[int, ...]]

    @property
    def fingerprints(self) -> set[str]:
        return set(self.scenarios_by_fingerprint)


def load_dataset_physical_lineages(
    *,
    transitions_csv: str | Path,
    raw_dir: str | Path,
    label: str,
) -> PhysicalDatasetLineages:
    transitions_path = Path(transitions_csv)
    raw_path = Path(raw_dir)
    scenario_ids = tuple(
        sorted(
            set(
                load_scenario_ids(
                    transitions_path,
                    limit=None,
                )
            )
        )
    )
    if not scenario_ids:
        raise ValueError(
            f"{label} transitions contain no scenario IDs: "
            f"{transitions_path}"
        )

    lineages = build_scenario_lineages(
        raw_dir=raw_path,
        scenario_ids=scenario_ids,
    )
    missing = sorted(set(scenario_ids) - set(lineages))
    unexpected = sorted(set(lineages) - set(scenario_ids))
    if missing or unexpected:
        raise ValueError(
            f"{label} physical lineage coverage mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}."
        )

    grouped: dict[str, list[int]] = defaultdict(list)
    for scenario_id in scenario_ids:
        grouped[lineages[scenario_id].fingerprint].append(
            int(scenario_id)
        )

    return PhysicalDatasetLineages(
        label=str(label),
        transitions_csv=transitions_path,
        raw_dir=raw_path,
        scenarios_by_fingerprint={
            fingerprint: tuple(sorted(ids))
            for fingerprint, ids in sorted(grouped.items())
        },
    )


def require_disjoint_physical_lineages(
    first: PhysicalDatasetLineages,
    second: PhysicalDatasetLineages,
) -> None:
    overlap = sorted(first.fingerprints & second.fingerprints)
    if not overlap:
        return

    details = []
    for fingerprint in overlap[:10]:
        details.append(
            f"fingerprint={fingerprint}, "
            f"{first.label} scenario_ids="
            f"{list(first.scenarios_by_fingerprint[fingerprint])}, "
            f"{second.label} scenario_ids="
            f"{list(second.scenarios_by_fingerprint[fingerprint])}"
        )

    raise ValueError(
        f"{first.label} and {second.label} physical lineages overlap: "
        + "; ".join(details)
    )


def validate_physical_dataset_isolation(
    *,
    pool_transitions_csv: str | Path,
    pool_raw_dir: str | Path,
    eval_transitions_csv: str | Path,
    eval_raw_dir: str | Path,
    final_test_transitions_csv: str | Path,
    final_test_raw_dir: str | Path,
    tuning_transitions_csv: str | Path | None = None,
    tuning_raw_dir: str | Path | None = None,
) -> dict[str, PhysicalDatasetLineages]:
    if (tuning_transitions_csv is None) != (tuning_raw_dir is None):
        raise ValueError(
            "Tuning transitions and raw directory must be provided together."
        )

    datasets = {
        "pool": load_dataset_physical_lineages(
            transitions_csv=pool_transitions_csv,
            raw_dir=pool_raw_dir,
            label="Pool",
        ),
        "eval": load_dataset_physical_lineages(
            transitions_csv=eval_transitions_csv,
            raw_dir=eval_raw_dir,
            label="Evaluation",
        ),
        "final_test": load_dataset_physical_lineages(
            transitions_csv=final_test_transitions_csv,
            raw_dir=final_test_raw_dir,
            label="final-test",
        ),
    }

    if tuning_transitions_csv is not None and tuning_raw_dir is not None:
        datasets["tuning"] = load_dataset_physical_lineages(
            transitions_csv=tuning_transitions_csv,
            raw_dir=tuning_raw_dir,
            label="Tuning",
        )

    names = tuple(datasets)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            require_disjoint_physical_lineages(
                datasets[first_name],
                datasets[second_name],
            )

    return datasets
