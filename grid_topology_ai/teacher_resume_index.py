from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


RESUME_INDEX_VERSION = 1


def resume_index_path(checkpoint_path: Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(
        f"{checkpoint.stem}_resume_index.jsonl"
    )


def _checkpoint_size(checkpoint_path: Path) -> int:
    try:
        return int(Path(checkpoint_path).stat().st_size)
    except FileNotFoundError:
        return 0


def _encoded(record: dict[str, object]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def write_resume_snapshot(
    checkpoint_path: Path,
    contract_fingerprint: str,
    completed_scenario_ids: Sequence[int],
) -> None:
    index_path = resume_index_path(checkpoint_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": RESUME_INDEX_VERSION,
        "kind": "snapshot",
        "contract_fingerprint": str(contract_fingerprint),
        "checkpoint_size": _checkpoint_size(checkpoint_path),
        "completed_scenario_ids": sorted(
            {int(value) for value in completed_scenario_ids}
        ),
    }

    temp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    temp_path.write_text(_encoded(payload), encoding="utf-8")
    temp_path.replace(index_path)


def append_resume_delta(
    checkpoint_path: Path,
    contract_fingerprint: str,
    scenario_id: int,
    complete: bool,
    checkpoint_start: int,
) -> None:
    checkpoint_end = _checkpoint_size(checkpoint_path)
    start = int(checkpoint_start)
    if start < 0 or checkpoint_end <= start:
        raise ValueError(
            "Resume index delta requires checkpoint growth."
        )

    index_path = resume_index_path(checkpoint_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RESUME_INDEX_VERSION,
        "kind": "delta",
        "contract_fingerprint": str(contract_fingerprint),
        "checkpoint_start": start,
        "checkpoint_size": checkpoint_end,
        "scenario_id": int(scenario_id),
        "complete": bool(complete),
    }

    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(_encoded(payload))
        handle.flush()


def load_resume_index(
    checkpoint_path: Path,
    contract_fingerprint: str,
    allowed_scenario_ids: Sequence[int],
) -> set[int] | None:
    index_path = resume_index_path(checkpoint_path)
    if not index_path.exists():
        return None

    actual_size = _checkpoint_size(checkpoint_path)
    expected_contract = str(contract_fingerprint)
    allowed = {int(value) for value in allowed_scenario_ids}

    completed: set[int] | None = None
    covered_size: int | None = None

    try:
        with index_path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if not isinstance(record, dict):
                    return None
                if int(record.get("version", -1)) != RESUME_INDEX_VERSION:
                    return None
                if record.get("contract_fingerprint") != expected_contract:
                    return None

                kind = record.get("kind")
                if completed is None:
                    if kind != "snapshot":
                        return None

                    snapshot_size = int(record.get("checkpoint_size", -1))
                    if snapshot_size < 0 or snapshot_size > actual_size:
                        return None

                    scenario_ids = record.get("completed_scenario_ids")
                    if not isinstance(scenario_ids, list):
                        return None

                    completed = {int(value) for value in scenario_ids}
                    covered_size = snapshot_size
                    continue

                if kind != "delta" or covered_size is None:
                    return None

                start = int(record.get("checkpoint_start", -1))
                end = int(record.get("checkpoint_size", -1))
                if (
                    start != covered_size
                    or end <= start
                    or end > actual_size
                ):
                    return None

                scenario_id = int(record["scenario_id"])
                complete = record.get("complete")
                if not isinstance(complete, bool):
                    return None

                if complete:
                    completed.add(scenario_id)
                else:
                    completed.discard(scenario_id)
                covered_size = end
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    if completed is None or covered_size != actual_size:
        return None

    return completed & allowed
