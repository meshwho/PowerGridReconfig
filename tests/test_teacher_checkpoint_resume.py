from __future__ import annotations

import json

from scripts.self_play import generate_impact_teacher_parallel_fast as teacher
from scripts.self_play.generate_impact_teacher_provenance import (
    load_scenario_checkpoints,
)


def test_exception_checkpoint_is_retried(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    records = [
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 1,
            "ok": False,
            "reason": "exception",
            "rows": [],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 2,
            "ok": False,
            "reason": "no_teacher_action_found",
            "rows": [],
        },
        {
            "version": teacher.CHECKPOINT_VERSION,
            "scenario_id": 3,
            "ok": True,
            "reason": None,
            "rows": [{"state_id": "state-3"}],
        },
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    restored = load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3],
    )

    assert 1 not in restored
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert restored[3]["ok"] is True
