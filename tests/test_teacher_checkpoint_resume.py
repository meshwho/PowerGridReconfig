from __future__ import annotations

import json

from scripts.self_play import generate_impact_teacher_redispatch_runtime as teacher


def test_resume_restores_completed_work_and_retries_incomplete_work(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    records = [
        {
            "scenario_id": 1,
            "ok": False,
            "reason": "exception",
            "rows": [],
        },
        {
            "scenario_id": 2,
            "ok": False,
            "reason": "no_teacher_action_found",
            "rows": [],
        },
        {
            "scenario_id": 3,
            "ok": True,
            "reason": None,
            "rows": [],
        },
        {
            "scenario_id": 4,
            "ok": True,
            "reason": None,
            "rows": [{"state_id": "state-4"}],
        },
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
        + '{"scenario_id":5,"ok":true',
        encoding="utf-8",
    )

    restored = teacher.load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3, 4, 5],
    )

    assert 1 not in restored
    assert restored[2]["reason"] == "no_teacher_action_found"
    assert 3 not in restored
    assert restored[4]["ok"] is True
    assert 5 not in restored


def test_resume_ignores_scenarios_outside_current_run(tmp_path) -> None:
    checkpoint_path = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "scenario_id": 99,
                "ok": True,
                "reason": None,
                "rows": [{"state_id": "other-run"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert teacher.load_scenario_checkpoints(
        checkpoint_path=checkpoint_path,
        allowed_scenario_ids=[1, 2, 3],
    ) == {}
