from __future__ import annotations

from pathlib import Path

import grid_topology_ai.teacher_runtime as teacher


def _append_checkpoint(path: Path, payload: bytes) -> int:
    with path.open("ab") as handle:
        handle.write(payload)
    return path.stat().st_size


def test_resume_index_tracks_snapshot_and_contiguous_deltas(tmp_path) -> None:
    checkpoint = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint.write_bytes(b"first\n")

    teacher._write_resume_snapshot(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        completed_scenario_ids=[1, 2],
    )

    start = checkpoint.stat().st_size
    _append_checkpoint(checkpoint, b"second\n")
    teacher._append_resume_delta(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        scenario_id=3,
        complete=True,
        checkpoint_start=start,
    )

    restored = teacher._load_resume_index(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        allowed_scenario_ids=[1, 2, 3, 4],
    )

    assert restored == {1, 2, 3}


def test_resume_index_rejects_unindexed_checkpoint_tail(tmp_path) -> None:
    checkpoint = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint.write_bytes(b"first\n")
    teacher._write_resume_snapshot(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        completed_scenario_ids=[1],
    )
    _append_checkpoint(checkpoint, b"unindexed\n")

    assert teacher._load_resume_index(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        allowed_scenario_ids=[1, 2],
    ) is None


def test_resume_index_rejects_different_contract(tmp_path) -> None:
    checkpoint = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint.write_bytes(b"first\n")
    teacher._write_resume_snapshot(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        completed_scenario_ids=[1],
    )

    assert teacher._load_resume_index(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-b",
        allowed_scenario_ids=[1],
    ) is None


def test_resume_index_delta_can_mark_scenario_retryable(tmp_path) -> None:
    checkpoint = tmp_path / "teacher_checkpoint.jsonl"
    checkpoint.write_bytes(b"first\n")
    teacher._write_resume_snapshot(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        completed_scenario_ids=[1, 2],
    )

    start = checkpoint.stat().st_size
    _append_checkpoint(checkpoint, b"retry\n")
    teacher._append_resume_delta(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        scenario_id=2,
        complete=False,
        checkpoint_start=start,
    )

    assert teacher._load_resume_index(
        checkpoint_path=checkpoint,
        contract_fingerprint="contract-a",
        allowed_scenario_ids=[1, 2],
    ) == {1}
