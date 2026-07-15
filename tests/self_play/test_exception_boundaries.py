from __future__ import annotations

from pathlib import Path


SELF_PLAY_FILES_WITH_REMOVED_BROAD_CATCHES = (
    Path("grid_topology_ai/self_play/replay.py"),
    Path("grid_topology_ai/self_play/iteration.py"),
    Path("grid_topology_ai/self_play/pipeline.py"),
    Path("grid_topology_ai/self_play/run_state.py"),
)


def test_silent_self_play_fallback_files_do_not_use_broad_exception_catches() -> None:
    bad = [
        str(path)
        for path in SELF_PLAY_FILES_WITH_REMOVED_BROAD_CATCHES
        if "except Exception" in path.read_text(encoding="utf-8")
    ]

    assert bad == []


def test_self_play_broad_exception_inventory_is_limited_to_stage_logger() -> None:
    matches = []
    for path in Path("grid_topology_ai/self_play").glob("*.py"):
        if "except Exception" in path.read_text(encoding="utf-8"):
            matches.append(path.as_posix())

    assert matches == ["grid_topology_ai/self_play/stages.py"]
    assert "Intentional top-level logging boundary" in Path(
        "grid_topology_ai/self_play/stages.py"
    ).read_text(encoding="utf-8")


def test_evaluation_worker_broad_exception_boundary_is_documented() -> None:
    text = Path("grid_topology_ai/evaluation/checkpoint.py").read_text(encoding="utf-8")

    assert text.count("except Exception") == 1
    assert "Intentional process-worker boundary" in text
