from __future__ import annotations

from pathlib import Path


SELF_PLAY_FILES_WITH_REMOVED_BROAD_CATCHES = (
    Path("grid_topology_ai/self_play/replay.py"),
    Path("grid_topology_ai/self_play/iteration.py"),
    Path("grid_topology_ai/self_play/pipeline.py"),
)


def test_silent_self_play_fallback_files_do_not_use_broad_exception_catches() -> None:
    bad = [
        str(path)
        for path in SELF_PLAY_FILES_WITH_REMOVED_BROAD_CATCHES
        if "except Exception" in path.read_text(encoding="utf-8")
    ]

    assert bad == []


def test_self_play_broad_exception_boundaries_are_documented() -> None:
    matches = sorted(
        path.as_posix()
        for path in Path("grid_topology_ai/self_play").glob("*.py")
        if "except Exception" in path.read_text(encoding="utf-8")
    )

    assert matches == [
        "grid_topology_ai/self_play/examples.py",
        "grid_topology_ai/self_play/provenance.py",
        "grid_topology_ai/self_play/stages.py",
    ]

    examples_source = Path(
        "grid_topology_ai/self_play/examples.py"
    ).read_text(encoding="utf-8")
    assert examples_source.count("except Exception") == 1
    assert "del self.examples[original_count:]" in examples_source
    assert "path.unlink(missing_ok=True)" in examples_source

    provenance_source = Path(
        "grid_topology_ai/self_play/provenance.py"
    ).read_text(encoding="utf-8")
    assert provenance_source.count("except Exception") == 1
    assert "pd.read_parquet(path, columns=requested)" in provenance_source
    assert "frame = pd.read_parquet(path)" in provenance_source

    stages_source = Path(
        "grid_topology_ai/self_play/stages.py"
    ).read_text(encoding="utf-8")
    assert stages_source.count("except Exception") == 1
    assert "traceback.print_exc()" in stages_source


def test_evaluation_worker_broad_exception_boundary_is_documented() -> None:
    text = Path("grid_topology_ai/evaluation/checkpoint.py").read_text(
        encoding="utf-8"
    )

    assert text.count("except Exception") == 1
    assert "Intentional process-worker boundary" in text
