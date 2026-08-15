from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from grid_topology_ai.teacher_runtime_profile import (
    TeacherRuntimeProfiler,
    cache_counter_delta,
    cache_snapshot,
    estimate_object_bytes,
)
from scripts.diagnostics.profile_teacher_runtime import load_task_config


def test_importing_profile_cli_does_not_import_teacher_stack():
    code = "\n".join(
        [
            "import sys",
            "import scripts.diagnostics.profile_teacher_runtime",
            (
                "assert 'scripts.self_play.generate_impact_teacher_redispatch' "
                "not in sys.modules"
            ),
            (
                "assert 'scripts.self_play.generate_impact_teacher_parallel_fast' "
                "not in sys.modules"
            ),
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_runtime_profiler_preserves_call_result_and_restores_method():
    class Target:
        def value(self, left: int, *, right: int) -> tuple[int, int]:
            return left, right

    original = Target.value
    profiler = TeacherRuntimeProfiler()
    profiler.patch(Target, "value", "target.value")

    assert Target().value(3, right=7) == (3, 7)

    snapshot = profiler.snapshot()
    assert snapshot["calls"]["target.value"] == 1
    assert snapshot["timers_sec"]["target.value"] >= 0.0

    profiler.restore()
    assert Target.value is original


def test_runtime_profiler_records_exception_without_changing_it():
    class Target:
        def fail(self) -> None:
            raise ValueError("expected")

    profiler = TeacherRuntimeProfiler()
    profiler.patch(Target, "fail", "target.fail")

    with pytest.raises(ValueError, match="expected"):
        Target().fail()

    snapshot = profiler.snapshot()
    assert snapshot["calls"]["target.fail"] == 1
    assert snapshot["timers_sec"]["target.fail"] >= 0.0
    profiler.restore()


def test_estimate_object_bytes_does_not_double_count_shared_arrays():
    array = np.zeros(4096, dtype=np.float64)

    shared = estimate_object_bytes({"a": array, "b": array})
    independent = estimate_object_bytes(
        {"a": array.copy(), "b": array.copy()}
    )

    assert shared > int(array.nbytes)
    assert independent > shared


def test_cache_snapshot_reports_counters_containers_and_estimated_bytes():
    class Backend:
        def __init__(self) -> None:
            self._cache = {"one": np.ones(64, dtype=np.float64)}
            self._topology_cache = {"topology": [1, 2, 3]}

        def performance_info(self):
            return {
                "hits": 4,
                "misses": 6,
                "exact_cache_hits": 3,
                "ignored_object": object(),
            }

    class ActionSpace:
        def __init__(self) -> None:
            self._structural_action_mask_cache = {"one": np.ones(4, dtype=bool)}

        def cache_info(self):
            return {"hits": 8, "misses": 2}

    snapshot = cache_snapshot(Backend(), ActionSpace())

    assert snapshot["backend"]["hits"] == 4
    assert snapshot["backend"]["misses"] == 6
    assert "ignored_object" not in snapshot["backend"]
    assert snapshot["action_space"]["hits"] == 8
    assert snapshot["containers"]["backend"]["_cache"] == 1
    assert snapshot["containers"]["backend"]["_topology_cache"] == 1
    assert snapshot["estimated_bytes"]["backend"] > 0
    assert snapshot["estimated_bytes"]["action_space"] > 0


def test_cache_counter_delta_uses_only_monotonic_work_counters():
    before = {
        "backend": {
            "hits": 10,
            "misses": 20,
            "exact_cache_hits": 7,
            "tolerant_cache_hits": 2,
        },
        "action_space": {"hits": 30, "misses": 12},
    }
    after = {
        "backend": {
            "hits": 14,
            "misses": 25,
            "exact_cache_hits": 9,
            "tolerant_cache_hits": 3,
        },
        "action_space": {"hits": 38, "misses": 15},
    }

    delta = cache_counter_delta(before, after)

    assert delta["backend"] == {
        "hits": 4,
        "misses": 5,
        "exact_cache_hits": 2,
        "tolerant_cache_hits": 1,
    }
    assert delta["action_space"] == {"hits": 8, "misses": 3}


def test_load_task_config_accepts_checkpoint_payload_and_plain_config(tmp_path):
    plain_path = tmp_path / "plain.json"
    plain_path.write_text(
        json.dumps({"depth": 5, "beam_width": 20}),
        encoding="utf-8",
    )

    nested_path = tmp_path / "nested.json"
    nested_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_config": {"depth": 6, "beam_width": 40},
            }
        ),
        encoding="utf-8",
    )

    assert load_task_config(plain_path) == {"depth": 5, "beam_width": 20}
    assert load_task_config(nested_path) == {"depth": 6, "beam_width": 40}


def test_profiler_snapshot_is_json_serializable():
    profiler = TeacherRuntimeProfiler()

    with profiler.measure("section"):
        pass

    encoded = json.dumps(profiler.snapshot(), allow_nan=False)
    assert "section" in encoded
