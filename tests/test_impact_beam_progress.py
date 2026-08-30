from __future__ import annotations

from grid_topology_ai.search.teacher import (
    ImpactBeamSearchConfig,
    ImpactBeamSearchPlanner,
)
from grid_topology_ai.search.teacher import LODFScreenedImpactBeamSearchPlanner


class _FakeProgressBar:
    def __init__(self, total: int | None) -> None:
        self.n = 0
        self.total = total
        self.postfix = None
        self.refresh_count = 0
        self.closed = False

    def update(self, delta: int) -> None:
        self.n += int(delta)

    def set_postfix(self, postfix: dict) -> None:
        self.postfix = postfix

    def refresh(self) -> None:
        self.refresh_count += 1

    def close(self) -> None:
        self.closed = True


def test_progress_update_tracks_real_evaluation_count() -> None:
    planner = ImpactBeamSearchPlanner(
        ImpactBeamSearchConfig(progress_update_every=10)
    )
    bar = _FakeProgressBar(total=100)
    planner._progress_bar = bar

    planner.evaluated_actions = 9
    planner._update_progress()
    assert bar.n == 0

    planner.evaluated_actions = 10
    planner._update_progress(postfix={"evaluated": 10})
    assert bar.n == 10
    assert bar.postfix == {"evaluated": 10}

    planner.evaluated_actions = 20
    planner._update_progress()
    assert bar.n == 20


def test_progress_close_flushes_remainder_and_uses_actual_total() -> None:
    planner = ImpactBeamSearchPlanner(
        ImpactBeamSearchConfig(progress_update_every=10)
    )
    bar = _FakeProgressBar(total=100)
    planner._progress_bar = bar

    planner.evaluated_actions = 20
    planner._update_progress()
    planner.evaluated_actions = 23
    planner._close_progress()

    assert bar.n == 23
    assert bar.total == 23
    assert bar.refresh_count == 1
    assert bar.closed is True
    assert planner._progress_bar is None


def test_lodf_progress_estimate_uses_screened_candidate_limit() -> None:
    config = ImpactBeamSearchConfig(
        max_depth=5,
        beam_width=20,
        candidate_pool_size=160,
    )
    planner = LODFScreenedImpactBeamSearchPlanner(
        config=config,
        lodf_screen_top_k=70,
    )

    assert planner._estimated_progress_total() == 70 * (1 + 20 * 4)


def test_unscreened_progress_estimate_keeps_full_candidate_pool() -> None:
    config = ImpactBeamSearchConfig(
        max_depth=5,
        beam_width=20,
        candidate_pool_size=160,
    )
    planner = ImpactBeamSearchPlanner(config)

    assert planner._estimated_progress_total() == 160 * (1 + 20 * 4)
