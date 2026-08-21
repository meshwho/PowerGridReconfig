from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def generation_inputs(tmp_path: Path) -> Callable[[tuple[int, ...]], tuple[Path, Path]]:
    """Create the files required by canonical generation preflight checks."""

    def create(scenario_ids: tuple[int, ...] = (1,)) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        for name in ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet"):
            (raw_dir / name).touch()
        transitions = tmp_path / "transitions.csv"
        rows = "".join(f"{scenario_id}\n" for scenario_id in scenario_ids)
        transitions.write_text(f"scenario_id\n{rows}", encoding="utf-8")
        return raw_dir, transitions

    return create
