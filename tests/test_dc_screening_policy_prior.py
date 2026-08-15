from __future__ import annotations

from pathlib import Path


def test_dc_cache_stores_only_physical_screening_result() -> None:
    root = Path(__file__).resolve().parents[1]
    cache_source = (
        root / "grid_topology_ai" / "cache" / "dc_screening.py"
    ).read_text(encoding="utf-8")
    screener_source = (
        root / "grid_topology_ai" / "search" / "dc_action_screener.py"
    ).read_text(encoding="utf-8")

    assert "policy_prior" not in cache_source
    assert "policy_prior" in screener_source
    assert "_score_from_physical_result" in screener_source
