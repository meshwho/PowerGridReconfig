from __future__ import annotations

from pathlib import Path


def test_action_space_contains_no_legacy_parallel_cache_layers() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "grid_topology_ai" / "action_space.py").read_text(
        encoding="utf-8"
    )

    for fragment in (
        "_structural_action_mask_cache",
        "_operational_action_mask_cache",
        "_valid_actions_cache",
        "_connectivity_mask_cache",
        "_make_cache_key",
        "_loading_signature",
    ):
        assert fragment not in source


def test_dc_screener_does_not_depend_on_power_flow_cache_internals() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "grid_topology_ai" / "search" / "dc_action_screener.py"
    ).read_text(encoding="utf-8")

    assert "_make_cache_key_from_state" not in source
    assert "._cache" not in source
    assert "DCScreeningCache" in source


def test_cache_policy_is_kept_under_cache_package() -> None:
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "grid_topology_ai" / "cache"

    assert (cache_dir / "byte_lru.py").exists()
    assert (cache_dir / "exact_power_flow.py").exists()
    assert (cache_dir / "structural_topology.py").exists()
    assert (cache_dir / "dc_screening.py").exists()
