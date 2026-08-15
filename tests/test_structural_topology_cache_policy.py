from __future__ import annotations

from pathlib import Path


def test_structural_cache_module_contains_no_ac_power_flow_solver_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "grid_topology_ai" / "cache" / "structural_topology.py"
    ).read_text(encoding="utf-8")

    assert "runpf" not in source
    assert "rundcpf" not in source
    assert "pypower_backend" not in source


def test_dc_screening_cache_module_contains_no_ac_solver_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "grid_topology_ai" / "cache" / "dc_screening.py"
    ).read_text(encoding="utf-8")

    assert "runpf" not in source
    assert "pypower_backend" not in source
