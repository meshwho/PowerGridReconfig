from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_physical_objective_has_no_heavy_or_cyclic_imports():
    text = (ROOT / "grid_topology_ai/physical_objective.py").read_text()
    for token in ("numpy", "pandas", "torch", "grid_topology_ai.environment", "grid_topology_ai.search", "grid_topology_ai.self_play"):
        assert token not in text


def test_core_files_import_physical_objective_contract():
    files = [
        "data_adapter.py", "reward.py", "environment.py", "search/mcts.py",
        "search/continuation_gate.py", "self_play/generation.py",
        "evaluation/metrics.py", "evaluation/checkpoint.py",
    ]
    for rel in files:
        assert "physical_objective" in (ROOT / "grid_topology_ai" / rel).read_text()


def test_selected_literal_threshold_comparisons_were_removed():
    targets = [
        "search/continuation_gate.py", "self_play/generation.py",
        "evaluation/metrics.py", "data_adapter.py",
    ]
    forbidden = ("loading - 100.0", "loading - 120.0", "final_loading > 100.0")
    for rel in targets:
        text = (ROOT / "grid_topology_ai" / rel).read_text()
        for token in forbidden:
            assert token not in text
