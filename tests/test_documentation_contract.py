from __future__ import annotations

import re
from pathlib import Path


def test_readme_documents_current_self_play_contract() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    required = [
        "pool-guided",
        "MCTS visit",
        "scenario_id",
        "validation loss",
        "PF_ALG",
        "outcome_value_target",
        "normalization",
        "--plan-only",
        "--validate-only",
        "--resume",
        "iteration_complete.json",
        "pyproject.toml",
        "constraints/py311.txt",
        "docs/self_play.md",
    ]
    for phrase in required:
        assert phrase in text

    forbidden = [
        "The current pipeline is teacher-based supervised learning",
        "The next stage is model-guided planning and self-play",
        "Future self-play layer",
        "MCTS is part of the future",
        "MCTS is experimental infrastructure rather than the main training pipeline",
        "legacy discounted_return_from_step / value_scale",
        "Later, MCTS should produce softer policy distributions",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_readme_local_markdown_links_exist() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+\.md)\)", text)
    assert links
    for link in links:
        assert Path(link).is_file(), link


def test_documented_module_paths_exist() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (scripts\.[A-Za-z0-9_.]+)", text))
    for module in modules:
        assert Path(*module.split(".")).with_suffix(".py").is_file(), module


def test_readme_artifact_tree_places_generation_outputs_under_raw() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    raw_index = text.index("raw/")
    selected_index = text.index("selected_transitions.csv")
    examples_index = text.index("examples.csv")
    generate_log_index = text.index("generate.log")

    assert raw_index < selected_index < examples_index < generate_log_index
