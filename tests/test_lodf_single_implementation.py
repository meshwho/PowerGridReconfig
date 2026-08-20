from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEACHER_RUNTIME = (
    ROOT
    / "scripts"
    / "self_play"
    / "generate_impact_teacher_redispatch_runtime.py"
)
LODF_MODULE = ROOT / "grid_topology_ai" / "physics" / "lodf.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            assert node.end_lineno is not None
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"Function {name!r} was not found.")


def test_teacher_runtime_delegates_lodf_math_to_shared_module() -> None:
    source = _source(TEACHER_RUNTIME)

    # Parse first so this also catches accidental damage while editing the large
    # teacher entrypoint without importing its heavy runtime dependencies.
    ast.parse(source)

    assert "from grid_topology_ai.physics.lodf import (" in source
    assert "np.linalg.pinv" not in source
    assert "BRANCH_FEATURE_COLUMNS" not in source

    wrapper = _function_source(source, "rank_actions_by_lodf_screening")
    assert "build_lodf_structure(state)" in wrapper
    assert "rank_actions_with_lodf_structure(" in wrapper
    assert "np.linalg" not in wrapper
    assert "branch_features" not in wrapper


def test_lodf_safety_score_has_one_production_definition() -> None:
    lodf_source = _source(LODF_MODULE)
    teacher_source = _source(TEACHER_RUNTIME)

    lodf_tree = ast.parse(lodf_source)
    teacher_tree = ast.parse(teacher_source)

    lodf_definitions = [
        node
        for node in lodf_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "lodf_loading_safety_score"
    ]
    teacher_definitions = [
        node
        for node in teacher_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "lodf_loading_safety_score"
    ]

    assert len(lodf_definitions) == 1
    assert teacher_definitions == []


def test_dense_pseudoinverse_is_only_a_lodf_fallback() -> None:
    lodf_source = _source(LODF_MODULE)
    teacher_source = _source(TEACHER_RUNTIME)

    assert "np.linalg.pinv" in lodf_source
    assert "np.linalg.pinv" not in teacher_source
    assert "splu(" in lodf_source
