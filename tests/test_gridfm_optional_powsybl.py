from __future__ import annotations

import ast
from pathlib import Path


def test_gridfm_shim_does_not_import_optional_powsybl_at_package_import() -> None:
    source_path = Path("gridfm_datakit/__init__.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "gridfm_datakit":
            continue

        imported = {alias.name for alias in node.names}
        assert "powsybl" not in imported
