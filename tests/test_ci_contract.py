from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_contract() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    required = (
        "actions/checkout@v7",
        "actions/setup-python@v6",
        'python-version: "3.11"',
        "python -m pytest -q",
        "python -m pip check",
        "windows-latest",
        "ubuntu-latest",
        "--plan-only",
        "python -m build",
    )
    forbidden = (
        "continue-on-error",
        "pull_request_target",
        "np.in1d =",
        "setattr(np",
        "hasattr(np",
    )

    for pattern in required:
        assert pattern in text
    for pattern in forbidden:
        assert pattern not in text
