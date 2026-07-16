from __future__ import annotations

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    path = ROOT / "pyproject.toml"
    assert path.exists()
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_pyproject_build_and_python_contract() -> None:
    data = _pyproject()

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["project"]["requires-python"] == ">=3.11,<3.12"


def test_core_and_optional_dependency_contract() -> None:
    project = _pyproject()["project"]
    deps = project["dependencies"]
    extras = project["optional-dependencies"]

    assert any(dep.startswith("numpy>=") and "<2.0" in dep for dep in deps)
    assert "PYPOWER==5.1.19" in deps
    assert any(dep.startswith("scipy") for dep in deps)
    assert any(dep.startswith("torch") for dep in deps)
    assert "pandapower==2.14.11" in extras["data"]
    assert any(dep.startswith("pytest") for dep in extras["test"])
    assert any(dep.startswith("build") for dep in extras["test"])


def test_constraints_and_requirements_contract() -> None:
    constraints = (ROOT / "constraints" / "py311.txt").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "numpy==1.26.4" in constraints
    assert "PYPOWER==5.1.19" in constraints
    assert "constraints/py311.txt" in requirements
    assert ".[data,test,train]" in requirements


def test_production_backend_has_no_numpy_runtime_monkeypatch() -> None:
    text = (ROOT / "grid_topology_ai" / "pypower_backend.py").read_text(encoding="utf-8")

    forbidden = ("np.in1d =", "setattr(np", "hasattr(np", "np.isin")
    assert all(pattern not in text for pattern in forbidden)
