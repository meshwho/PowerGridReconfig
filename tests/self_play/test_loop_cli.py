from __future__ import annotations

import pytest

from scripts.self_play import loop as loop_module


def test_build_parser_supports_config() -> None:
    args = loop_module.build_parser().parse_args(["config.yaml"])
    assert args.config == "config.yaml"


def test_build_parser_supports_validate_only() -> None:
    args = loop_module.build_parser().parse_args(["config.yaml", "--validate-only"])
    assert args.validate_only is True


def test_build_parser_supports_plan_only() -> None:
    args = loop_module.build_parser().parse_args(["config.yaml", "--plan-only"])
    assert args.plan_only is True


def test_build_parser_supports_resume() -> None:
    args = loop_module.build_parser().parse_args(["config.yaml", "--resume"])
    assert args.resume is True


def test_main_returns_zero(monkeypatch) -> None:
    monkeypatch.setattr(loop_module, "run_loop", lambda **kwargs: None)
    assert loop_module.main(["config.yaml"]) == 0


def test_main_passes_flags_to_run_loop(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(loop_module, "run_loop", lambda **kwargs: calls.append(kwargs))
    loop_module.main(["config.yaml", "--validate-only", "--plan-only", "--resume"])
    assert calls == [{"config_path": "config.yaml", "validate_only": True, "plan_only": True, "resume": True}]


def test_help_works() -> None:
    with pytest.raises(SystemExit) as excinfo:
        loop_module.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
