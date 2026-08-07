from pathlib import Path

from scripts.data.gridfm_compat_cli import (
    _julia_path,
    _windows_safe_correct_network,
)


def test_julia_path_uses_forward_slashes() -> None:
    path = r"C:\Users\timof\AppData\Local\grid.m"

    assert _julia_path(path) == "C:/Users/timof/AppData/Local/grid.m"


def test_corrected_network_is_written_under_work_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "installed_package"
    source_dir.mkdir()
    source_path = source_dir / "pglib_opf_case118_ieee.m"
    source_path.write_text("function mpc = case118\n", encoding="utf-8")

    work_root = tmp_path / "work"
    calls: list[list[str]] = []

    class FakeNetworkModule:
        STATE = {"project": "test-project"}

        @staticmethod
        def executable() -> str:
            return "julia"

        @staticmethod
        def run_julia(code, *, project, executable) -> None:
            calls.append(list(code))
            output_path = code[-1].split('"')[1]
            Path(output_path).write_text("corrected network\n", encoding="utf-8")

    corrected = Path(
        _windows_safe_correct_network(
            str(source_path),
            network_module=FakeNetworkModule,
            work_root=work_root,
        )
    )

    assert corrected.parent == work_root / "gridfm_networks"
    assert corrected.name == "pglib_opf_case118_ieee_corrected.m"
    assert corrected.read_text(encoding="utf-8") == "corrected network\n"
    assert not (source_dir / "pglib_opf_case118_ieee_corrected.m").exists()

    assert len(calls) == 1
    assert "\\" not in calls[0][2]
    assert "\\" not in calls[0][3]
