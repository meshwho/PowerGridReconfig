from __future__ import annotations

import filecmp
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _julia_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _windows_safe_correct_network(
    network_path: str,
    *,
    network_module: Any,
    work_root: Path,
    force: bool = False,
) -> str:
    source_path = Path(network_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Network file not found: {source_path}")

    cache_dir = work_root / "gridfm_networks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_source = cache_dir / source_path.name
    corrected_path = cached_source.with_name(
        f"{cached_source.stem}_corrected{cached_source.suffix}"
    )

    source_changed = (
        not cached_source.exists()
        or not filecmp.cmp(source_path, cached_source, shallow=False)
    )

    if source_changed:
        shutil.copy2(source_path, cached_source)
        corrected_path.unlink(missing_ok=True)

    if corrected_path.exists() and not force:
        return str(corrected_path)

    fd, tmp_name = tempfile.mkstemp(suffix=".m", dir=cache_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        julia_code = [
            "using PowerModels",
            "PowerModels.silence()",
            f'data = PowerModels.parse_file("{_julia_path(cached_source)}")',
            f'PowerModels.export_matpower("{_julia_path(tmp_path)}", data)',
        ]

        network_module.run_julia(
            julia_code,
            project=network_module.STATE["project"],
            executable=network_module.executable(),
        )

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Julia produced empty MATPOWER file")

        shutil.move(str(tmp_path), corrected_path)
        return str(corrected_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    import gridfm_datakit.cli as gridfm_cli
    import gridfm_datakit.network as gridfm_network

    if os.name == "nt":
        work_root = Path(os.environ.get("TEMP", tempfile.gettempdir()))

        def correct_network(network_path: str, force: bool = False) -> str:
            return _windows_safe_correct_network(
                network_path,
                network_module=gridfm_network,
                work_root=work_root,
                force=force,
            )

        gridfm_network.correct_network = correct_network

    gridfm_cli.main()


if __name__ == "__main__":
    main()
