from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _atomic_write_text(
    *,
    path: Path,
    content: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)

        temporary_path.replace(path)
        return path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_json(payload: Mapping[str, Any], path: Path) -> Path:
    content = json.dumps(
        dict(payload),
        indent=2,
        ensure_ascii=False,
    )

    return _atomic_write_text(path=path, content=content)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(content).hexdigest()


def sha256_files(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
) -> str:
    files = [Path(path) for path in paths]

    if not files:
        raise ValueError("paths must not be empty")

    missing = [
        path
        for path in files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Files are missing while calculating a combined hash: "
            + ", ".join(str(path) for path in missing)
        )

    root_path = (
        None
        if root is None
        else Path(root).resolve()
    )

    def file_name(path: Path) -> str:
        resolved = path.resolve()

        if root_path is None:
            return resolved.as_posix()

        try:
            return resolved.relative_to(root_path).as_posix()
        except ValueError:
            return resolved.as_posix()

    digest = hashlib.sha256()

    for path in sorted(files, key=file_name):
        digest.update(file_name(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")

    return digest.hexdigest()


def read_git_state(project_root: Path) -> dict[str, object]:
    project_root = Path(project_root)

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    revision = run_git("rev-parse", "HEAD")

    if not revision:
        raise RuntimeError(
            f"Could not determine Git revision in {project_root}"
        )

    return {
        "revision": revision,
        "dirty": bool(run_git("status", "--porcelain")),
    }
