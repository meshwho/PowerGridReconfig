from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

JsonObject = dict[str, Any]


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


def load_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")

    return dict(payload)


def save_json(payload: Mapping[str, Any], path: Path) -> Path:
    content = json.dumps(
        dict(payload),
        indent=2,
        ensure_ascii=False,
    )

    return _atomic_write_text(path=path, content=content)


def save_yaml(payload: Mapping[str, Any], path: Path) -> Path:
    content = yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        sort_keys=False,
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
