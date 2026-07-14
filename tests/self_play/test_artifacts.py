from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from grid_topology_ai.self_play.artifacts import (
    load_json,
    save_json,
    save_yaml,
    sha256_file,
)


def test_save_and_load_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = {
        "nested": {"value": 3},
        "items": [1, 2, 3],
        "enabled": True,
        "missing": None,
        "text": "тест",
    }

    save_json(payload, path)

    assert load_json(path) == payload


def test_load_json_rejects_non_object(tmp_path: Path) -> None:
    for index, content in enumerate(("[]", '"string"', "42")):
        path = tmp_path / f"bad_{index}.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError):
            load_json(path)


def test_load_json_exposes_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_json(path)


def test_save_json_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"

    save_json({"value": 1}, path)

    assert path.is_file()


def test_save_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    save_json({"old": True}, path)

    save_json({"new": True}, path)

    assert load_json(path) == {"new": True}


def test_save_json_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    save_json({"value": 1}, path)

    temporary_files = [
        item
        for item in tmp_path.iterdir()
        if item.name.startswith(f".{path.name}.")
        and item.name.endswith(".tmp")
    ]
    assert temporary_files == []


def test_save_yaml_preserves_order_and_unicode(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    payload = {
        "first": 1,
        "second": "значение",
        "third": {"enabled": True},
    }

    save_yaml(payload, path)

    text = path.read_text(encoding="utf-8")
    assert yaml.safe_load(text) == payload
    assert text.splitlines()[0].startswith("first:")
    assert "значение" in text


def test_save_yaml_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.yaml"

    save_yaml({"value": 1}, path)

    assert path.is_file()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    data = b"artifact bytes"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"data")

    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=0)

    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=-1)
