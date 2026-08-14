from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from grid_topology_ai.pf_cache_store import PersistentPFCacheStore


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_store_round_trip_survives_reopen(tmp_path: Path) -> None:
    cache_key = _key("first")

    with PersistentPFCacheStore(tmp_path) as store:
        assert not store.contains(cache_key)
        assert store.put(cache_key, b"payload") is True
        assert store.contains(cache_key)
        assert store.get(cache_key) == b"payload"
        assert store.put(cache_key, b"replacement") is False
        assert store.get(cache_key) == b"payload"

    with PersistentPFCacheStore(tmp_path) as reopened:
        assert reopened.contains(cache_key)
        assert reopened.get(cache_key) == b"payload"
        info = reopened.info()
        assert info.entries == 1
        assert info.index_entries == 1
        assert info.database_bytes > 0


def test_store_refreshes_one_key_written_by_another_connection(tmp_path: Path) -> None:
    cache_key = _key("shared")

    with PersistentPFCacheStore(tmp_path) as first:
        with PersistentPFCacheStore(tmp_path) as second:
            assert not second.contains(cache_key)
            assert first.put(cache_key, b"shared payload") is True

            assert not second.contains(cache_key)
            assert second.get(cache_key) == b"shared payload"
            assert second.contains(cache_key)


def test_read_only_store_can_read_but_not_write(tmp_path: Path) -> None:
    cache_key = _key("read-only")
    with PersistentPFCacheStore(tmp_path) as store:
        store.put(cache_key, b"payload")

    with PersistentPFCacheStore(tmp_path, read_only=True) as store:
        assert store.get(cache_key) == b"payload"
        with pytest.raises(PermissionError, match="read-only"):
            store.put(_key("other"), b"other")


def test_payload_version_mismatch_is_a_cache_miss(tmp_path: Path) -> None:
    cache_key = _key("versioned")
    with PersistentPFCacheStore(tmp_path) as store:
        store.put(cache_key, b"payload", payload_version=2)
        assert store.get(cache_key, payload_version=1) is None
        assert store.get(cache_key, payload_version=2) == b"payload"


def test_corrupted_payload_is_not_returned(tmp_path: Path) -> None:
    cache_key = _key("corrupt")
    with PersistentPFCacheStore(tmp_path) as store:
        store.put(cache_key, b"good")
        database_path = store.database_path

    connection = sqlite3.connect(database_path)
    connection.execute(
        "UPDATE entries SET payload = ? WHERE cache_key = ?",
        (sqlite3.Binary(b"bad"), cache_key),
    )
    connection.commit()
    connection.close()

    with PersistentPFCacheStore(tmp_path) as store:
        assert store.get(cache_key) is None


def test_store_does_not_assume_a_machine_specific_cache_path(tmp_path: Path) -> None:
    root = tmp_path / "arbitrary" / "cache-root"
    with PersistentPFCacheStore(root, namespace="warm") as store:
        assert store.database_path == root / "warm" / "cache.sqlite3"
        store.put(_key("portable"), b"payload")

    assert (root / "warm" / "cache.sqlite3").is_file()
