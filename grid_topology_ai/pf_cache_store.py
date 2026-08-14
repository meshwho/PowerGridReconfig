from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


_SCHEMA_VERSION = 1
_DEFAULT_PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class PFCacheStoreInfo:
    entries: int
    index_entries: int
    database_bytes: int


class PersistentPFCacheStore:
    """Persistent fingerprint -> payload storage with a hot in-memory index.

    The caller owns the cache root. No machine-specific path is assumed here.
    Each process opens its own store/SQLite connection.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: str = "exact",
        read_only: bool = False,
        timeout: float = 30.0,
    ) -> None:
        namespace = str(namespace).strip()
        if not namespace or namespace in {".", ".."}:
            raise ValueError("namespace must be a normal directory name.")
        if "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must not contain path separators.")

        self.root = Path(root).expanduser()
        self.namespace = namespace
        self.read_only = bool(read_only)
        self.directory = self.root / namespace
        self.database_path = self.directory / "cache.sqlite3"

        if self.read_only:
            if not self.database_path.is_file():
                raise FileNotFoundError(self.database_path)
            uri = f"file:{self.database_path.as_posix()}?mode=ro"
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=float(timeout),
            )
            self._require_schema()
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.database_path,
                timeout=float(timeout),
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._create_schema()

        self._index: dict[str, int] = {}
        self.refresh_index()

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                cache_key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(name, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._connection.commit()
        self._require_schema()

    def _require_schema(self) -> None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE name = 'schema_version'"
        ).fetchone()
        if row is None or int(row[0]) != _SCHEMA_VERSION:
            raise ValueError("Unsupported persistent PF cache schema version.")

    @staticmethod
    def _validate_key(cache_key: str) -> str:
        key = str(cache_key).strip().lower()
        if len(key) != 64:
            raise ValueError("cache_key must be a full SHA-256 hex digest.")
        try:
            bytes.fromhex(key)
        except ValueError as exc:
            raise ValueError("cache_key must be a full SHA-256 hex digest.") from exc
        return key

    def refresh_index(self) -> int:
        rows = self._connection.execute(
            "SELECT rowid, cache_key FROM entries"
        ).fetchall()
        self._index = {
            str(cache_key): int(rowid)
            for rowid, cache_key in rows
        }
        return len(self._index)

    def _refresh_key(self, cache_key: str) -> int | None:
        row = self._connection.execute(
            "SELECT rowid FROM entries WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        rowid = int(row[0])
        self._index[cache_key] = rowid
        return rowid

    def contains(self, cache_key: str) -> bool:
        key = self._validate_key(cache_key)
        return key in self._index

    def get(
        self,
        cache_key: str,
        *,
        payload_version: int = _DEFAULT_PAYLOAD_VERSION,
    ) -> bytes | None:
        key = self._validate_key(cache_key)
        rowid = self._index.get(key)
        if rowid is None:
            rowid = self._refresh_key(key)
        if rowid is None:
            return None

        row = self._connection.execute(
            """
            SELECT payload, payload_sha256, payload_version
            FROM entries
            WHERE rowid = ? AND cache_key = ?
            """,
            (rowid, key),
        ).fetchone()
        if row is None:
            self._index.pop(key, None)
            return None

        payload = bytes(row[0])
        stored_sha256 = str(row[1])
        stored_version = int(row[2])
        if stored_version != int(payload_version):
            return None
        if hashlib.sha256(payload).hexdigest() != stored_sha256:
            return None
        return payload

    def put(
        self,
        cache_key: str,
        payload: bytes,
        *,
        payload_version: int = _DEFAULT_PAYLOAD_VERSION,
    ) -> bool:
        if self.read_only:
            raise PermissionError("Persistent PF cache store is read-only.")

        key = self._validate_key(cache_key)
        value = bytes(payload)
        payload_sha256 = hashlib.sha256(value).hexdigest()

        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO entries(
                cache_key,
                payload,
                payload_sha256,
                payload_version,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                key,
                sqlite3.Binary(value),
                payload_sha256,
                int(payload_version),
                float(time.time()),
            ),
        )
        self._connection.commit()
        inserted = cursor.rowcount == 1
        rowid = self._refresh_key(key)
        if rowid is None:
            raise RuntimeError("Persistent PF cache insert was not visible after commit.")
        return inserted

    def info(self) -> PFCacheStoreInfo:
        row = self._connection.execute("SELECT COUNT(*) FROM entries").fetchone()
        entries = 0 if row is None else int(row[0])
        try:
            database_bytes = int(self.database_path.stat().st_size)
        except OSError:
            database_bytes = 0
        return PFCacheStoreInfo(
            entries=entries,
            index_entries=len(self._index),
            database_bytes=database_bytes,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PersistentPFCacheStore:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
