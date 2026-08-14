from __future__ import annotations

import hashlib
import sqlite3
import time
import zlib
from dataclasses import dataclass
from pathlib import Path


_SCHEMA_VERSION = 2
_DEFAULT_PAYLOAD_VERSION = 1
_DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024**3
_COMPRESSION_LEVEL = 1


@dataclass(frozen=True)
class PFCacheStoreInfo:
    entries: int
    index_entries: int
    database_bytes: int
    payload_bytes: int
    max_payload_bytes: int | None


class PersistentPFCacheStore:
    """Bounded persistent fingerprint -> payload storage.

    Payloads are compressed before they reach SQLite and the store evicts the
    oldest entries once its logical payload budget is exceeded. Version 2 uses
    a new database file so an oversized legacy ``cache.sqlite3`` is never
    reopened accidentally after upgrading.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: str = "exact",
        read_only: bool = False,
        timeout: float = 30.0,
        max_payload_bytes: int | None = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        namespace = str(namespace).strip()
        if not namespace or namespace in {".", ".."}:
            raise ValueError("namespace must be a normal directory name.")
        if "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must not contain path separators.")

        if max_payload_bytes is None:
            self.max_payload_bytes = None
        else:
            self.max_payload_bytes = int(max_payload_bytes)
            if self.max_payload_bytes <= 0:
                raise ValueError("max_payload_bytes must be positive or None.")

        self.root = Path(root).expanduser()
        self.namespace = namespace
        self.read_only = bool(read_only)
        self.directory = self.root / namespace
        self.database_path = self.directory / "cache_v2.sqlite3"

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
            new_database = not self.database_path.exists()
            self._connection = sqlite3.connect(
                self.database_path,
                timeout=float(timeout),
            )
            if new_database:
                self._connection.execute("PRAGMA auto_vacuum=FULL")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA wal_autocheckpoint=1000")
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
                payload_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS entries_created_at "
            "ON entries(created_at, cache_key)"
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(name, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(name, value) VALUES('payload_bytes', '0')"
        )
        row = self._connection.execute(
            "SELECT COALESCE(SUM(payload_bytes), 0) FROM entries"
        ).fetchone()
        total = 0 if row is None else int(row[0])
        self._connection.execute(
            "UPDATE metadata SET value = ? WHERE name = 'payload_bytes'",
            (str(total),),
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
            self._index.pop(cache_key, None)
            return None
        rowid = int(row[0])
        self._index[cache_key] = rowid
        return rowid

    def contains(self, cache_key: str) -> bool:
        key = self._validate_key(cache_key)
        return self._refresh_key(key) is not None

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
            SELECT payload, payload_sha256, payload_version, payload_bytes
            FROM entries
            WHERE rowid = ? AND cache_key = ?
            """,
            (rowid, key),
        ).fetchone()
        if row is None:
            rowid = self._refresh_key(key)
            if rowid is None:
                return None
            row = self._connection.execute(
                """
                SELECT payload, payload_sha256, payload_version, payload_bytes
                FROM entries
                WHERE rowid = ? AND cache_key = ?
                """,
                (rowid, key),
            ).fetchone()
        if row is None:
            return None

        packed = bytes(row[0])
        stored_sha256 = str(row[1])
        stored_version = int(row[2])
        stored_bytes = int(row[3])
        if stored_version != int(payload_version):
            return None
        if stored_bytes != len(packed):
            return None

        try:
            payload = zlib.decompress(packed)
        except zlib.error:
            return None
        if hashlib.sha256(payload).hexdigest() != stored_sha256:
            return None
        return payload

    def _payload_total_locked(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE name = 'payload_bytes'"
        ).fetchone()
        if row is None:
            raise RuntimeError("Persistent PF cache payload counter is missing.")
        return int(row[0])

    def _set_payload_total_locked(self, value: int) -> None:
        self._connection.execute(
            "UPDATE metadata SET value = ? WHERE name = 'payload_bytes'",
            (str(max(int(value), 0)),),
        )

    def _evict_locked(self, total: int) -> tuple[int, list[str]]:
        if self.max_payload_bytes is None:
            return total, []

        removed: list[str] = []
        while total > self.max_payload_bytes:
            victim = self._connection.execute(
                """
                SELECT cache_key, payload_bytes
                FROM entries
                ORDER BY rowid ASC
                LIMIT 1
                """
            ).fetchone()
            if victim is None:
                break
            victim_key = str(victim[0])
            victim_bytes = int(victim[1])
            self._connection.execute(
                "DELETE FROM entries WHERE cache_key = ?",
                (victim_key,),
            )
            total -= victim_bytes
            removed.append(victim_key)

        return max(total, 0), removed

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
        packed = zlib.compress(value, level=_COMPRESSION_LEVEL)
        if (
            self.max_payload_bytes is not None
            and len(packed) > self.max_payload_bytes
        ):
            return False

        payload_sha256 = hashlib.sha256(value).hexdigest()
        created_at = float(time.time())

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                "SELECT rowid FROM entries WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                self._connection.rollback()
                self._index[key] = int(existing[0])
                return False

            cursor = self._connection.execute(
                """
                INSERT INTO entries(
                    cache_key,
                    payload,
                    payload_sha256,
                    payload_version,
                    payload_bytes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    sqlite3.Binary(packed),
                    payload_sha256,
                    int(payload_version),
                    len(packed),
                    created_at,
                ),
            )
            total = self._payload_total_locked() + len(packed)
            total, removed = self._evict_locked(total)
            self._set_payload_total_locked(total)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

        for removed_key in removed:
            self._index.pop(removed_key, None)
        self._index[key] = int(cursor.lastrowid)
        return key not in removed

    def _database_bytes(self) -> int:
        total = 0
        for path in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            try:
                total += int(path.stat().st_size)
            except OSError:
                pass
        return total

    def info(self) -> PFCacheStoreInfo:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM entries"
        ).fetchone()
        entries = 0 if row is None else int(row[0])
        payload_row = self._connection.execute(
            "SELECT value FROM metadata WHERE name = 'payload_bytes'"
        ).fetchone()
        payload_bytes = 0 if payload_row is None else int(payload_row[0])
        return PFCacheStoreInfo(
            entries=entries,
            index_entries=len(self._index),
            database_bytes=self._database_bytes(),
            payload_bytes=payload_bytes,
            max_payload_bytes=self.max_payload_bytes,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PersistentPFCacheStore:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
