from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PERSISTENT_EXACT_CACHE_SCHEMA_VERSION = 1
PERSISTENT_EXACT_CACHE_DIR_ENV = "POWERGRID_EXACT_PERSISTENT_CACHE_DIR"
PERSISTENT_EXACT_CACHE_MAX_BYTES_ENV = "POWERGRID_EXACT_PERSISTENT_CACHE_MAX_BYTES"
PERSISTENT_EXACT_CACHE_DISABLED_ENV = "POWERGRID_DISABLE_PERSISTENT_EXACT_CACHE"
DEFAULT_PERSISTENT_EXACT_CACHE_BYTES = 8 * 1024 * 1024 * 1024

_MAGIC = b"PGRPF001"
_HEADER = struct.Struct("<8sBB")
_SHAPE = struct.Struct("<II")
_LENGTH = struct.Struct("<I")
_DIGEST_BYTES = 32
_KIND_SUCCESS = 1
_KIND_NOT_CONVERGED = 2
_ACCESS_REFRESH_NS = 60 * 1_000_000_000
_DB_NAME = f"exact_power_flow_v{PERSISTENT_EXACT_CACHE_SCHEMA_VERSION}.sqlite3"


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PersistentPowerFlowSuccess:
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray


@dataclass(frozen=True, slots=True)
class PersistentPowerFlowFailure:
    message: str


PersistentPowerFlowRecord = PersistentPowerFlowSuccess | PersistentPowerFlowFailure


def _append_array(buffer: bytearray, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Persistent PF arrays must be two-dimensional.")
    buffer.extend(_SHAPE.pack(int(array.shape[0]), int(array.shape[1])))
    buffer.extend(array.tobytes(order="C"))


def _encode_success(bus: np.ndarray, branch: np.ndarray, gen: np.ndarray) -> bytes:
    body = bytearray(
        _HEADER.pack(
            _MAGIC,
            PERSISTENT_EXACT_CACHE_SCHEMA_VERSION,
            _KIND_SUCCESS,
        )
    )
    _append_array(body, bus)
    _append_array(body, branch)
    _append_array(body, gen)
    return bytes(body) + hashlib.sha256(body).digest()


def _encode_failure(message: str) -> bytes:
    encoded = str(message).encode("utf-8")
    body = bytearray(
        _HEADER.pack(
            _MAGIC,
            PERSISTENT_EXACT_CACHE_SCHEMA_VERSION,
            _KIND_NOT_CONVERGED,
        )
    )
    body.extend(_LENGTH.pack(len(encoded)))
    body.extend(encoded)
    return bytes(body) + hashlib.sha256(body).digest()


def _read_array(payload: memoryview, offset: int) -> tuple[np.ndarray, int]:
    if offset + _SHAPE.size > len(payload):
        raise ValueError("Truncated persistent PF array header.")
    rows, cols = _SHAPE.unpack_from(payload, offset)
    offset += _SHAPE.size
    count = int(rows) * int(cols)
    byte_count = count * np.dtype(np.float64).itemsize
    end = offset + byte_count
    if end > len(payload):
        raise ValueError("Truncated persistent PF array payload.")
    array = np.frombuffer(payload[offset:end], dtype=np.float64, count=count)
    return _readonly(array.reshape((int(rows), int(cols)))), end


def _decode_record(blob: bytes) -> PersistentPowerFlowRecord:
    if len(blob) < _HEADER.size + _DIGEST_BYTES:
        raise ValueError("Persistent PF cache payload is truncated.")

    body = blob[:-_DIGEST_BYTES]
    expected_digest = blob[-_DIGEST_BYTES:]
    if hashlib.sha256(body).digest() != expected_digest:
        raise ValueError("Persistent PF cache payload checksum mismatch.")

    magic, version, kind = _HEADER.unpack_from(body, 0)
    if magic != _MAGIC or int(version) != PERSISTENT_EXACT_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported persistent PF cache payload schema.")

    payload = memoryview(body)
    offset = _HEADER.size
    if kind == _KIND_SUCCESS:
        bus, offset = _read_array(payload, offset)
        branch, offset = _read_array(payload, offset)
        gen, offset = _read_array(payload, offset)
        if offset != len(payload):
            raise ValueError("Persistent PF success payload has trailing bytes.")
        return PersistentPowerFlowSuccess(bus=bus, branch=branch, gen=gen)

    if kind == _KIND_NOT_CONVERGED:
        if offset + _LENGTH.size > len(payload):
            raise ValueError("Truncated persistent PF failure payload.")
        (length,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        end = offset + int(length)
        if end != len(payload):
            raise ValueError("Persistent PF failure payload length mismatch.")
        return PersistentPowerFlowFailure(
            bytes(payload[offset:end]).decode("utf-8")
        )

    raise ValueError("Unknown persistent PF cache record kind.")


def _env_disabled() -> bool:
    value = os.environ.get(PERSISTENT_EXACT_CACHE_DISABLED_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class PersistentExactPowerFlowCache:
    """Bounded cross-process L2 for exact AC power-flow outcomes.

    SQLite is only a durable key/value store. The caller supplies the exact
    SHA-256 physical identity; no fuzzy lookup, warm-start selection, or solver
    transformation exists in this layer.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = DEFAULT_PERSISTENT_EXACT_CACHE_BYTES,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max(int(max_bytes), 256 * 1024)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / _DB_NAME
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.corruptions = 0
        self._disabled = False
        self._connection: sqlite3.Connection | None = None
        self._payload_budget = max(int(self.max_bytes * 0.90), 64 * 1024)
        self._open()

    @classmethod
    def from_environment(cls) -> "PersistentExactPowerFlowCache | None":
        if _env_disabled():
            return None
        raw_dir = os.environ.get(PERSISTENT_EXACT_CACHE_DIR_ENV)
        if not raw_dir:
            return None
        raw_max = os.environ.get(PERSISTENT_EXACT_CACHE_MAX_BYTES_ENV)
        try:
            max_bytes = (
                DEFAULT_PERSISTENT_EXACT_CACHE_BYTES
                if not raw_max
                else int(raw_max)
            )
            return cls(raw_dir, max_bytes=max_bytes)
        except (OSError, ValueError):
            return None

    def _open(self) -> None:
        try:
            connection = sqlite3.connect(
                str(self.path),
                timeout=5.0,
                check_same_thread=False,
            )
            connection.execute("PRAGMA busy_timeout=1000")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            max_pages = max(self.max_bytes // max(page_size, 1), 64)
            connection.execute(f"PRAGMA max_page_count={int(max_pages)}")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, PERSISTENT_EXACT_CACHE_SCHEMA_VERSION):
                connection.close()
                self._disabled = True
                return
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "cache_key BLOB PRIMARY KEY, "
                "payload BLOB NOT NULL, "
                "size_bytes INTEGER NOT NULL, "
                "accessed_ns INTEGER NOT NULL"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS entries_accessed_ns "
                "ON entries(accessed_ns)"
            )
            if version == 0:
                connection.execute(
                    f"PRAGMA user_version={PERSISTENT_EXACT_CACHE_SCHEMA_VERSION}"
                )
            connection.commit()
            self._connection = connection
        except (sqlite3.Error, OSError, ValueError):
            self._disabled = True
            self._connection = None

    def _delete_key(self, key: bytes) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.execute(
                "DELETE FROM entries WHERE cache_key = ?",
                (sqlite3.Binary(key),),
            )
            connection.commit()
        except sqlite3.Error:
            pass

    def lookup(self, key: bytes) -> PersistentPowerFlowRecord | None:
        connection = self._connection
        if self._disabled or connection is None:
            self.misses += 1
            return None
        try:
            row = connection.execute(
                "SELECT payload, accessed_ns FROM entries WHERE cache_key = ?",
                (sqlite3.Binary(key),),
            ).fetchone()
        except sqlite3.Error:
            self.misses += 1
            return None
        if row is None:
            self.misses += 1
            return None

        try:
            record = _decode_record(bytes(row[0]))
        except (ValueError, UnicodeError, OverflowError):
            self.corruptions += 1
            self.misses += 1
            self._delete_key(key)
            return None

        now = time.time_ns()
        if now - int(row[1]) >= _ACCESS_REFRESH_NS:
            try:
                connection.execute(
                    "UPDATE entries SET accessed_ns = ? WHERE cache_key = ?",
                    (now, sqlite3.Binary(key)),
                )
                connection.commit()
            except sqlite3.Error:
                pass
        self.hits += 1
        return record

    def _store_blob(self, key: bytes, payload: bytes) -> bool:
        connection = self._connection
        if self._disabled or connection is None:
            return False
        size_bytes = len(payload) + len(key) + 64
        if size_bytes > self._payload_budget:
            return False

        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT size_bytes FROM entries WHERE cache_key = ?",
                (sqlite3.Binary(key),),
            ).fetchone()
            current = int(
                connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
                ).fetchone()[0]
            )
            if existing is not None:
                current -= int(existing[0])
                connection.execute(
                    "DELETE FROM entries WHERE cache_key = ?",
                    (sqlite3.Binary(key),),
                )

            while current + size_bytes > self._payload_budget:
                victims = connection.execute(
                    "SELECT cache_key, size_bytes FROM entries "
                    "ORDER BY accessed_ns ASC LIMIT 32"
                ).fetchall()
                if not victims:
                    connection.execute("ROLLBACK")
                    return False
                for victim_key, victim_size in victims:
                    connection.execute(
                        "DELETE FROM entries WHERE cache_key = ?",
                        (victim_key,),
                    )
                    current -= int(victim_size)
                    self.evictions += 1
                    if current + size_bytes <= self._payload_budget:
                        break

            connection.execute(
                "INSERT INTO entries(cache_key, payload, size_bytes, accessed_ns) "
                "VALUES (?, ?, ?, ?)",
                (
                    sqlite3.Binary(key),
                    sqlite3.Binary(payload),
                    size_bytes,
                    time.time_ns(),
                ),
            )
            connection.execute("COMMIT")
            return True
        except sqlite3.Error:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return False

    def store_success(
        self,
        key: bytes,
        *,
        bus: np.ndarray,
        branch: np.ndarray,
        gen: np.ndarray,
    ) -> bool:
        return self._store_blob(key, _encode_success(bus, branch, gen))

    def store_not_converged(self, key: bytes, message: str) -> bool:
        return self._store_blob(key, _encode_failure(message))

    def discard(self, key: bytes) -> None:
        self._delete_key(key)

    def clear(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.execute("DELETE FROM entries")
            connection.commit()
        except sqlite3.Error:
            pass

    def info(self) -> dict[str, int | bool | str]:
        connection = self._connection
        entries = 0
        payload_bytes = 0
        if connection is not None and not self._disabled:
            try:
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM entries"
                ).fetchone()
                entries = int(row[0])
                payload_bytes = int(row[1])
            except sqlite3.Error:
                pass

        disk_bytes = 0
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                disk_bytes += int(candidate.stat().st_size)
            except OSError:
                pass

        return {
            "enabled": not self._disabled and connection is not None,
            "path": str(self.path),
            "entries": entries,
            "payload_bytes": payload_bytes,
            "max_bytes": int(self.max_bytes),
            "disk_bytes": disk_bytes,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "corruptions": int(self.corruptions),
        }

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass

    def __del__(self) -> None:
        self.close()
