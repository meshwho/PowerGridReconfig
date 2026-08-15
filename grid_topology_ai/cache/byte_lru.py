from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class ByteLRUInfo:
    entries: int
    bytes: int
    max_bytes: int
    evictions: int


class ByteLRUCache(Generic[K, V]):
    """Small LRU with an explicit caller-supplied byte budget.

    Values are admitted only when the full entry fits in ``max_bytes``. The
    cache never estimates object graphs itself; callers provide the owned byte
    size so cache policy remains predictable and cheap in hot paths.
    """

    def __init__(self, max_bytes: int) -> None:
        max_bytes = int(max_bytes)
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative.")

        self.max_bytes = max_bytes
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._bytes = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        return int(self._bytes)

    @property
    def evictions(self) -> int:
        return int(self._evictions)

    def get(self, key: K) -> V | None:
        stored = self._entries.get(key)
        if stored is None:
            return None

        self._entries.move_to_end(key)
        return stored[0]

    def put(self, key: K, value: V, *, size_bytes: int) -> bool:
        size_bytes = int(size_bytes)
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative.")

        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]

        if self.max_bytes == 0 or size_bytes > self.max_bytes:
            return False

        self._entries[key] = (value, size_bytes)
        self._bytes += size_bytes

        while self._bytes > self.max_bytes:
            _old_key, (_old_value, old_size) = self._entries.popitem(last=False)
            self._bytes -= old_size
            self._evictions += 1

        return key in self._entries

    def discard(self, key: K) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]

    def clear(self, *, reset_evictions: bool = False) -> None:
        self._entries.clear()
        self._bytes = 0
        if reset_evictions:
            self._evictions = 0

    def info(self) -> ByteLRUInfo:
        return ByteLRUInfo(
            entries=len(self._entries),
            bytes=int(self._bytes),
            max_bytes=int(self.max_bytes),
            evictions=int(self._evictions),
        )
