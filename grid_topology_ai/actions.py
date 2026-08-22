from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Literal

import networkx as nx
import numpy as np

from grid_topology_ai.cache import ByteLRUCache
from grid_topology_ai.config import ActionSpaceConfig
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState


SlotKind = Literal[
    "stop",
    "branch_status",
]

ActionKind = Literal[
    "stop",
    "set_branch_status",
]

ActionType = Literal[
    "do_nothing",
    "switch_off_branch",
    "switch_on_branch",
]

STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT = "stop_plus_branch_status_v1"


def _fingerprint_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def topology_action_payload(
    action_space_config: "ActionSpaceConfig",
    action_layout: object,
) -> dict[str, object]:
    """Serialize the actual action configuration and ordered layout."""

    return {
        "topology_action_config": action_space_config.to_contract_dict(),
        "action_layout": action_layout_to_list(action_layout),
    }


def require_topology_action_payload(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_action_space_config: "ActionSpaceConfig | None" = None,
    expected_action_layout: object | None = None,
) -> tuple["ActionSpaceConfig", tuple["ActionSlot", ...]]:
    """Validate the actual action configuration and ordered layout."""

    if payload.get("topology_action_config") is None:
        raise ValueError(f"Missing topology_action_config for {source}.")
    if payload.get("action_layout") is None:
        raise ValueError(f"Missing action_layout for {source}.")

    raw_config = payload["topology_action_config"]
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid topology_action_config JSON for {source}."
            ) from exc
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Invalid topology_action_config for {source}.")

    observed_config = ActionSpaceConfig.from_contract_mapping(raw_config)
    observed_layout = action_layout_from_value(payload["action_layout"])
    if (
        expected_action_space_config is not None
        and observed_config.to_contract_dict()
        != expected_action_space_config.to_contract_dict()
    ):
        raise ValueError(f"Topology action config mismatch for {source}.")
    if expected_action_layout is not None and tuple(observed_layout) != tuple(
        expected_action_layout
    ):
        raise ValueError(f"Action layout mismatch for {source}.")
    return observed_config, observed_layout


def _non_negative_int(
    name: str,
    value: object,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")

    parsed = int(value)

    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")

    return parsed


def _binary_status(
    name: str,
    value: object,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be either 0 or 1.")

    parsed = float(value)

    if not math.isfinite(parsed) or parsed not in (0.0, 1.0):
        raise ValueError(f"{name} must be either 0 or 1.")

    return int(parsed)


@dataclass(frozen=True, slots=True)
class ActionSlot:
    """
    One stable position in the policy vector.

    A slot identifies the controlled object. It does not
    describe the state-dependent command applied to it.
    """

    action_id: int
    kind: SlotKind
    target_id: int | None
    target_pos: int | None

    def __post_init__(self) -> None:
        action_id = _non_negative_int(
            "action_id",
            self.action_id,
        )

        if self.kind == "stop":
            if action_id != 0:
                raise ValueError("The stop slot must use action_id 0.")

            if self.target_id is not None or self.target_pos is not None:
                raise ValueError("The stop slot must not have a target.")

            return

        if self.kind != "branch_status":
            raise ValueError(f"Unsupported action slot kind: {self.kind!r}.")

        if self.target_id is None or self.target_pos is None:
            raise ValueError("A branch-status slot requires target_id and target_pos.")

        target_id = _non_negative_int(
            "target_id",
            self.target_id,
        )
        target_pos = _non_negative_int(
            "target_pos",
            self.target_pos,
        )

        expected_action_id = 1 + target_pos

        if action_id != expected_action_id:
            raise ValueError(
                "Branch-status action_id must equal "
                f"1 + target_pos. Expected "
                f"{expected_action_id}, got {action_id}."
            )

        object.__setattr__(
            self,
            "action_id",
            action_id,
        )
        object.__setattr__(
            self,
            "target_id",
            target_id,
        )
        object.__setattr__(
            self,
            "target_pos",
            target_pos,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": int(self.action_id),
            "kind": str(self.kind),
            "target_id": (None if self.target_id is None else int(self.target_id)),
            "target_pos": (None if self.target_pos is None else int(self.target_pos)),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
    ) -> "ActionSlot":
        if not isinstance(data, Mapping):
            raise ValueError("Action slot must be a mapping.")

        required = {
            "action_id",
            "kind",
            "target_id",
            "target_pos",
        }

        unknown = set(data) - required

        if unknown:
            raise ValueError(f"Unknown action slot fields: {sorted(unknown)}.")

        missing = required - set(data)

        if missing:
            raise ValueError(f"Missing action slot fields: {sorted(missing)}.")

        return cls(
            action_id=data["action_id"],
            kind=data["kind"],
            target_id=data["target_id"],
            target_pos=data["target_pos"],
        )


@dataclass(frozen=True, slots=True)
class GridFMAction:
    """
    One executable topology action.

    Branch actions use one stable policy slot per branch.
    The current branch status determines whether the command
    opens or closes that branch.
    """

    action_id: int
    action_type: ActionType
    branch_id: int | None = None
    branch_pos: int | None = None
    target_status: int | None = None

    def __post_init__(self) -> None:
        action_id = _non_negative_int(
            "action_id",
            self.action_id,
        )

        if self.action_type == "do_nothing":
            if action_id != 0:
                raise ValueError("do_nothing must use action_id 0.")

            if (
                self.branch_id is not None
                or self.branch_pos is not None
                or self.target_status is not None
            ):
                raise ValueError(
                    "do_nothing must not have a branch target or target_status."
                )

            object.__setattr__(
                self,
                "action_id",
                action_id,
            )
            return

        if self.action_type not in {
            "switch_off_branch",
            "switch_on_branch",
        }:
            raise ValueError(f"Unsupported action type: {self.action_type!r}.")

        if self.branch_id is None or self.branch_pos is None:
            raise ValueError(
                "A branch-status action requires branch_id and branch_pos."
            )

        branch_id = _non_negative_int(
            "branch_id",
            self.branch_id,
        )
        branch_pos = _non_negative_int(
            "branch_pos",
            self.branch_pos,
        )

        expected_action_id = 1 + branch_pos

        if action_id != expected_action_id:
            raise ValueError(
                "Branch action_id must equal "
                f"1 + branch_pos. Expected "
                f"{expected_action_id}, got {action_id}."
            )

        expected_target_status = 0 if self.action_type == "switch_off_branch" else 1

        target_status = (
            expected_target_status
            if self.target_status is None
            else _binary_status(
                "target_status",
                self.target_status,
            )
        )

        if target_status != expected_target_status:
            raise ValueError(
                f"{self.action_type} requires target_status={expected_target_status}."
            )

        object.__setattr__(
            self,
            "action_id",
            action_id,
        )
        object.__setattr__(
            self,
            "branch_id",
            branch_id,
        )
        object.__setattr__(
            self,
            "branch_pos",
            branch_pos,
        )
        object.__setattr__(
            self,
            "target_status",
            target_status,
        )

    @property
    def kind(self) -> ActionKind:
        if self.action_type == "do_nothing":
            return "stop"

        return "set_branch_status"

    @property
    def target_id(self) -> int | None:
        return self.branch_id

    @property
    def target_pos(self) -> int | None:
        return self.branch_pos


def build_branch_action_slots(
    branch_ids: Iterable[int],
) -> tuple[ActionSlot, ...]:
    """
    Build the stable stop-plus-branch policy layout.

    Slot 0 is stop. Slot 1 + branch_pos controls the branch
    stored at that position.
    """

    slots: list[ActionSlot] = [
        ActionSlot(
            action_id=0,
            kind="stop",
            target_id=None,
            target_pos=None,
        )
    ]

    for branch_pos, branch_id in enumerate(branch_ids):
        slots.append(
            ActionSlot(
                action_id=1 + branch_pos,
                kind="branch_status",
                target_id=int(branch_id),
                target_pos=branch_pos,
            )
        )

    return tuple(slots)


def action_layout_to_list(
    action_layout: Iterable[ActionSlot],
) -> list[dict[str, object]]:
    slots = tuple(action_layout)

    if not slots:
        raise ValueError("Action layout must not be empty.")

    action_ids = [int(slot.action_id) for slot in slots]
    expected_action_ids = list(range(len(slots)))

    if action_ids != expected_action_ids:
        raise ValueError(
            "Action layout IDs must be contiguous and "
            f"ordered from 0. Observed {action_ids}."
        )

    return [slot.to_dict() for slot in slots]


def action_layout_from_value(
    value: object,
) -> tuple[ActionSlot, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid action layout JSON.") from exc

    if not isinstance(value, list):
        raise ValueError("Action layout must be a list.")

    slots = tuple(ActionSlot.from_mapping(item) for item in value)

    action_layout_to_list(slots)

    return slots


def action_layout_fingerprint(
    action_layout: Iterable[ActionSlot],
) -> str:
    return _fingerprint_json(action_layout_to_list(action_layout))


def require_branch_status_policy_layout(
    action_layout: Iterable[ActionSlot],
) -> str:
    slots = tuple(action_layout)

    action_layout_to_list(slots)

    if slots[0].kind != "stop":
        raise ValueError("The current policy head requires stop at action_id 0.")

    if any(slot.kind != "branch_status" for slot in slots[1:]):
        raise ValueError(
            "The current policy head supports only branch-status action slots."
        )

    return STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT


def branch_status_signature(
    branch_ids: Iterable[int],
    branch_status: Iterable[float],
) -> tuple[tuple[int, int], ...]:
    """
    Return the canonical branch topology represented by
    stable branch IDs and their current statuses.
    """

    ids = tuple(
        _non_negative_int(
            "branch_id",
            branch_id,
        )
        for branch_id in branch_ids
    )
    statuses = tuple(
        _binary_status(
            "branch_status",
            status,
        )
        for status in branch_status
    )

    if len(ids) != len(statuses):
        raise ValueError("branch_ids and branch_status must have the same length.")

    if len(set(ids)) != len(ids):
        raise ValueError("branch_ids must be unique.")

    return tuple(zip(ids, statuses))


# Topology-only action precomputation is owned by the action runtime.
DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES = 8 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 128


def _hash_array(digest: Any, values: np.ndarray, *, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def structural_topology_fingerprint(
    state: Any,
    *,
    require_connected_after_switch: bool,
    closeable_branch_ids: tuple[int, ...],
) -> bytes:
    """Return the exact identity of topology-only action validity."""

    branch_ids = np.asarray(state.branch_ids, dtype=np.int64)
    branch_status = np.asarray(state.branch_status, dtype=np.float64)
    edge_index = np.asarray(state.edge_index, dtype=np.int64)

    if branch_ids.ndim != 1 or branch_status.ndim != 1:
        raise ValueError("branch_ids and branch_status must be one-dimensional.")
    if len(branch_ids) != len(branch_status):
        raise ValueError("branch_ids and branch_status length mismatch.")
    if edge_index.shape != (2, len(branch_ids)):
        raise ValueError("edge_index must have shape [2, num_branches].")

    active = np.asarray(branch_status > 0.0, dtype=np.uint8)
    num_buses = int(np.asarray(state.bus_features).shape[0])

    digest = hashlib.sha256()
    digest.update(b"structural-topology-v1\0")
    digest.update(np.asarray([num_buses], dtype=np.int64).tobytes())
    _hash_array(digest, branch_ids, dtype=np.int64)
    _hash_array(digest, edge_index, dtype=np.int64)
    _hash_array(digest, active, dtype=np.uint8)
    digest.update(b"\x01" if require_connected_after_switch else b"\x00")
    _hash_array(
        digest,
        np.asarray(closeable_branch_ids, dtype=np.int64),
        dtype=np.int64,
    )
    return digest.digest()


@dataclass(frozen=True, slots=True)
class _PackedMask:
    values: np.ndarray
    length: int

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "_PackedMask":
        values = np.asarray(mask, dtype=bool)
        if values.ndim != 1:
            raise ValueError("Structural action mask must be one-dimensional.")

        packed = np.packbits(values).copy()
        packed.setflags(write=False)
        return cls(values=packed, length=int(values.size))

    @property
    def owned_bytes(self) -> int:
        return int(self.values.nbytes)

    def unpack(self) -> np.ndarray:
        return np.unpackbits(self.values, count=self.length).astype(bool, copy=False)


class StructuralTopologyCache:
    """Byte-bounded cache for topology-only action masks."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES,
    ) -> None:
        self._cache: ByteLRUCache[bytes, _PackedMask] = ByteLRUCache(max_bytes)
        self.hits = 0
        self.misses = 0

    def lookup(
        self,
        state: Any,
        *,
        require_connected_after_switch: bool,
        closeable_branch_ids: tuple[int, ...],
    ) -> tuple[bytes, np.ndarray | None]:
        key = structural_topology_fingerprint(
            state,
            require_connected_after_switch=require_connected_after_switch,
            closeable_branch_ids=closeable_branch_ids,
        )
        packed = self._cache.get(key)
        if packed is None:
            self.misses += 1
            return key, None

        self.hits += 1
        return key, packed.unpack()

    def store(self, key: bytes, mask: np.ndarray) -> bool:
        packed = _PackedMask.from_mask(mask)
        return self._cache.put(
            key,
            packed,
            size_bytes=packed.owned_bytes + len(key) + _ENTRY_OVERHEAD_BYTES,
        )

    def clear(self, *, reset_counters: bool = True) -> None:
        self._cache.clear(reset_evictions=reset_counters)
        if reset_counters:
            self.hits = 0
            self.misses = 0

    def info(self) -> dict[str, int]:
        info = self._cache.info()
        return {
            "size": int(info.entries),
            "bytes": int(info.bytes),
            "max_bytes": int(info.max_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(info.evictions),
        }


__all__ = [
    "ActionSlot",
    "ActionSpaceConfig",
    "ActionType",
    "GridFMAction",
    "GridFMActionSpace",
    "ActionKind",
]


class GridFMActionSpace:
    """Structural and operational topology-action validity.

    The only cached value is the topology-only structural mask. Dynamic loading
    filters and action objects are derived from the current state on every call,
    so cache reuse cannot hide a change in operational state.
    """

    def __init__(
        self,
        require_connected_after_switch: bool = True,
        min_loading_for_switch_percent: float = 0.0,
        closeable_branch_ids: tuple[int, ...] = (),
        enable_cache: bool = True,
        structural_cache_max_bytes: int = DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES,
    ) -> None:
        self._config = ActionSpaceConfig(
            require_connected_after_switch=require_connected_after_switch,
            min_loading_for_switch_percent=min_loading_for_switch_percent,
            enable_cache=enable_cache,
            closeable_branch_ids=closeable_branch_ids,
        )
        self._loading_column_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
        self._structural_cache = StructuralTopologyCache(
            max_bytes=int(structural_cache_max_bytes)
        )

    @property
    def config(self) -> ActionSpaceConfig:
        return self._config

    @property
    def require_connected_after_switch(self) -> bool:
        return self._config.require_connected_after_switch

    @property
    def min_loading_for_switch_percent(self) -> float:
        return self._config.min_loading_for_switch_percent

    @property
    def closeable_branch_ids(self) -> tuple[int, ...]:
        return self._config.closeable_branch_ids

    @property
    def enable_cache(self) -> bool:
        return self._config.enable_cache

    @property
    def cache_hits(self) -> int:
        return int(self._structural_cache.hits)

    @property
    def cache_misses(self) -> int:
        return int(self._structural_cache.misses)

    def clear_cache(self) -> None:
        self._structural_cache.clear(reset_counters=True)

    def cache_info(self) -> dict[str, object]:
        info: dict[str, object] = dict(self._structural_cache.info())
        info["enabled"] = bool(self.enable_cache)
        total = int(info["hits"]) + int(info["misses"])
        info["hit_rate"] = float(info["hits"]) / float(total) if total > 0 else 0.0
        return info

    def _switch_connectivity_mask(self, state: GridFMState) -> np.ndarray:
        """Return branches that can be opened without creating an island."""

        num_branches = len(state.branch_ids)
        connectivity_ok = np.zeros(num_branches, dtype=bool)
        num_buses = int(state.bus_features.shape[0])

        graph = nx.Graph()
        graph.add_nodes_from(range(num_buses))

        pair_counter: Counter[tuple[int, int]] = Counter()
        pair_by_branch_pos: dict[int, tuple[int, int]] = {}
        active_positions: list[int] = []

        for branch_pos in range(num_branches):
            if state.branch_status[branch_pos] <= 0:
                continue

            from_bus = int(state.edge_index[0, branch_pos])
            to_bus = int(state.edge_index[1, branch_pos])
            active_positions.append(branch_pos)

            if from_bus == to_bus:
                connectivity_ok[branch_pos] = True
                continue

            pair = (min(from_bus, to_bus), max(from_bus, to_bus))
            pair_counter[pair] += 1
            pair_by_branch_pos[branch_pos] = pair
            graph.add_edge(from_bus, to_bus)

        if not nx.is_connected(graph):
            return connectivity_ok

        bridges = {
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in nx.bridges(graph)
        }

        for branch_pos in active_positions:
            pair = pair_by_branch_pos.get(branch_pos)
            if pair is None:
                connectivity_ok[branch_pos] = True
            elif pair_counter[pair] > 1:
                connectivity_ok[branch_pos] = True
            else:
                connectivity_ok[branch_pos] = pair not in bridges

        return connectivity_ok

    def _require_known_closeable_branches(self, state: GridFMState) -> None:
        if not self.closeable_branch_ids:
            return

        known = {int(branch_id) for branch_id in state.branch_ids}
        unknown = sorted(set(self.closeable_branch_ids) - known)
        if unknown:
            raise ValueError(
                "closeable_branch_ids contains branches that are absent from "
                f"the current grid: {unknown}."
            )

    def build_action_slots(self, state: GridFMState) -> tuple[ActionSlot, ...]:
        return build_branch_action_slots(state.branch_ids)

    def action_layout_fingerprint(self, state: GridFMState) -> str:
        return action_layout_fingerprint(self.build_action_slots(state))

    def build_all_actions(self, state: GridFMState) -> list[GridFMAction]:
        actions: list[GridFMAction] = []

        for slot in self.build_action_slots(state):
            if slot.kind == "stop":
                actions.append(
                    GridFMAction(
                        action_id=slot.action_id,
                        action_type="do_nothing",
                    )
                )
                continue

            if slot.kind != "branch_status":
                raise RuntimeError(f"Unsupported action slot kind: {slot.kind!r}.")

            assert slot.target_id is not None
            assert slot.target_pos is not None
            active = self._is_branch_active(state, slot.target_pos)
            actions.append(
                GridFMAction(
                    action_id=slot.action_id,
                    action_type=("switch_off_branch" if active else "switch_on_branch"),
                    branch_id=slot.target_id,
                    branch_pos=slot.target_pos,
                    target_status=0 if active else 1,
                )
            )

        return actions

    def structural_action_mask(self, state: GridFMState) -> np.ndarray:
        """Return topology-only action validity."""

        self._require_known_closeable_branches(state)

        cache_key: bytes | None = None
        if self.enable_cache:
            cache_key, cached = self._structural_cache.lookup(
                state,
                require_connected_after_switch=self.require_connected_after_switch,
                closeable_branch_ids=self.closeable_branch_ids,
            )
            if cached is not None:
                return cached

        actions = self.build_all_actions(state)
        mask = np.zeros(len(actions), dtype=bool)
        mask[0] = True

        connectivity_ok = (
            self._switch_connectivity_mask(state)
            if self.require_connected_after_switch
            else np.ones(len(state.branch_ids), dtype=bool)
        )
        closeable = set(self.closeable_branch_ids)

        for action in actions[1:]:
            assert action.branch_id is not None
            assert action.branch_pos is not None
            assert action.target_status is not None

            if action.target_status == 0:
                if not self._is_branch_active(state, action.branch_pos):
                    continue
                if self.require_connected_after_switch and not bool(
                    connectivity_ok[action.branch_pos]
                ):
                    continue
                mask[action.action_id] = True
                continue

            if self._is_branch_active(state, action.branch_pos):
                continue
            if action.branch_id in closeable:
                mask[action.action_id] = True

        if self.enable_cache:
            assert cache_key is not None
            self._structural_cache.store(cache_key, mask)

        return mask

    def operational_action_mask(self, state: GridFMState) -> np.ndarray:
        """Apply current loading filters on top of the structural mask."""

        mask = self.structural_action_mask(state)
        if self.min_loading_for_switch_percent <= 0.0:
            return mask

        for action in self.build_all_actions(state)[1:]:
            if not bool(mask[action.action_id]):
                continue
            assert action.branch_pos is not None
            if action.target_status == 0 and not self._passes_loading_filter(
                state, action.branch_pos
            ):
                mask[action.action_id] = False

        return mask

    def valid_actions(self, state: GridFMState) -> list[GridFMAction]:
        actions = self.build_all_actions(state)
        mask = self.operational_action_mask(state)
        return [action for action in actions if bool(mask[action.action_id])]

    def invalid_actions(self, state: GridFMState) -> list[GridFMAction]:
        actions = self.build_all_actions(state)
        mask = self.operational_action_mask(state)
        return [action for action in actions if not bool(mask[action.action_id])]

    def loading_priority(
        self,
        state: GridFMState,
        action: GridFMAction,
    ) -> float | None:
        if action.kind != "set_branch_status":
            return None
        if action.target_status != 0 or action.branch_pos is None:
            return None
        return float(state.branch_features[action.branch_pos, self._loading_column_idx])

    @staticmethod
    def _is_branch_active(state: GridFMState, branch_pos: int) -> bool:
        return bool(state.branch_status[branch_pos] > 0)

    def _passes_loading_filter(self, state: GridFMState, branch_pos: int) -> bool:
        if self.min_loading_for_switch_percent <= 0.0:
            return True
        loading = float(state.branch_features[branch_pos, self._loading_column_idx])
        return loading >= self.min_loading_for_switch_percent


def make_do_nothing_action() -> GridFMAction:
    """Construct the canonical STOP/no-op topology action."""
    return GridFMAction(
        action_id=0,
        action_type="do_nothing",
        branch_id=None,
        branch_pos=None,
    )
