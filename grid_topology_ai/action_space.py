from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np

from grid_topology_ai.cache import (
    DEFAULT_STRUCTURAL_TOPOLOGY_CACHE_BYTES,
    StructuralTopologyCache,
)
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.topology_actions import (
    ActionKind,
    ActionSlot,
    ActionSpaceConfig,
    ActionType,
    GridFMAction,
    action_layout_fingerprint,
    build_branch_action_slots,
)

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
        info["hit_rate"] = (
            float(info["hits"]) / float(total)
            if total > 0
            else 0.0
        )
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
                    action_type=(
                        "switch_off_branch" if active else "switch_on_branch"
                    ),
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
                if (
                    self.require_connected_after_switch
                    and not bool(connectivity_ok[action.branch_pos])
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
            if (
                action.target_status == 0
                and not self._passes_loading_filter(state, action.branch_pos)
            ):
                mask[action.action_id] = False

        return mask

    def valid_action_mask(self, state: GridFMState) -> np.ndarray:
        return self.operational_action_mask(state)

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
        return float(
            state.branch_features[action.branch_pos, self._loading_column_idx]
        )

    @staticmethod
    def _is_branch_active(state: GridFMState, branch_pos: int) -> bool:
        return bool(state.branch_status[branch_pos] > 0)

    def _passes_loading_filter(self, state: GridFMState, branch_pos: int) -> bool:
        if self.min_loading_for_switch_percent <= 0.0:
            return True
        loading = float(
            state.branch_features[branch_pos, self._loading_column_idx]
        )
        return loading >= self.min_loading_for_switch_percent
