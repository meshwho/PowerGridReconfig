from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMState,
)
from collections import Counter

ActionType = Literal["do_nothing", "switch_off_branch"]


@dataclass(frozen=True)
class GridFMAction:
    """
    One topology switching action.

    action_id:
        Integer action ID used by RL / neural network.

    action_type:
        "do_nothing" or "switch_off_branch".

    branch_id:
        Original GridFM branch idx.
        For do_nothing, branch_id is None.

    branch_pos:
        Position of this branch in state.branch_features / state.edge_index.
        For do_nothing, branch_pos is None.
    """

    action_id: int
    action_type: ActionType
    branch_id: int | None = None
    branch_pos: int | None = None


class GridFMActionSpace:
    """
    Action space for GridFM topology switching.

    Current MVP:
        - do nothing
        - switch off one active branch

    Important:
    This class does not run power flow.
    It only checks structural/topological validity of actions.

    Later, the simulator will apply an action and run AC power flow.
    """

    def __init__(
            self,
            require_connected_after_switch: bool = True,
            min_loading_for_switch_percent: float = 0.0,
            enable_cache: bool = True,
    ):
        """
        Action space for topology switching.

        require_connected_after_switch:
            If True, an action is valid only if the grid remains connected
            after switching off the selected branch.

        min_loading_for_switch_percent:
            Optional filter for candidate switching actions.
            If > 0, only branches with loading above this threshold are considered
            switchable.

        enable_cache:
            Cache valid actions and valid masks for repeated MCTS states.
        """

        self.require_connected_after_switch = bool(require_connected_after_switch)
        self.min_loading_for_switch_percent = float(min_loading_for_switch_percent)
        self.enable_cache = bool(enable_cache)

        self._valid_action_mask_cache: dict[tuple, np.ndarray] = {}
        self._valid_actions_cache: dict[tuple, list[GridFMAction]] = {}
        self._connectivity_mask_cache: dict[tuple, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self) -> None:
        self._valid_action_mask_cache.clear()
        self._valid_actions_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self._connectivity_mask_cache.clear()

    def cache_info(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "enabled": self.enable_cache,
            "mask_cache_size": len(self._valid_action_mask_cache),
            "valid_actions_cache_size": len(self._valid_actions_cache),
            "connectivity_cache_size": len(self._connectivity_mask_cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def _switch_connectivity_mask(self, state: GridFMState) -> np.ndarray:
        """
        Compute which branch switch-off actions keep the grid connected.

        This replaces the expensive old approach:

            for each branch:
                copy MultiGraph
                remove edge
                run nx.is_connected()

        New approach:

            build active simple graph once
            compute graph bridges once
            account for parallel branches using edge multiplicity

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_branches].
            result[branch_pos] = True means removing this branch does not create islands.
        """

        cache_key = ("connectivity", self._make_cache_key(state))

        if self.enable_cache and cache_key in self._connectivity_mask_cache:
            return self._connectivity_mask_cache[cache_key].copy()

        num_branches = len(state.branch_ids)

        # Default is False. We only mark structurally safe switch-offs as True.
        connectivity_ok = np.zeros(num_branches, dtype=bool)

        num_buses = state.bus_features.shape[0]

        graph = nx.Graph()
        graph.add_nodes_from(range(num_buses))

        pair_counter: Counter[tuple[int, int]] = Counter()
        pair_by_branch_pos: dict[int, tuple[int, int]] = {}

        active_branch_positions: list[int] = []

        for branch_pos, branch_id in enumerate(state.branch_ids):
            if state.branch_status[branch_pos] <= 0:
                continue

            from_bus = int(state.edge_index[0, branch_pos])
            to_bus = int(state.edge_index[1, branch_pos])

            active_branch_positions.append(int(branch_pos))

            # Self-loop does not affect graph connectivity.
            if from_bus == to_bus:
                connectivity_ok[branch_pos] = True
                continue

            pair = (
                min(from_bus, to_bus),
                max(from_bus, to_bus),
            )

            pair_counter[pair] += 1
            pair_by_branch_pos[int(branch_pos)] = pair

            graph.add_edge(from_bus, to_bus)

        # If the current grid is already disconnected, do not allow more switching.
        if not nx.is_connected(graph):
            if self.enable_cache:
                self._connectivity_mask_cache[cache_key] = connectivity_ok.copy()
            return connectivity_ok

        bridge_pairs = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in nx.bridges(graph)
        }

        for branch_pos in active_branch_positions:
            pair = pair_by_branch_pos.get(int(branch_pos))

            if pair is None:
                # self-loop case
                connectivity_ok[branch_pos] = True
                continue

            # If there are parallel active branches between the same buses,
            # removing one physical branch cannot disconnect the grid.
            if pair_counter[pair] > 1:
                connectivity_ok[branch_pos] = True
                continue

            # If the simple edge is not a bridge, removing it is safe.
            connectivity_ok[branch_pos] = pair not in bridge_pairs

        if self.enable_cache:
            self._connectivity_mask_cache[cache_key] = connectivity_ok.copy()

        return connectivity_ok

    def _make_cache_key(self, state) -> tuple:
        """
        Valid actions depend only on topology for the current topology-switching stage.

        Later, if we add redispatch constraints or dynamic limits, this key may need
        to include more information.
        """

        return (
            int(state.scenario_id),
            tuple(int(x) for x in sorted(state.outaged_branch_ids)),
            bool(self.require_connected_after_switch),
        )

    def build_all_actions(self, state: GridFMState) -> list[GridFMAction]:
        """
        Build the full fixed action list for one state.

        Action indexing convention:
            action_id = 0              -> do nothing
            action_id = 1 + branch_pos -> switch off branch at branch_pos

        This convention is very useful later for neural network policy output:
            policy_logits shape = [1 + num_branches]
        """

        actions: list[GridFMAction] = [
            GridFMAction(
                action_id=0,
                action_type="do_nothing",
                branch_id=None,
                branch_pos=None,
            )
        ]

        for branch_pos, branch_id in enumerate(state.branch_ids):
            actions.append(
                GridFMAction(
                    action_id=1 + branch_pos,
                    action_type="switch_off_branch",
                    branch_id=int(branch_id),
                    branch_pos=int(branch_pos),
                )
            )

        return actions

    def valid_action_mask(self, state: GridFMState) -> np.ndarray:
        cache_key = self._make_cache_key(state)

        if self.enable_cache and cache_key in self._valid_action_mask_cache:
            self.cache_hits += 1
            return self._valid_action_mask_cache[cache_key].copy()

        if self.enable_cache:
            self.cache_misses += 1

        num_branches = len(state.branch_ids)

        mask = np.zeros(1 + num_branches, dtype=bool)

        # do_nothing is structurally legal.
        mask[0] = True

        if self.require_connected_after_switch:
            connectivity_ok = self._switch_connectivity_mask(state)
        else:
            connectivity_ok = np.ones(num_branches, dtype=bool)

        for branch_pos in range(num_branches):
            action_id = 1 + branch_pos

            if not self._is_branch_active(state, branch_pos):
                mask[action_id] = False
                continue

            if not self._passes_loading_filter(state, branch_pos):
                mask[action_id] = False
                continue

            if self.require_connected_after_switch:
                if not bool(connectivity_ok[branch_pos]):
                    mask[action_id] = False
                    continue

            mask[action_id] = True

        if self.enable_cache:
            self._valid_action_mask_cache[cache_key] = mask.copy()

        return mask

    def valid_actions(self, state):
        cache_key = self._make_cache_key(state)

        if self.enable_cache and cache_key in self._valid_actions_cache:
            self.cache_hits += 1
            return list(self._valid_actions_cache[cache_key])

        if self.enable_cache:
            self.cache_misses += 1

        all_actions = self.build_all_actions(state)
        mask = self.valid_action_mask(state)

        valid = [
            action
            for action in all_actions
            if bool(mask[action.action_id])
        ]

        if self.enable_cache:
            self._valid_actions_cache[cache_key] = list(valid)

        return valid

    def invalid_actions(self, state: GridFMState) -> list[GridFMAction]:
        """
        Return invalid actions.

        This is mostly useful for debugging.
        """

        all_actions = self.build_all_actions(state)
        mask = self.valid_action_mask(state)

        return [
            action
            for action in all_actions
            if not mask[action.action_id]
        ]

    def _is_branch_active(self, state: GridFMState, branch_pos: int) -> bool:
        """
        Check if branch is currently in service.
        """

        return bool(state.branch_status[branch_pos] > 0)

    def _passes_loading_filter(self, state: GridFMState, branch_pos: int) -> bool:
        """
        Optional filter based on branch loading.

        For now this usually returns True because the default threshold is 0.
        """

        if self.min_loading_for_switch_percent <= 0:
            return True

        loading = float(state.branch_features[branch_pos, self.loading_column_idx])

        return loading >= self.min_loading_for_switch_percent

    @staticmethod
    def _build_active_multigraph(state: GridFMState) -> nx.MultiGraph:
        """
        Build NetworkX MultiGraph from active branches.

        Why MultiGraph and not Graph?
        Power grids can have parallel lines between the same buses.
        If we used a simple Graph, removing one parallel line could accidentally
        look like removing all parallel lines.

        Each edge key is the original GridFM branch ID.
        """

        num_buses = state.bus_features.shape[0]

        graph = nx.MultiGraph()
        graph.add_nodes_from(range(num_buses))

        for branch_pos, branch_id in enumerate(state.branch_ids):
            if state.branch_status[branch_pos] <= 0:
                continue

            from_bus = int(state.edge_index[0, branch_pos])
            to_bus = int(state.edge_index[1, branch_pos])

            graph.add_edge(
                from_bus,
                to_bus,
                key=int(branch_id),
                branch_pos=int(branch_pos),
            )

        return graph

    @staticmethod
    def _keeps_grid_connected_after_removal(
        graph: nx.MultiGraph,
        state: GridFMState,
        branch_pos: int,
    ) -> bool:
        """
        Check whether removing one active branch keeps the grid connected.

        This is a purely topological safety filter.

        It does not guarantee that power flow will be good after switching.
        It only guarantees that the action does not immediately create islands.
        """

        branch_id = int(state.branch_ids[branch_pos])
        from_bus = int(state.edge_index[0, branch_pos])
        to_bus = int(state.edge_index[1, branch_pos])

        test_graph = graph.copy()

        if not test_graph.has_edge(from_bus, to_bus, key=branch_id):
            return False

        test_graph.remove_edge(from_bus, to_bus, key=branch_id)

        return nx.is_connected(test_graph)