from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np

from grid_topology_ai.data_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMState,
)
from grid_topology_ai.topology_actions import (
    ActionSlot,
    ActionSpaceConfig,
    ActionType,
    GridFMAction,
    build_branch_action_slots,
    ActionKind,
    branch_status_signature,
)

__all__ = [
    "ActionSlot",
    "ActionSpaceConfig",
    "ActionType",
    "GridFMAction",
    "GridFMActionSpace",
]



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
            closeable_branch_ids: tuple[int, ...] = (),
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

        self._config = ActionSpaceConfig(
            require_connected_after_switch=(
                require_connected_after_switch
            ),
            min_loading_for_switch_percent=(
                min_loading_for_switch_percent
            ),
            enable_cache=enable_cache,
            closeable_branch_ids=(
                closeable_branch_ids
            ),
        )

        self._loading_column_idx = (
            BRANCH_FEATURE_COLUMNS.index(
                "loading_percent"
            )
        )

        self._structural_action_mask_cache: dict[tuple, np.ndarray,] = {}
        self._operational_action_mask_cache: dict[tuple, np.ndarray,] = {}
        self._valid_actions_cache: dict[tuple,list[GridFMAction],] = {}
        self._connectivity_mask_cache: dict[tuple,np.ndarray,] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def config(self) -> ActionSpaceConfig:
        return self._config

    @property
    def require_connected_after_switch(
            self,
    ) -> bool:
        return (
            self._config.require_connected_after_switch
        )

    @property
    def min_loading_for_switch_percent(
            self,
    ) -> float:
        return (
            self._config.min_loading_for_switch_percent
        )

    @property
    def closeable_branch_ids(
            self,
    ) -> tuple[int, ...]:
        return self._config.closeable_branch_ids

    @property
    def enable_cache(self) -> bool:
        return self._config.enable_cache

    @property
    def min_loading_for_switch_percent(
        self,
    ) -> float:
        return (
            self._config.min_loading_for_switch_percent
        )

    @property
    def enable_cache(self) -> bool:
        return self._config.enable_cache

    def clear_cache(self) -> None:
        self._structural_action_mask_cache.clear()
        self._operational_action_mask_cache.clear()
        self._valid_actions_cache.clear()
        self._connectivity_mask_cache.clear()

        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total
            if total > 0
            else 0.0
        )

        return {
            "enabled": self.enable_cache,
            # Compatibility field: valid_action_mask() remains
            # the operational mask.
            "mask_cache_size": len(
                self._operational_action_mask_cache
            ),
            "structural_mask_cache_size": len(
                self._structural_action_mask_cache
            ),
            "operational_mask_cache_size": len(
                self._operational_action_mask_cache
            ),
            "valid_actions_cache_size": len(
                self._valid_actions_cache
            ),
            "connectivity_cache_size": len(
                self._connectivity_mask_cache
            ),
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

        cache_key = ("connectivity", self._structural_cache_key(state))

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

    def _structural_cache_key(
            self,
            state: GridFMState,
    ) -> tuple:
        """
        Cache key for topology-dependent action validity.
        """

        return (
            "structural",
            int(state.scenario_id),
            branch_status_signature(
                state.branch_ids,
                state.branch_status,
            ),
            bool(
                self.require_connected_after_switch
            ),
            self.closeable_branch_ids,
        )

    def _loading_signature(
        self,
        state: GridFMState,
    ) -> tuple[float, ...] | None:
        """
        Return loading values used by the operational filter.

        With a disabled loading threshold, operational validity
        is independent of branch loading.
        """

        if (
            self.min_loading_for_switch_percent
            <= 0.0
        ):
            return None

        loading_values = state.branch_features[
            :,
            self._loading_column_idx,
        ]

        return tuple(
            float(value)
            for value in loading_values
        )

    def _make_cache_key(
        self,
        state: GridFMState,
    ) -> tuple:
        """
        Cache key for operational action validity.

        This key includes both immutable action-space
        configuration and the loading values read by the
        operational filter.
        """

        return (
            "operational",
            self._config,
            self._structural_cache_key(state),
            self._loading_signature(state),
        )

    def _require_known_closeable_branches(
        self,
        state: GridFMState,
    ) -> None:
        if not self.closeable_branch_ids:
            return

        known_branch_ids = {
            int(branch_id)
            for branch_id in state.branch_ids
        }
        unknown_branch_ids = sorted(
            set(self.closeable_branch_ids)
            - known_branch_ids
        )

        if unknown_branch_ids:
            raise ValueError(
                "closeable_branch_ids contains branches "
                "that are absent from the current grid: "
                f"{unknown_branch_ids}."
            )

    def build_action_slots(
        self,
        state: GridFMState,
    ) -> tuple[ActionSlot, ...]:
        """
        Return the stable policy layout for the grid.

        The slot layout depends on branch identity and order,
        not on the current branch status.
        """

        return build_branch_action_slots(
            state.branch_ids
        )

    def build_all_actions(
            self,
            state: GridFMState,
    ) -> list[GridFMAction]:
        """
        Build one executable action per stable policy slot.

        An active branch produces an opening command.
        An inactive branch produces a closing command, but
        legality is still decided by the action mask.
        """

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
                raise RuntimeError(
                    "Unsupported action slot kind: "
                    f"{slot.kind!r}."
                )

            assert slot.target_id is not None
            assert slot.target_pos is not None

            is_active = self._is_branch_active(
                state,
                slot.target_pos,
            )

            actions.append(
                GridFMAction(
                    action_id=slot.action_id,
                    action_type=(
                        "switch_off_branch"
                        if is_active
                        else "switch_on_branch"
                    ),
                    branch_id=slot.target_id,
                    branch_pos=slot.target_pos,
                    target_status=(
                        0 if is_active else 1
                    ),
                )
            )

        return actions

    def structural_action_mask(
            self,
            state: GridFMState,
    ) -> np.ndarray:
        """
        Return structurally valid topology actions.

        Opening an active branch may be restricted by
        connectivity. Closing is allowed only for explicitly
        configured normally-open branches.
        """

        self._require_known_closeable_branches(
            state
        )

        cache_key = self._structural_cache_key(
            state
        )

        if (
                self.enable_cache
                and cache_key
                in self._structural_action_mask_cache
        ):
            self.cache_hits += 1
            return (
                self._structural_action_mask_cache[
                    cache_key
                ].copy()
            )

        if self.enable_cache:
            self.cache_misses += 1

        actions = self.build_all_actions(state)
        mask = np.zeros(
            len(actions),
            dtype=bool,
        )
        mask[0] = True

        if self.require_connected_after_switch:
            connectivity_ok = (
                self._switch_connectivity_mask(
                    state
                )
            )
        else:
            connectivity_ok = np.ones(
                len(state.branch_ids),
                dtype=bool,
            )

        closeable = set(
            self.closeable_branch_ids
        )

        for action in actions[1:]:
            assert action.branch_id is not None
            assert action.branch_pos is not None
            assert action.target_status is not None

            if action.target_status == 0:
                if not self._is_branch_active(
                        state,
                        action.branch_pos,
                ):
                    continue

                if (
                        self.require_connected_after_switch
                        and not bool(
                    connectivity_ok[
                        action.branch_pos
                    ]
                )
                ):
                    continue

                mask[action.action_id] = True
                continue

            if self._is_branch_active(
                    state,
                    action.branch_pos,
            ):
                continue

            if action.branch_id not in closeable:
                continue

            mask[action.action_id] = True

        if self.enable_cache:
            self._structural_action_mask_cache[
                cache_key
            ] = mask.copy()

        return mask

    def operational_action_mask(
        self,
        state: GridFMState,
    ) -> np.ndarray:
        """
        Return actions allowed by structural and operational
        action-space constraints.

        The loading threshold is an operational candidate
        filter and applies only to branch-switch actions.
        """

        cache_key = self._make_cache_key(
            state
        )

        if (
            self.enable_cache
            and cache_key
            in self._operational_action_mask_cache
        ):
            self.cache_hits += 1
            return (
                self._operational_action_mask_cache[
                    cache_key
                ].copy()
            )

        if self.enable_cache:
            self.cache_misses += 1

        mask = self.structural_action_mask(
            state
        )

        actions = self.build_all_actions(
            state
        )

        for action in actions[1:]:
            if not bool(
                    mask[action.action_id]
            ):
                continue

            assert action.branch_pos is not None
            assert action.target_status is not None

            # Loading is meaningful only for opening an
            # active branch. An inactive tie-line normally
            # has zero flow before it is closed.
            if (
                    action.target_status == 0
                    and not self._passes_loading_filter(
                state,
                action.branch_pos,
            )
            ):
                mask[action.action_id] = False

        if self.enable_cache:
            self._operational_action_mask_cache[
                cache_key
            ] = mask.copy()

        return mask

    def valid_action_mask(
        self,
        state: GridFMState,
    ) -> np.ndarray:
        """
        Compatibility alias for the operational action mask.
        """

        return self.operational_action_mask(
            state
        )

    def valid_actions(
            self,
            state: GridFMState,
    ) -> list[GridFMAction]:
        cache_key = self._make_cache_key(
            state
        )

        if (
                self.enable_cache
                and cache_key
                in self._valid_actions_cache
        ):
            self.cache_hits += 1
            return list(
                self._valid_actions_cache[
                    cache_key
                ]
            )

        if self.enable_cache:
            self.cache_misses += 1

        all_actions = self.build_all_actions(
            state
        )
        mask = self.operational_action_mask(
            state
        )

        valid = [
            action
            for action in all_actions
            if bool(mask[action.action_id])
        ]

        if self.enable_cache:
            self._valid_actions_cache[
                cache_key
            ] = list(valid)

        return valid

    def invalid_actions(self, state: GridFMState) -> list[GridFMAction]:
        """
        Return invalid actions.

        This is mostly useful for debugging.
        """

        all_actions = self.build_all_actions(state)
        mask = self.operational_action_mask(state)

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

        loading = float(state.branch_features[branch_pos, self._loading_column_idx,])

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