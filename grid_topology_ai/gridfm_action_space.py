from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

from grid_topology_ai.gridfm_adapter import (
    BRANCH_FEATURE_COLUMNS,
    GridFMState,
)


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
    ):
        """
        Parameters
        ----------
        require_connected_after_switch:
            If True, an action is valid only if disconnecting the selected
            branch does not split the grid into islands.

        min_loading_for_switch_percent:
            Optional filter. If > 0, branches with loading below this threshold
            are not considered switch-off candidates.

            For now we keep it at 0.0 because sometimes a low-loaded branch
            may still be useful in a multi-step topology reconfiguration.
        """

        self.require_connected_after_switch = require_connected_after_switch
        self.min_loading_for_switch_percent = min_loading_for_switch_percent

        self.loading_column_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

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
        """
        Compute boolean mask of valid actions.

        Returns
        -------
        np.ndarray
            Boolean array of shape [1 + num_branches].

        mask[i] = True means action i is allowed.
        mask[i] = False means action i is forbidden.

        Why action mask is important:
        We do not want the neural network or MCTS to waste time on obviously
        illegal or dangerous actions.
        """

        num_branches = len(state.branch_ids)

        mask = np.zeros(1 + num_branches, dtype=bool)

        # do_nothing is always legal.
        mask[0] = True

        graph = self._build_active_multigraph(state)

        # If the initial graph is already disconnected, we do not allow additional
        # switch-off actions in the first MVP.
        initial_graph_connected = nx.is_connected(graph)

        for branch_pos in range(num_branches):
            action_id = 1 + branch_pos

            if not self._is_branch_active(state, branch_pos):
                mask[action_id] = False
                continue

            if not self._passes_loading_filter(state, branch_pos):
                mask[action_id] = False
                continue

            if self.require_connected_after_switch:
                if not initial_graph_connected:
                    mask[action_id] = False
                    continue

                if not self._keeps_grid_connected_after_removal(
                    graph=graph,
                    state=state,
                    branch_pos=branch_pos,
                ):
                    mask[action_id] = False
                    continue

            mask[action_id] = True

        return mask

    def valid_actions(self, state: GridFMState) -> list[GridFMAction]:
        """
        Return only valid actions.
        """

        all_actions = self.build_all_actions(state)
        mask = self.valid_action_mask(state)

        return [
            action
            for action in all_actions
            if mask[action.action_id]
        ]

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