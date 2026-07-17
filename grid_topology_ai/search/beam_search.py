from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from grid_topology_ai.search.continuation_gate import topology_penalty
from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.termination import TerminationReason


@dataclass(frozen=True)
class BeamSearchConfig:
    """
    Configuration for depth-limited beam search.

    max_depth:
        Maximum number of topology switching steps.

    beam_width:
        Number of best partial sequences kept after each depth level.

    top_k_actions:
        Number of switch-off actions considered at each state.
        These actions are selected by current branch loading.

    gamma:
        Discount factor for future rewards.

    include_stop_action:
        Whether to include do_nothing/stop as a candidate action.
    """

    max_depth: int = 3
    beam_width: int = 5
    top_k_actions: int = 30
    gamma: float = 0.95
    include_stop_action: bool = True


@dataclass
class BeamSearchNode:
    """
    One partial sequence in beam search.

    The node contains its own environment clone, so different branches of
    the search tree do not modify each other.
    """

    env: TopologySwitchingEnv
    action_ids: list[int] = field(default_factory=list)
    branch_ids: list[int | None] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    discounted_return: float = 0.0
    undiscounted_return: float = 0.0
    depth: int = 0
    done: bool = False
    solved: bool = False
    termination_reason: TerminationReason | None = None
    last_step_result: TopologyStepResult | None = None

    def short_sequence(self) -> str:
        """
        Human-readable sequence of branch IDs.
        """

        parts = []

        for branch_id in self.branch_ids:
            if branch_id is None:
                parts.append("stop")
            else:
                parts.append(str(branch_id))

        return " -> ".join(parts) if parts else "(root)"


@dataclass(frozen=True)
class BeamSearchResult:
    """
    Final result of beam search.
    """

    scenario_id: int
    best_node: BeamSearchNode
    final_beam: list[BeamSearchNode]
    config: BeamSearchConfig


class BeamSearchPlanner:
    """
    Depth-limited beam search planner for topology switching.

    This is not MCTS yet.

    Purpose:
        find a good multi-step sequence before implementing AlphaZero/MCTS.

    Important:
        Unlike greedy search, beam search does not commit to only the best
        immediate action. It keeps several promising partial sequences.
    """

    def __init__(self, config: BeamSearchConfig):
        self.config = config
        self.loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    def search(
        self,
        env: TopologySwitchingEnv,
        scenario_id: int,
    ) -> BeamSearchResult:
        """
        Run beam search from one initial scenario.

        The input env is used as a factory and is not modified permanently.
        """

        root_env = env.clone()
        root_env.reset(scenario_id)

        root = BeamSearchNode(
            env=root_env,
            action_ids=[],
            branch_ids=[],
            rewards=[],
            discounted_return=0.0,
            undiscounted_return=0.0,
            depth=0,
            done=bool(root_env.done),
            solved=bool(root_env.solved),
            termination_reason=root_env.termination_reason,
            last_step_result=None,
        )

        beam: list[BeamSearchNode] = [root]
        completed: list[BeamSearchNode] = []

        for depth in range(self.config.max_depth):
            candidates: list[BeamSearchNode] = []

            for node in beam:
                if node.done:
                    completed.append(node)
                    candidates.append(node)
                    continue

                actions = self._candidate_actions(node.env)

                for action in actions:
                    child = self._expand_node(node, action)

                    if child is None:
                        continue

                    candidates.append(child)

                    if child.done:
                        completed.append(child)

            if not candidates:
                break

            candidates = self._sort_nodes(candidates)

            beam = candidates[: self.config.beam_width]

            # If a solved node is already the best, we can still continue
            # if there are other nonterminal nodes, but for MVP this is enough.
            if beam[0].solved:
                break

        all_final = completed + beam

        if not all_final:
            best_node = root
        else:
            best_node = self._sort_nodes(all_final)[0]

        return BeamSearchResult(
            scenario_id=int(scenario_id),
            best_node=best_node,
            final_beam=self._sort_nodes(beam),
            config=self.config,
        )

    def _candidate_actions(self, env: TopologySwitchingEnv) -> list[GridFMAction]:
        """
        Select candidate actions from the current environment state.

        We include:
            - optional stop/do_nothing;
            - top-K switch-off actions by current loading.
        """

        state = env.current_state

        if state is None:
            return []

        valid_actions = env.valid_actions()

        stop_actions = [
            action
            for action in valid_actions
            if action.action_type == "do_nothing"
        ]

        switch_actions = [
            action
            for action in valid_actions
            if action.action_type == "switch_off_branch"
        ]

        switch_actions = sorted(
            switch_actions,
            key=lambda action: float(
                state.branch_features[action.branch_pos, self.loading_idx]
            ),
            reverse=True,
        )

        if self.config.top_k_actions > 0:
            switch_actions = switch_actions[: self.config.top_k_actions]

        selected: list[GridFMAction] = []

        if self.config.include_stop_action:
            selected.extend(stop_actions)

        selected.extend(switch_actions)

        return selected

    def _expand_node(
        self,
        node: BeamSearchNode,
        action: GridFMAction,
    ) -> BeamSearchNode | None:
        """
        Expand one node by one action.

        Returns None if the action fails unexpectedly.
        """

        child_env = node.env.clone()

        try:
            # Use action_id instead of passing the action object directly.
            # This forces the child environment to validate the action in
            # its own current state.
            step_result = child_env.step(action.action_id)
        except Exception:
            return None

        reward = float(step_result.reward)

        discounted_reward = (self.config.gamma ** node.depth) * reward

        child = BeamSearchNode(
            env=child_env,
            action_ids=[*node.action_ids, int(action.action_id)],
            branch_ids=[*node.branch_ids, action.branch_id],
            rewards=[*node.rewards, reward],
            discounted_return=float(node.discounted_return + discounted_reward),
            undiscounted_return=float(node.undiscounted_return + reward),
            depth=node.depth + 1,
            done=bool(step_result.done),
            solved=bool(step_result.solved),
            termination_reason=step_result.info.get("termination_reason"),
            last_step_result=step_result,
        )

        return child

    @staticmethod
    def _sort_nodes(nodes: list[BeamSearchNode]) -> list[BeamSearchNode]:
        """
        Sort nodes by quality.

        Priority:
            1. solved nodes first;
            2. higher discounted return;
            3. shorter sequence if returns are similar.
        """

        return sorted(
            nodes,
            key=lambda node: (
                int(node.solved),
                node.discounted_return,
                -node.depth,
            ),
            reverse=True,
        )
