from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator
import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv


@dataclass(frozen=True)
class MCTSConfig:
    """
    Configuration for single-agent MCTS.

    This is AlphaZero-like, but for topology switching.

    num_simulations:
        Number of MCTS simulations from the root state.

    max_depth:
        Maximum number of topology switching steps inside one search.

    top_k_actions:
        Number of switch-off actions considered at each node.
        Actions are selected by current branch loading.

    gamma:
        Discount factor for future rewards.

    c_puct:
        Exploration constant used in PUCT.

    leaf_penalty_weight:
        Weight of heuristic penalty used to evaluate non-terminal leaf states.

    include_stop_action:
        If True, include do_nothing as a stop action.
    """

    num_simulations: int = 100
    max_depth: int = 4
    top_k_actions: int = 30
    gamma: float = 0.95
    c_puct: float = 1.5
    leaf_penalty_weight: float = 0.10
    include_stop_action: bool = True
    stop_prior: float = 1.0
    terminal_unsolved_penalty: float = 500.0
    # Internal MCTS value normalization.
    # Environment rewards are kept unchanged, but MCTS selection/backups
    # use scaled values to make Q comparable with the PUCT exploration term.
    value_scale: float = 1000.0

    # First Play Urgency.
    # Value assigned to unvisited actions during PUCT selection.
    # Without this, unvisited actions get Q=0 and may look too attractive
    # when all explored actions have negative Q.
    fpu_value: float = -0.25
    # Heuristic prior smoothing.
    # 1.0 = original sharp prior.
    # 0.5 = sqrt prior, better exploration before neural policy exists.
    prior_exponent: float = 0.5
    min_switch_prior_score: float = 1.0
    # Stop policy:
    #   "never"             - never include stop action;
    #   "solved_only"       - include stop only when no overloads remain;
    #   "no_hard_overloads" - include stop when hard overloads are removed;
    #   "always"            - always include stop.
    #
    # For topology switching + redispatch architecture, the best default is:
    #   no_hard_overloads
    #
    # Meaning:
    #   while hard overload exists -> continue topology switching;
    #   after hard overload is removed -> MCTS may hand off to redispatch.
    stop_policy: str = "no_hard_overloads"

@dataclass
class MCTSNode:
    """
    One node in the MCTS tree.

    Each node owns a cloned environment representing the state at this node.
    """

    env: TopologySwitchingEnv
    depth: int
    prior: float = 1.0

    action_id_from_parent: int | None = None
    branch_id_from_parent: int | None = None
    reward_from_parent: float = 0.0
    step_result_from_parent: TopologyStepResult | None = None

    visit_count: int = 0
    total_value: float = 0.0

    is_expanded: bool = False
    action_priors: dict[int, float] = field(default_factory=dict)
    actions_by_id: dict[int, GridFMAction] = field(default_factory=dict)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def done(self) -> bool:
        return bool(self.env.done)

    @property
    def solved(self) -> bool:
        return bool(self.env.solved)


@dataclass(frozen=True)
class MCTSResult:
    """
    Result of one MCTS search from one scenario.
    """

    scenario_id: int
    root: MCTSNode
    best_action_id: int | None
    best_branch_id: int | None
    visit_counts: dict[int, int]
    policy: dict[int, float]
    principal_action_ids: list[int]
    principal_branch_ids: list[int | None]
    principal_rewards: list[float]
    principal_return: float
    principal_final_metrics: dict[str, Any]
    config: MCTSConfig


class MCTSPlanner:
    """
    AlphaZero-style MCTS planner for topology switching.

    Current version:
        - no neural network yet;
        - heuristic priors based on branch loading;
        - heuristic leaf value based on remaining security penalty.

    Later:
        heuristic prior  -> policy network
        heuristic value  -> value network
    """

    def __init__(
            self,
            config: MCTSConfig,
            evaluator: "NeuralPolicyValueEvaluator | None" = None,
    ):
        self.config = config
        self.evaluator = evaluator

        self.loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
        self.status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

    def search(
            self,
            env: TopologySwitchingEnv,
            scenario_id: int,
    ) -> MCTSResult:
        """
        Run MCTS from one initial emergency scenario.

        This is a convenience wrapper:
            reset env to scenario_id
            then run search_from_env()
        """

        root_env = env.clone()
        root_env.reset(int(scenario_id))

        return self.search_from_env(root_env)

    def search_from_env(
        self,
        env: TopologySwitchingEnv,
    ) -> MCTSResult:
        """
        Run MCTS from the current environment state.

        This is the method required for AlphaZero-like self-play.

        In self-play / online control we do not execute a full planned sequence.
        Instead:
            1. run MCTS from the current state;
            2. execute only the best first action;
            3. observe the new state;
            4. run MCTS again.

        Therefore this method must NOT reset the environment.
        """

        if env.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        root_env = env.clone()

        scenario_id = (
            int(root_env.initial_scenario_id)
            if root_env.initial_scenario_id is not None
            else int(root_env.current_state.scenario_id)
        )

        root = MCTSNode(
            env=root_env,
            depth=0,
            prior=1.0,
        )

        self._expand_node(root)

        for _ in range(self.config.num_simulations):
            self._run_one_simulation(root)

        visit_counts = {
            action_id: child.visit_count
            for action_id, child in root.children.items()
        }

        total_visits = sum(visit_counts.values())

        if total_visits > 0:
            policy = {
                action_id: count / total_visits
                for action_id, count in visit_counts.items()
            }
        else:
            policy = {}

        best_action_id = None
        best_branch_id = None

        if visit_counts:
            best_action_id = max(
                visit_counts,
                key=lambda action_id: visit_counts[action_id],
            )

            best_child = root.children[best_action_id]
            best_branch_id = best_child.branch_id_from_parent

        (
            principal_action_ids,
            principal_branch_ids,
            principal_rewards,
            principal_return,
            final_metrics,
        ) = self._principal_variation(root)

        return MCTSResult(
            scenario_id=scenario_id,
            root=root,
            best_action_id=best_action_id,
            best_branch_id=best_branch_id,
            visit_counts=visit_counts,
            policy=policy,
            principal_action_ids=principal_action_ids,
            principal_branch_ids=principal_branch_ids,
            principal_rewards=principal_rewards,
            principal_return=principal_return,
            principal_final_metrics=final_metrics,
            config=self.config,
        )

    def _should_include_stop_action(self, state: GridFMState) -> bool:
        """
        Decide whether stop/do_nothing should be available at this state.

        In our architecture, stop means:
            - solved, if the grid is already secure;
            - handoff_to_redispatch, if topology switching should stop.

        Default logic:
            stop is available after hard overloads are removed.
        """

        if not self.config.include_stop_action:
            return False

        stop_policy = self.config.stop_policy

        num_overloaded = int(state.metrics["num_overloaded_branches"])
        num_hard_overloaded = int(
            state.metrics["num_hard_overloaded_branches"]
        )

        if stop_policy == "never":
            return False

        if stop_policy == "always":
            return True

        if stop_policy == "solved_only":
            return num_overloaded == 0 and num_hard_overloaded == 0

        if stop_policy == "no_hard_overloads":
            return num_hard_overloaded == 0

        raise ValueError(f"Unknown stop_policy: {stop_policy}")

    def _run_one_simulation(self, root: MCTSNode) -> None:
        """
        Run one MCTS simulation.

        Selection:
            follow PUCT.

        Expansion:
            create a new child when an unvisited action is selected.

        Evaluation:
            use heuristic leaf value.

        Backup:
            propagate discounted return backward.
        """

        node = root
        path: list[MCTSNode] = []

        while True:
            if node.done or node.depth >= self.config.max_depth:
                leaf_value = self._leaf_value(node)
                break

            if not node.is_expanded:
                self._expand_node(node)
                leaf_value = self._leaf_value(node)
                break

            action_id = self._select_action_id(node)

            if action_id is None:
                leaf_value = self._leaf_value(node)
                break

            child = node.children.get(action_id)

            if child is None:
                child = self._create_child(node, action_id)

                if child is None:
                    # If action failed unexpectedly, remove it from priors.
                    node.action_priors.pop(action_id, None)
                    node.actions_by_id.pop(action_id, None)
                    leaf_value = self._leaf_value(node)
                    break

                node.children[action_id] = child
                path.append(child)

                if not child.done and child.depth < self.config.max_depth:
                    self._expand_node(child)

                leaf_value = self._leaf_value(child)
                break

            path.append(child)
            node = child

        root.visit_count += 1

        self._backup(path, leaf_value)

    def _expand_node(self, node: MCTSNode) -> None:
        """
        Compute candidate actions and priors for a node.

        If neural evaluator is available:
            use neural policy as action prior.

        If neural evaluator is not available:
            use heuristic loading-based prior.
        """

        if node.done or node.depth >= self.config.max_depth:
            node.is_expanded = True
            return

        state = node.env.current_state

        if state is None:
            node.is_expanded = True
            return

        valid_actions = node.env.valid_actions()
        action_mask = node.env.valid_action_mask()

        neural_policy = None

        if self.evaluator is not None:
            neural_policy, _ = self.evaluator.evaluate(
                state=state,
                action_mask=action_mask,
            )

        stop_actions = [
            action for action in valid_actions if action.action_type == "do_nothing"
        ]

        switch_actions = [
            action for action in valid_actions if action.action_type == "switch_off_branch"
        ]

        selected: list[GridFMAction] = []

        if self._should_include_stop_action(state):
            selected.extend(stop_actions)

        if neural_policy is not None:
            # Main candidate source: actions preferred by the neural policy.
            switch_by_policy = sorted(
                switch_actions,
                key=lambda action: float(neural_policy[action.action_id]),
                reverse=True,
            )

            # Safety backup: also keep some physically high-loaded branches.
            # This prevents a weak early network from completely ignoring
            # obviously important overloaded corridors.
            switch_by_loading = sorted(
                switch_actions,
                key=lambda action: float(
                    state.branch_features[action.branch_pos, self.loading_idx]
                ),
                reverse=True,
            )

            selected_switches: list[GridFMAction] = []
            seen_action_ids: set[int] = set()

            for action in switch_by_policy[: self.config.top_k_actions]:
                if action.action_id not in seen_action_ids:
                    selected_switches.append(action)
                    seen_action_ids.add(action.action_id)

            loading_backup_k = max(5, self.config.top_k_actions // 4)

            for action in switch_by_loading[:loading_backup_k]:
                if action.action_id not in seen_action_ids:
                    selected_switches.append(action)
                    seen_action_ids.add(action.action_id)

            selected.extend(selected_switches)

        else:
            # Heuristic fallback: top-K actions by current branch loading.
            switch_actions = sorted(
                switch_actions,
                key=lambda action: float(
                    state.branch_features[action.branch_pos, self.loading_idx]
                ),
                reverse=True,
            )

            if self.config.top_k_actions > 0:
                switch_actions = switch_actions[: self.config.top_k_actions]

            selected.extend(switch_actions)

        if not selected:
            node.is_expanded = True
            return

        raw_scores: dict[int, float] = {}

        for action in selected:
            node.actions_by_id[action.action_id] = action

            if neural_policy is not None:
                score = float(neural_policy[action.action_id])
                raw_scores[action.action_id] = max(score, 1e-8)
            else:
                if action.action_type == "do_nothing":
                    raw_scores[action.action_id] = self.config.stop_prior
                else:
                    loading = float(
                        state.branch_features[action.branch_pos, self.loading_idx]
                    )

                    base_score = max(
                        loading - 80.0,
                        self.config.min_switch_prior_score,
                    )

                    score = base_score ** self.config.prior_exponent
                    raw_scores[action.action_id] = score

        score_sum = sum(raw_scores.values())

        if score_sum <= 0:
            uniform = 1.0 / len(raw_scores)
            node.action_priors = {
                action_id: uniform for action_id in raw_scores
            }
        else:
            node.action_priors = {
                action_id: score / score_sum
                for action_id, score in raw_scores.items()
            }

        node.is_expanded = True

    def _select_action_id(self, node: MCTSNode) -> int | None:
        """
        Select action by PUCT.

        score = Q + U

        Q:
            mean value from previous simulations.

        U:
            exploration term based on prior and visit counts.
        """

        if not node.action_priors:
            return None

        best_action_id = None
        best_score = -float("inf")

        parent_visits = max(node.visit_count, 1)
        sqrt_parent_visits = sqrt(parent_visits)

        for action_id, prior in node.action_priors.items():
            child = node.children.get(action_id)

            if child is None:
                child_visits = 0
                q_value = self.config.fpu_value
            else:
                child_visits = child.visit_count
                q_value = child.mean_value

            exploration = (
                self.config.c_puct
                * prior
                * sqrt_parent_visits
                / (1 + child_visits)
            )

            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_action_id = action_id

        return best_action_id

    def _create_child(
        self,
        parent: MCTSNode,
        action_id: int,
    ) -> MCTSNode | None:
        """
        Apply one action to a cloned environment and create child node.
        """

        action = parent.actions_by_id.get(action_id)

        if action is None:
            return None

        child_env = parent.env.clone()

        try:
            step_result = child_env.step(action_id)
        except Exception:
            return None

        child = MCTSNode(
            env=child_env,
            depth=parent.depth + 1,
            prior=parent.action_priors.get(action_id, 0.0),
            action_id_from_parent=int(action_id),
            branch_id_from_parent=(
                None if action.branch_id is None else int(action.branch_id)
            ),
            reward_from_parent=float(step_result.reward),
            step_result_from_parent=step_result,
        )

        return child

    def _backup(self, path: list[MCTSNode], leaf_value: float) -> None:
        """
        Back up discounted return through the selected path.

        If path is:
            root -> child1 -> child2

        We propagate:
            G(child2) = reward2 + gamma * leaf_value
            G(child1) = reward1 + gamma * G(child2)
        """

        value = float(leaf_value)

        for node in reversed(path):
            scaled_reward = self._scale_value(node.reward_from_parent)
            value = float(scaled_reward + self.config.gamma * value)

            node.visit_count += 1
            node.total_value += value

    def _leaf_value(self, node: MCTSNode) -> float:
        """
        Value of a leaf state.

        If neural evaluator is available:
            use neural value for non-terminal states.

        If no evaluator is available:
            use heuristic penalty-based value.
        """

        state = node.env.current_state

        if state is None:
            return self._scale_value(-1000.0)

        penalty = self._state_penalty(state)

        if node.solved:
            return 0.0

        if node.done and not node.solved:
            raw_value = -self.config.terminal_unsolved_penalty - (
                    self.config.leaf_penalty_weight * penalty
            )
            return self._scale_value(raw_value)

        if self.evaluator is not None:
            action_mask = node.env.valid_action_mask()

            _, neural_value = self.evaluator.evaluate(
                state=state,
                action_mask=action_mask,
            )

            return float(neural_value)

        raw_value = -self.config.leaf_penalty_weight * penalty

        return self._scale_value(raw_value)

    def _scale_value(self, value: float) -> float:
        """
        Scale raw environment rewards/values for MCTS internals.

        Environment rewards may be hundreds or thousands.
        PUCT works better when Q values are roughly in a small range.
        """

        scaled = float(value) / float(self.config.value_scale)

        # Keep extreme failures from completely dominating the tree.
        return float(np.clip(scaled, -5.0, 5.0))

    def _state_penalty(self, state: GridFMState) -> float:
        """
        Same security penalty idea as in reward.py.

        Lower penalty means better state.
        """

        active_loading = self._active_loadings(state)

        total_overload = float(np.sum(np.maximum(active_loading - 100.0, 0.0)))
        hard_overload = float(np.sum(np.maximum(active_loading - 120.0, 0.0)))

        num_overloaded = int(state.metrics["num_overloaded_branches"])
        num_hard_overloaded = int(state.metrics["num_hard_overloaded_branches"])

        voltage_penalty = float(state.metrics.get("total_voltage_violation", 0.0))

        penalty = (
            2.0 * total_overload
            + 5.0 * hard_overload
            + 10.0 * num_overloaded
            + 30.0 * num_hard_overloaded
            + 500.0 * voltage_penalty
        )

        return float(penalty)

    def _active_loadings(self, state: GridFMState) -> np.ndarray:
        status = state.branch_features[:, self.status_idx]
        loading = state.branch_features[:, self.loading_idx]

        return loading[status > 0]

    def _principal_variation(
        self,
        root: MCTSNode,
    ) -> tuple[list[int], list[int | None], list[float], float, dict[str, Any]]:
        """
        Follow the most visited child from root until the path ends.
        """

        action_ids: list[int] = []
        branch_ids: list[int | None] = []
        rewards: list[float] = []

        node = root
        discounted_return = 0.0
        discount = 1.0

        while node.children:
            best_action_id = max(
                node.children,
                key=lambda action_id: node.children[action_id].visit_count,
            )

            child = node.children[best_action_id]

            action_ids.append(int(best_action_id))
            branch_ids.append(child.branch_id_from_parent)
            rewards.append(float(child.reward_from_parent))

            discounted_return += discount * float(child.reward_from_parent)
            discount *= self.config.gamma

            node = child

            if node.done:
                break

        final_state = node.env.current_state

        if final_state is None:
            final_metrics: dict[str, Any] = {}
        else:
            final_metrics = dict(final_state.metrics)

        return action_ids, branch_ids, rewards, float(discounted_return), final_metrics