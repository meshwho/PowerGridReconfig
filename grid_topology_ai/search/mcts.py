from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.physical_objective import (
    assess_physical_state,
    stop_allowed_for_policy,
)
from grid_topology_ai.return_contract import (
    DEFAULT_HEURISTIC_UTILITY_SCALE,
    heuristic_terminal_utility_estimate,
    require_bounded_utility,
    require_discount_factor,
    terminal_utility_from_outcome,
)
from grid_topology_ai.search.dc_action_screener import DCActionScreener

if TYPE_CHECKING:
    from grid_topology_ai.models.neural_evaluator import NeuralPolicyValueEvaluator


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for single-agent AlphaZero-style MCTS.

    ``gamma`` discounts the common terminal utility contract. Dense environment
    rewards are retained on nodes for diagnostics, but never enter PUCT backup.
    """

    num_simulations: int = 100
    max_depth: int = 4
    top_k_actions: int = 30
    gamma: float = 0.95
    c_puct: float = 1.5

    # Temporary constructor compatibility for callers migrated in the next
    # commit. This value is ignored and never enters leaf evaluation or backup.
    leaf_penalty_weight: float = 0.0

    heuristic_utility_scale: float = DEFAULT_HEURISTIC_UTILITY_SCALE
    include_stop_action: bool = True
    stop_prior: float = 1.0
    fpu_value: float = -0.25

    prior_exponent: float = 0.5
    min_switch_prior_score: float = 1.0
    stop_policy: str = "no_hard_overloads"

    use_root_dirichlet_noise: bool = False
    root_dirichlet_alpha: float = 0.30
    root_exploration_fraction: float = 0.25
    random_seed: int | None = None

    use_dc_screening: bool = False
    dc_top_k_actions: int = 30
    dc_candidate_pool: int = 120
    dc_keep_policy_actions: int = 5
    dc_keep_loading_actions: int = 5
    dc_policy_weight: float = 0.0
    dc_failure_penalty: float = 1_000_000_000.0
    dc_max_depth: int = 0


@dataclass
class MCTSNode:
    """One node in the MCTS tree."""

    env: TopologySwitchingEnv
    depth: int
    prior: float = 1.0

    action_id_from_parent: int | None = None
    branch_id_from_parent: int | None = None
    # Diagnostic potential shaping only; never used in backup or selection.
    reward_from_parent: float = 0.0
    step_result_from_parent: TopologyStepResult | None = None

    visit_count: int = 0
    total_value: float = 0.0
    neural_value: float | None = None
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
    """Result of one MCTS search from one scenario."""

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
    """AlphaZero-style MCTS planner for topology switching.

    Neural leaf values and MCTS Q values share the exact same semantics:
    expected discounted terminal utility in ``[-1, 1]``.
    """

    def __init__(
        self,
        config: MCTSConfig,
        evaluator: "NeuralPolicyValueEvaluator | None" = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.config = config
        self.evaluator = evaluator
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG
        self.gamma = require_discount_factor(config.gamma)
        self.heuristic_utility_scale = float(config.heuristic_utility_scale)
        if (
            not np.isfinite(self.heuristic_utility_scale)
            or self.heuristic_utility_scale <= 0
        ):
            raise ValueError("heuristic_utility_scale must be finite and > 0")
        require_bounded_utility(config.fpu_value, context="MCTS fpu_value")

        self.loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
        self.rng = np.random.default_rng(config.random_seed)

        self.dc_screener = None
        if self.config.use_dc_screening:
            self.dc_screener = DCActionScreener(
                top_k=self.config.dc_top_k_actions,
                candidate_pool=self.config.dc_candidate_pool,
                policy_weight=self.config.dc_policy_weight,
                failure_penalty=self.config.dc_failure_penalty,
                enable_cache=True,
                physics_config=self.physics_config,
            )

    def search(
        self,
        env: TopologySwitchingEnv,
        scenario_id: int,
    ) -> MCTSResult:
        root_env = env.clone()
        root_env.reset(int(scenario_id))
        return self.search_from_env(root_env)

    def search_from_env(self, env: TopologySwitchingEnv) -> MCTSResult:
        if env.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        root_env = env.clone()
        scenario_id = (
            int(root_env.initial_scenario_id)
            if root_env.initial_scenario_id is not None
            else int(root_env.current_state.scenario_id)
        )
        root = MCTSNode(env=root_env, depth=0, prior=1.0)
        self._expand_node(root)
        self._add_root_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            self._run_one_simulation(root)

        visit_counts = {
            action_id: child.visit_count
            for action_id, child in root.children.items()
        }
        total_visits = sum(visit_counts.values())
        policy = (
            {
                action_id: count / total_visits
                for action_id, count in visit_counts.items()
            }
            if total_visits > 0
            else {}
        )

        best_action_id = None
        best_branch_id = None
        if visit_counts:
            best_action_id = max(
                visit_counts,
                key=lambda action_id: visit_counts[action_id],
            )
            best_branch_id = root.children[best_action_id].branch_id_from_parent

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

    def _add_root_dirichlet_noise(self, root: MCTSNode) -> None:
        if not self.config.use_root_dirichlet_noise or not root.action_priors:
            return
        action_ids = list(root.action_priors)
        alpha = float(self.config.root_dirichlet_alpha)
        epsilon = float(self.config.root_exploration_fraction)
        if alpha <= 0.0 or epsilon <= 0.0:
            return
        noise = self.rng.dirichlet(alpha=[alpha for _ in action_ids])
        for action_id, noise_value in zip(action_ids, noise, strict=True):
            old_prior = float(root.action_priors[action_id])
            root.action_priors[action_id] = (
                (1.0 - epsilon) * old_prior
                + epsilon * float(noise_value)
            )
        total = sum(root.action_priors.values())
        if total > 0.0:
            root.action_priors = {
                action_id: prior / total
                for action_id, prior in root.action_priors.items()
            }

    def _should_include_stop_action(self, state: GridFMState) -> bool:
        return stop_allowed_for_policy(
            assess_physical_state(state.metrics),
            stop_policy=self.config.stop_policy,
            include_stop_action=self.config.include_stop_action,
        )

    def _run_one_simulation(self, root: MCTSNode) -> None:
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
            neural_policy, neural_value = self.evaluator.evaluate(
                state=state,
                action_mask=action_mask,
            )
            node.neural_value = require_bounded_utility(
                neural_value,
                context="neural leaf value",
            )

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
        selected: list[GridFMAction] = []
        if self._should_include_stop_action(state):
            selected.extend(stop_actions)

        if neural_policy is not None:
            switch_by_policy = sorted(
                switch_actions,
                key=lambda action: float(neural_policy[action.action_id]),
                reverse=True,
            )
            switch_by_loading = sorted(
                switch_actions,
                key=lambda action: float(
                    state.branch_features[action.branch_pos, self.loading_idx]
                ),
                reverse=True,
            )
            selected_switches: list[GridFMAction] = []
            seen_action_ids: set[int] = set()

            if self.config.use_dc_screening and self.dc_screener is not None:
                dc_pool: list[GridFMAction] = []
                dc_pool_seen: set[int] = set()
                if self.config.dc_candidate_pool <= 0:
                    pool_from_policy = switch_by_policy
                    pool_from_loading = switch_by_loading
                else:
                    pool_from_policy = switch_by_policy[
                        : self.config.dc_candidate_pool
                    ]
                    loading_pool_k = max(
                        self.config.dc_keep_loading_actions,
                        self.config.dc_candidate_pool // 4,
                    )
                    pool_from_loading = switch_by_loading[:loading_pool_k]

                for action in [*pool_from_policy, *pool_from_loading]:
                    if action.action_id in dc_pool_seen:
                        continue
                    dc_pool.append(action)
                    dc_pool_seen.add(action.action_id)

                dc_ranked = self.dc_screener.screen_actions(
                    state=state,
                    actions=dc_pool,
                    backend=node.env.backend,
                    neural_policy=neural_policy,
                    top_k=self.config.dc_top_k_actions,
                )
                for action in dc_ranked:
                    if action.action_id not in seen_action_ids:
                        selected_switches.append(action)
                        seen_action_ids.add(action.action_id)
                for action in switch_by_policy[
                    : self.config.dc_keep_policy_actions
                ]:
                    if action.action_id not in seen_action_ids:
                        selected_switches.append(action)
                        seen_action_ids.add(action.action_id)
                for action in switch_by_loading[
                    : self.config.dc_keep_loading_actions
                ]:
                    if action.action_id not in seen_action_ids:
                        selected_switches.append(action)
                        seen_action_ids.add(action.action_id)
            else:
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
                raw_scores[action.action_id] = max(
                    float(neural_policy[action.action_id]),
                    1e-8,
                )
            elif action.action_type == "do_nothing":
                raw_scores[action.action_id] = self.config.stop_prior
            else:
                loading = float(
                    state.branch_features[action.branch_pos, self.loading_idx]
                )
                base_score = max(
                    loading - 80.0,
                    self.config.min_switch_prior_score,
                )
                raw_scores[action.action_id] = (
                    base_score ** self.config.prior_exponent
                )

        score_sum = sum(raw_scores.values())
        if score_sum <= 0.0:
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
        if not node.action_priors:
            return None
        best_action_id = None
        best_score = -float("inf")
        sqrt_parent_visits = sqrt(max(node.visit_count, 1))
        for action_id, prior in node.action_priors.items():
            child = node.children.get(action_id)
            if child is None:
                child_visits = 0
                q_value = float(self.config.fpu_value)
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
        action = parent.actions_by_id.get(action_id)
        if action is None:
            return None
        child_env = parent.env.clone()
        try:
            step_result = child_env.step(action)
        except Exception:
            return None
        return MCTSNode(
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

    def _backup(self, path: list[MCTSNode], leaf_value: float) -> None:
        """Back up only discounted terminal utility.

        ``reward_from_parent`` is intentionally diagnostic and does not enter Q.
        A leaf value describes the leaf state; every traversed edge therefore
        contributes exactly one factor of ``gamma``.
        """

        value = require_bounded_utility(
            leaf_value,
            context="MCTS leaf utility",
        )
        for node in reversed(path):
            value = float(self.gamma * value)
            node.visit_count += 1
            node.total_value += value

    def _leaf_value(self, node: MCTSNode) -> float:
        """Evaluate a leaf under the shared terminal-utility contract."""

        if node.done:
            terminal_utility, _ = terminal_utility_from_outcome(
                node.solved,
                getattr(node.env, "termination_reason", None),
            )
            return terminal_utility

        state = node.env.current_state
        if state is None:
            return -1.0

        if self.evaluator is not None:
            if node.neural_value is not None:
                return require_bounded_utility(
                    node.neural_value,
                    context="cached neural leaf value",
                )
            action_mask = node.env.valid_action_mask()
            _, neural_value = self.evaluator.evaluate(
                state=state,
                action_mask=action_mask,
            )
            node.neural_value = require_bounded_utility(
                neural_value,
                context="neural leaf value",
            )
            return node.neural_value

        return heuristic_terminal_utility_estimate(
            state,
            physics_config=self.physics_config,
            utility_scale=self.heuristic_utility_scale,
        )

    def _principal_variation(
        self,
        root: MCTSNode,
    ) -> tuple[list[int], list[int | None], list[float], float, dict[str, Any]]:
        """Follow the most visited path and report shaping diagnostics."""

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
            discount *= self.gamma
            node = child
            if node.done:
                break

        final_state = node.env.current_state
        final_metrics = {} if final_state is None else dict(final_state.metrics)
        return (
            action_ids,
            branch_ids,
            rewards,
            float(discounted_return),
            final_metrics,
        )
