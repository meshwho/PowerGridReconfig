from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import floor, isfinite, sqrt
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.physics.objective import (
    assess_physical_state,
    stop_allowed_for_policy,
)
from grid_topology_ai.physics.utility import (
    CONTINUATION_GRID_UTILITY_WEIGHTS,
    CONTINUATION_SWITCH_PENALTY,
    state_security_penalty,
)
from grid_topology_ai.search.screening import DCActionScreener
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.value_targets import (
    DEFAULT_HEURISTIC_UTILITY_SCALE,
    TERMINAL_UTILITY_GAMMA,
    heuristic_terminal_utility_estimate,
    require_bounded_utility,
    require_discount_factor,
    terminal_utility_from_outcome,
)

if TYPE_CHECKING:
    from grid_topology_ai.evaluator import NeuralPolicyValueEvaluator


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for single-agent AlphaZero-style MCTS.

    Terminal utility is backed up without temporal discount. Dense environment
    rewards are retained on nodes for diagnostics, but never enter PUCT backup.
    """

    num_simulations: int = 100
    max_depth: int = 4
    top_k_actions: int = 30
    widening_coefficient: float = 2.0
    widening_exponent: float = 0.5
    exploration_quota: int = 2
    gamma: float = TERMINAL_UTILITY_GAMMA
    c_puct: float = 1.5

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

    # -1 enables DC screening at every depth.
    # 0 enables it only at the root.
    # A positive value enables it through that node depth.
    dc_max_depth: int = 0

    def __post_init__(self) -> None:

        gamma = require_discount_factor(self.gamma)

        if gamma != TERMINAL_UTILITY_GAMMA:
            raise ValueError(
                "MCTS gamma must be exactly 1.0 for undiscounted terminal utility"
            )

        object.__setattr__(
            self,
            "gamma",
            gamma,
        )

        if isinstance(self.dc_max_depth, bool) or not isinstance(
            self.dc_max_depth,
            (int, np.integer),
        ):
            raise ValueError("dc_max_depth must be -1 or a non-negative integer.")

        dc_max_depth = int(self.dc_max_depth)

        if dc_max_depth < -1:
            raise ValueError("dc_max_depth must be -1 or a non-negative integer.")

        object.__setattr__(
            self,
            "dc_max_depth",
            dc_max_depth,
        )

        if isinstance(self.exploration_quota, bool) or not isinstance(
            self.exploration_quota,
            (int, np.integer),
        ):
            raise ValueError("exploration_quota must be a non-negative integer.")

        exploration_quota = int(self.exploration_quota)

        if exploration_quota < 0:
            raise ValueError("exploration_quota must be a non-negative integer.")

        object.__setattr__(
            self,
            "exploration_quota",
            exploration_quota,
        )

        if isinstance(
            self.widening_coefficient,
            bool,
        ):
            raise ValueError(
                "widening_coefficient must be a finite non-negative number."
            )

        if isinstance(
            self.widening_exponent,
            bool,
        ):
            raise ValueError("widening_exponent must be a finite number in (0, 1].")

        coefficient = float(self.widening_coefficient)
        exponent = float(self.widening_exponent)

        if not isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                "widening_coefficient must be a finite non-negative number."
            )

        if not isfinite(exponent) or exponent <= 0.0 or exponent > 1.0:
            raise ValueError("widening_exponent must be a finite number in (0, 1].")

        object.__setattr__(
            self,
            "widening_coefficient",
            coefficient,
        )
        object.__setattr__(
            self,
            "widening_exponent",
            exponent,
        )


@dataclass
class MCTSNode:
    """One node in the MCTS tree."""

    env: TopologySwitchingEnv
    depth: int
    prior: float = 1.0

    action_from_parent: GridFMAction | None = None
    # Diagnostic potential shaping only; never used in backup or selection.
    reward_from_parent: float = 0.0
    step_result_from_parent: TopologyStepResult | None = None

    visit_count: int = 0
    total_value: float = 0.0
    neural_value: float | None = None
    is_expanded: bool = False

    # Complete legal action order retained for later widening.
    ranked_actions: list[GridFMAction] = field(default_factory=list)
    action_scores: dict[int, float] = field(default_factory=dict)

    # Effective PUCT weights. Root Dirichlet noise modifies these
    # without changing the retained legal action ranking.
    selection_scores: dict[int, float] = field(default_factory=dict)

    # Tail actions that still require one trial.
    forced_exploration_action_ids: list[int] = field(default_factory=list)

    # Only actions currently available to PUCT.
    action_priors: dict[int, float] = field(default_factory=dict)
    actions_by_id: dict[int, GridFMAction] = field(default_factory=dict)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def action_id_from_parent(self) -> int | None:
        if self.action_from_parent is None:
            return None

        return int(self.action_from_parent.action_id)

    @property
    def branch_id_from_parent(self) -> int | None:
        if self.action_from_parent is None or self.action_from_parent.branch_id is None:
            return None

        return int(self.action_from_parent.branch_id)

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

    root_legal_action_count: int
    root_considered_action_count: int
    root_visited_action_count: int
    root_action_coverage: float
    root_visited_action_coverage: float

    config: MCTSConfig


class MCTSPlanner:
    """AlphaZero-style MCTS planner for topology switching.

    Neural leaf values and MCTS Q values share the exact same semantics:
    expected terminal utility in ``[-1, 1]``.
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

    def reset_rng(
        self,
        random_seed: int | None,
    ) -> None:
        self.rng = np.random.default_rng(random_seed)

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
            action_id: child.visit_count for action_id, child in root.children.items()
        }

        root_legal_action_count = len(root.ranked_actions)
        root_considered_action_count = len(root.actions_by_id)
        root_visited_action_count = sum(
            1 for child in root.children.values() if child.visit_count > 0
        )

        root_action_coverage = self._action_coverage_rate(
            root_considered_action_count,
            root_legal_action_count,
        )
        root_visited_action_coverage = self._action_coverage_rate(
            root_visited_action_count,
            root_legal_action_count,
        )

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
            root_legal_action_count=(root_legal_action_count),
            root_considered_action_count=(root_considered_action_count),
            root_visited_action_count=(root_visited_action_count),
            root_action_coverage=(root_action_coverage),
            root_visited_action_coverage=(root_visited_action_coverage),
            config=self.config,
        )

    @staticmethod
    def _action_coverage_rate(
        count: int,
        legal_count: int,
    ) -> float:
        if legal_count <= 0:
            return 0.0

        return float(count) / float(legal_count)

    def _loading_priority(
        self,
        state: GridFMState,
        action: GridFMAction,
    ) -> float | None:
        if (
            action.kind != "set_branch_status"
            or action.target_status != 0
            or action.branch_pos is None
        ):
            return None

        return float(
            state.branch_features[
                action.branch_pos,
                self.loading_idx,
            ]
        )

    def _add_root_dirichlet_noise(
        self,
        root: MCTSNode,
    ) -> None:
        if not self.config.use_root_dirichlet_noise or not root.action_priors:
            return

        action_ids = list(root.action_priors)
        alpha = float(self.config.root_dirichlet_alpha)
        epsilon = float(self.config.root_exploration_fraction)

        if alpha <= 0.0 or epsilon <= 0.0:
            return

        noise = self.rng.dirichlet(alpha=[alpha for _ in action_ids])

        noisy_priors: dict[int, float] = {}

        for action_id, noise_value in zip(
            action_ids,
            noise,
            strict=True,
        ):
            old_prior = float(root.action_priors[action_id])
            noisy_priors[action_id] = (1.0 - epsilon) * old_prior + epsilon * float(
                noise_value
            )

        total = sum(noisy_priors.values())

        if total <= 0.0:
            return

        noisy_priors = {
            action_id: prior / total for action_id, prior in noisy_priors.items()
        }

        active_score_sum = sum(
            float(
                root.selection_scores.get(
                    action_id,
                    root.action_scores[action_id],
                )
            )
            for action_id in action_ids
        )

        if active_score_sum <= 0.0:
            active_score_sum = 1.0

        for action_id, prior in noisy_priors.items():
            root.selection_scores[action_id] = prior * active_score_sum

        root.action_priors = noisy_priors

    def _should_include_stop_action(self, state: GridFMState) -> bool:
        return stop_allowed_for_policy(
            assess_physical_state(state.metrics),
            stop_policy=self.config.stop_policy,
            include_stop_action=self.config.include_stop_action,
        )

    def _mcts_action_mask(
        self,
        *,
        state: GridFMState,
        operational_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply search-specific legality to the operational mask.

        The action space decides structural and operational
        validity. MCTS additionally decides whether the stop
        action is legal under the configured stop policy.
        """

        mask = np.asarray(
            operational_mask,
            dtype=bool,
        ).copy()

        if mask.ndim != 1:
            raise ValueError("MCTS action mask must be one-dimensional.")

        if mask.size == 0:
            raise ValueError("MCTS action mask must contain the stop action.")

        mask[0] = self._should_include_stop_action(state)

        return mask

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

            self._widen_node(node)

            action_id = self._next_forced_exploration_action(node)

            if action_id is None:
                action_id = self._select_action_id(node)
            if action_id is None:
                leaf_value = self._leaf_value(node)
                break

            child = node.children.get(action_id)
            if child is None:
                child = self._create_child(node, action_id)
                if child is None:
                    self._discard_action(
                        node,
                        action_id,
                    )
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

    @staticmethod
    def _extend_unique_actions(
        target: list[GridFMAction],
        source: list[GridFMAction],
        seen_action_ids: set[int],
    ) -> None:
        for action in source:
            action_id = int(action.action_id)

            if action_id in seen_action_ids:
                continue

            target.append(action)
            seen_action_ids.add(action_id)

    @staticmethod
    def _discard_action(
        node: MCTSNode,
        action_id: int,
    ) -> None:
        action_id = int(action_id)

        node.children.pop(action_id, None)
        node.action_priors.pop(action_id, None)
        node.actions_by_id.pop(action_id, None)
        node.action_scores.pop(action_id, None)
        node.selection_scores.pop(action_id, None)

        node.ranked_actions = [
            action
            for action in node.ranked_actions
            if int(action.action_id) != action_id
        ]

        node.forced_exploration_action_ids = [
            candidate_id
            for candidate_id in node.forced_exploration_action_ids
            if candidate_id != action_id
        ]

    @staticmethod
    def _set_active_actions(
        node: MCTSNode,
        actions: list[GridFMAction],
    ) -> None:
        actions_by_id: dict[int, GridFMAction] = {}

        for action in actions:
            action_id = int(action.action_id)

            if action_id not in actions_by_id:
                actions_by_id[action_id] = action

        if not actions_by_id:
            node.actions_by_id = {}
            node.action_priors = {}
            return

        active_scores = {
            action_id: float(
                node.selection_scores.get(
                    action_id,
                    node.action_scores[action_id],
                )
            )
            for action_id in actions_by_id
        }

        score_sum = sum(active_scores.values())

        if score_sum <= 0.0:
            uniform = 1.0 / len(active_scores)
            action_priors = {action_id: uniform for action_id in active_scores}
        else:
            action_priors = {
                action_id: score / score_sum
                for action_id, score in active_scores.items()
            }

        node.actions_by_id = actions_by_id
        node.action_priors = action_priors

    def _choose_exploration_actions(
        self,
        *,
        ranked_switches: list[GridFMAction],
        initial_switches: list[GridFMAction],
    ) -> list[GridFMAction]:
        quota = int(self.config.exploration_quota)

        if quota <= 0:
            return []

        initial_action_ids = {int(action.action_id) for action in initial_switches}

        tail = [
            action
            for action in ranked_switches
            if int(action.action_id) not in initial_action_ids
        ]

        if not tail:
            return []

        count = min(
            quota,
            len(tail),
        )

        indices = self.rng.permutation(len(tail))[:count]

        return [tail[int(index)] for index in indices]

    @staticmethod
    def _next_forced_exploration_action(
        node: MCTSNode,
    ) -> int | None:
        for action_id in node.forced_exploration_action_ids:
            if action_id not in node.actions_by_id:
                continue

            child = node.children.get(action_id)

            if child is None or child.visit_count == 0:
                return action_id

        return None

    def _target_switch_width(
        self,
        node: MCTSNode,
    ) -> int:
        ranked_switches = [
            action for action in node.ranked_actions if action.kind != "stop"
        ]

        total_switches = len(ranked_switches)

        if total_switches == 0:
            return 0

        if self.config.top_k_actions <= 0:
            return total_switches

        current_width = 0

        for action in ranked_switches:
            action_id = int(action.action_id)

            if action_id not in node.actions_by_id:
                break

            current_width += 1

        initial_width = min(
            int(self.config.top_k_actions),
            total_switches,
        )

        visits = max(
            int(node.visit_count),
            0,
        )

        if visits == 0:
            growth = 0
        else:
            growth = floor(
                self.config.widening_coefficient
                * (visits**self.config.widening_exponent)
            )

        return min(
            total_switches,
            max(
                current_width,
                initial_width + growth,
            ),
        )

    def _widen_node(
        self,
        node: MCTSNode,
    ) -> bool:
        if not node.is_expanded or not node.ranked_actions:
            return False

        ranked_switches = [
            action for action in node.ranked_actions if action.kind != "stop"
        ]

        target_width = self._target_switch_width(node)

        ranked_prefix = ranked_switches[:target_width]

        if all(int(action.action_id) in node.actions_by_id for action in ranked_prefix):
            return False

        active_actions = [
            action
            for action in node.ranked_actions
            if int(action.action_id) in node.actions_by_id
        ]
        active_action_ids = {int(action.action_id) for action in active_actions}

        self._extend_unique_actions(
            active_actions,
            ranked_prefix,
            active_action_ids,
        )

        self._set_active_actions(
            node,
            active_actions,
        )

        return True

    def _should_use_dc_screening(
        self,
        node: MCTSNode,
    ) -> bool:
        if not self.config.use_dc_screening or self.dc_screener is None:
            return False

        max_depth = int(self.config.dc_max_depth)

        return max_depth == -1 or int(node.depth) <= max_depth

    def _expand_node(
        self,
        node: MCTSNode,
    ) -> None:
        if node.done or node.depth >= self.config.max_depth:
            node.is_expanded = True
            return

        state = node.env.current_state

        if state is None:
            node.is_expanded = True
            return

        operational_mask = node.env.operational_action_mask()

        action_mask = self._mcts_action_mask(
            state=state,
            operational_mask=operational_mask,
        )

        valid_actions: list[GridFMAction] = []

        for action in node.env.valid_actions():
            action_id = int(action.action_id)

            if action_id < 0 or action_id >= action_mask.size:
                raise RuntimeError(
                    "Environment returned action_id "
                    f"{action_id} outside action mask of "
                    f"size {action_mask.size}."
                )

            if bool(action_mask[action_id]):
                valid_actions.append(action)

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

        stop_actions = [action for action in valid_actions if action.kind == "stop"]
        switch_actions = [action for action in valid_actions if action.kind != "stop"]

        loading_priorities: dict[int, float] = {}
        unscored_switches: list[GridFMAction] = []

        for action in switch_actions:
            loading = self._loading_priority(
                state,
                action,
            )

            if loading is None:
                unscored_switches.append(action)
                continue

            loading_priorities[int(action.action_id)] = float(loading)

        switch_by_loading = sorted(
            [
                action
                for action in switch_actions
                if int(action.action_id) in loading_priorities
            ],
            key=lambda action: loading_priorities[int(action.action_id)],
            reverse=True,
        )

        unscored_switches.sort(key=lambda action: int(action.action_id))

        ranked_switches: list[GridFMAction]
        initial_switches: list[GridFMAction]

        if neural_policy is not None:
            switch_by_policy = sorted(
                switch_actions,
                key=lambda action: float(neural_policy[action.action_id]),
                reverse=True,
            )
            initial_switches = []
            initial_seen: set[int] = set()

            if self._should_use_dc_screening(node):
                dc_pool: list[GridFMAction] = []
                dc_pool_seen: set[int] = set()

                if self.config.dc_candidate_pool <= 0:
                    pool_from_policy = switch_by_policy
                    pool_from_loading = switch_by_loading
                else:
                    pool_from_policy = switch_by_policy[: self.config.dc_candidate_pool]
                    loading_pool_k = max(
                        self.config.dc_keep_loading_actions,
                        self.config.dc_candidate_pool // 4,
                    )
                    pool_from_loading = switch_by_loading[:loading_pool_k]

                self._extend_unique_actions(
                    dc_pool,
                    pool_from_policy,
                    dc_pool_seen,
                )
                self._extend_unique_actions(
                    dc_pool,
                    pool_from_loading,
                    dc_pool_seen,
                )

                dc_ranked = self.dc_screener.rank_actions(
                    state=state,
                    actions=dc_pool,
                    backend=node.env.backend,
                    neural_policy=neural_policy,
                )

                if self.config.dc_top_k_actions > 0:
                    dc_ranked = dc_ranked[: self.config.dc_top_k_actions]

                self._extend_unique_actions(
                    initial_switches,
                    dc_ranked,
                    initial_seen,
                )
                self._extend_unique_actions(
                    initial_switches,
                    switch_by_policy[: self.config.dc_keep_policy_actions],
                    initial_seen,
                )
                self._extend_unique_actions(
                    initial_switches,
                    switch_by_loading[: self.config.dc_keep_loading_actions],
                    initial_seen,
                )
            else:
                self._extend_unique_actions(
                    initial_switches,
                    switch_by_policy[: self.config.top_k_actions],
                    initial_seen,
                )

                loading_backup_k = max(
                    5,
                    self.config.top_k_actions // 4,
                )
                self._extend_unique_actions(
                    initial_switches,
                    switch_by_loading[:loading_backup_k],
                    initial_seen,
                )

            ranked_switches = []
            ranked_seen: set[int] = set()

            # Preserve the existing shortlist order first.
            self._extend_unique_actions(
                ranked_switches,
                initial_switches,
                ranked_seen,
            )

            # Retain the complete legal tail instead of
            # permanently pruning it.
            self._extend_unique_actions(
                ranked_switches,
                switch_by_policy,
                ranked_seen,
            )
            self._extend_unique_actions(
                ranked_switches,
                switch_by_loading,
                ranked_seen,
            )
        else:
            ranked_switches = [
                *switch_by_loading,
                *unscored_switches,
            ]

            if self.config.top_k_actions > 0:
                initial_switches = [
                    *switch_by_loading[: self.config.top_k_actions],
                    *unscored_switches,
                ]
            else:
                initial_switches = list(ranked_switches)

        ranked_actions: list[GridFMAction] = []
        ranked_seen: set[int] = set()

        self._extend_unique_actions(
            ranked_actions,
            stop_actions,
            ranked_seen,
        )
        self._extend_unique_actions(
            ranked_actions,
            ranked_switches,
            ranked_seen,
        )

        active_actions: list[GridFMAction] = []
        active_seen: set[int] = set()

        self._extend_unique_actions(
            active_actions,
            stop_actions,
            active_seen,
        )
        self._extend_unique_actions(
            active_actions,
            initial_switches,
            active_seen,
        )

        if not ranked_actions:
            node.is_expanded = True
            return

        action_scores: dict[int, float] = {}

        for action in ranked_actions:
            action_id = int(action.action_id)

            if neural_policy is not None:
                action_scores[action_id] = max(
                    float(neural_policy[action_id]),
                    1e-8,
                )
            elif action.kind == "stop":
                action_scores[action_id] = float(self.config.stop_prior)
            else:
                loading = self._loading_priority(
                    state,
                    action,
                )

                base_score = float(self.config.min_switch_prior_score)

                if loading is not None:
                    base_score = max(
                        float(loading) - 80.0,
                        base_score,
                    )

                action_scores[action_id] = base_score**self.config.prior_exponent

        node.ranked_actions = ranked_actions
        node.action_scores = action_scores
        node.selection_scores = dict(action_scores)

        exploration_actions = self._choose_exploration_actions(
            ranked_switches=ranked_switches,
            initial_switches=initial_switches,
        )

        node.forced_exploration_action_ids = [
            int(action.action_id) for action in exploration_actions
        ]

        self._set_active_actions(
            node,
            [
                *active_actions,
                *exploration_actions,
            ],
        )

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
                self.config.c_puct * prior * sqrt_parent_visits / (1 + child_visits)
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
        if not step_result.power_flow_success:
            return None
        return MCTSNode(
            env=child_env,
            depth=parent.depth + 1,
            prior=parent.action_priors.get(action_id, 0.0),
            action_from_parent=action,
            reward_from_parent=float(step_result.reward),
            step_result_from_parent=step_result,
        )

    def _backup(self, path: list[MCTSNode], leaf_value: float) -> None:
        """Back up terminal utility without temporal discount."""

        value = require_bounded_utility(
            leaf_value,
            context="MCTS leaf utility",
        )

        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value

    def _leaf_value(self, node: MCTSNode) -> float:
        """Evaluate a leaf under the shared terminal-utility contract."""

        if node.done:
            terminal_utility, _ = terminal_utility_from_outcome(
                node.solved,
                getattr(node.env, "termination_reason", None),
                evidence=getattr(node.env, "terminal_outcome_evidence", None),
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
            action_mask = node.env.operational_action_mask()
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


_TEMPERATURE_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """Behavior policy and the action sampled from exactly that policy."""

    policy: dict[int, float]
    action_id: int


def normalize_policy(
    policy: Mapping[int, Real],
    *,
    context: str = "policy",
) -> dict[int, float]:
    """Validate a policy and normalize its positive probability mass."""
    if not policy:
        raise ValueError(f"{context} must not be empty.")

    positive_mass: dict[int, float] = {}
    total = 0.0

    for raw_action_id, raw_probability in policy.items():
        if (
            isinstance(raw_action_id, bool)
            or not isinstance(raw_action_id, Integral)
            or int(raw_action_id) < 0
        ):
            raise ValueError(f"{context} contains invalid action ID {raw_action_id!r}.")
        if isinstance(raw_probability, bool) or not isinstance(
            raw_probability,
            Real,
        ):
            raise TypeError(
                f"{context} probability for action {raw_action_id!r} must be numeric."
            )

        probability = float(raw_probability)
        if not math.isfinite(probability):
            raise ValueError(
                f"{context} probability for action {raw_action_id!r} must be finite."
            )
        if probability < 0.0:
            raise ValueError(
                f"{context} probability for action {raw_action_id!r} "
                "must be non-negative."
            )
        if probability == 0.0:
            continue

        action_id = int(raw_action_id)
        positive_mass[action_id] = probability
        total += probability

    if total <= 0.0:
        raise ValueError(f"{context} must contain positive probability mass.")

    return {
        action_id: probability / total
        for action_id, probability in positive_mass.items()
    }


def constrain_policy(
    policy: Mapping[int, Real],
    allowed_action_ids: Iterable[int],
    *,
    context: str = "policy",
) -> dict[int, float]:
    """Restrict a policy to allowed actions and renormalize remaining mass."""
    normalized = normalize_policy(policy, context=context)
    allowed = _validated_action_ids(allowed_action_ids, context=context)
    constrained = {
        action_id: probability
        for action_id, probability in normalized.items()
        if action_id in allowed
    }
    if not constrained:
        return {}
    return normalize_policy(constrained, context=f"constrained {context}")


def policy_at_temperature(
    policy: Mapping[int, Real],
    temperature: float,
    *,
    context: str = "policy",
) -> dict[int, float]:
    """Return the exact behavior policy induced by ``temperature``."""
    normalized = normalize_policy(policy, context=context)
    temperature = _validated_temperature(temperature)

    if temperature <= _TEMPERATURE_EPS:
        best_action_id = max(normalized, key=normalized.__getitem__)
        return {best_action_id: 1.0}

    action_ids = tuple(normalized)
    probabilities = np.fromiter(normalized.values(), dtype=np.float64)
    log_weights = np.log(probabilities) / temperature
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)

    return {
        action_id: float(weight)
        for action_id, weight in zip(action_ids, weights, strict=True)
    }


def select_policy_action(
    policy: Mapping[int, Real],
    temperature: float,
    rng: np.random.Generator,
    *,
    context: str = "policy",
) -> PolicySelection:
    """Build the behavior policy and sample one action from its support."""
    behavior_policy = policy_at_temperature(
        policy,
        temperature,
        context=context,
    )

    if len(behavior_policy) == 1:
        action_id = next(iter(behavior_policy))
    else:
        action_ids = np.fromiter(behavior_policy, dtype=np.int64)
        probabilities = np.fromiter(
            behavior_policy.values(),
            dtype=np.float64,
        )
        action_id = int(rng.choice(action_ids, p=probabilities))

    return PolicySelection(
        policy=behavior_policy,
        action_id=int(action_id),
    )


def select_action_from_policy(
    policy: Mapping[int, Real],
    temperature: float,
    rng: np.random.Generator,
    *,
    context: str = "policy",
) -> int:
    """Select one action from the temperature-adjusted behavior policy."""
    return select_policy_action(
        policy,
        temperature,
        rng,
        context=context,
    ).action_id


def require_action_in_policy_support(
    action_id: int,
    policy: Mapping[int, Real],
    *,
    context: str = "policy",
) -> None:
    """Reject execution of an action outside positive policy support."""
    normalized = normalize_policy(policy, context=context)
    if (
        isinstance(action_id, bool)
        or not isinstance(action_id, Integral)
        or int(action_id) not in normalized
    ):
        raise ValueError(f"Action {action_id!r} is outside the support of {context}.")


def _validated_action_ids(
    action_ids: Iterable[int],
    *,
    context: str,
) -> set[int]:
    validated: set[int] = set()
    for raw_action_id in action_ids:
        if (
            isinstance(raw_action_id, bool)
            or not isinstance(raw_action_id, Integral)
            or int(raw_action_id) < 0
        ):
            raise ValueError(
                f"Allowed actions for {context} contain invalid action ID "
                f"{raw_action_id!r}."
            )
        validated.add(int(raw_action_id))
    return validated


def _validated_temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("temperature must be numeric.")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative.")
    return temperature


@dataclass(frozen=True)
class BranchContinuation:
    action_id: int
    branch_id: int | None
    visits: int
    policy: float
    immediate_reward: float
    best_penalty: float
    improvement: float
    best_sequence_action_ids: list[int] = field(default_factory=list)
    best_sequence_branch_ids: list[int | None] = field(default_factory=list)
    best_sequence_rewards: list[float] = field(default_factory=list)
    best_final_metrics: dict[str, Any] = field(default_factory=dict)
    confidence_ok: bool = False
    allow: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ContinuationDecision:
    """Diagnostic continuation analysis; it does not execute an action."""

    allowed_action_ids: tuple[int, ...]
    recommended_action_id: int | None
    recommended_branch_id: int | None
    recommendation_reason: str
    root_penalty: float
    root_has_hard_overload: bool
    root_num_hard: int
    best_visit_action_id: int | None
    best_visit_branch_id: int | None
    best_allowed_action_id: int | None
    best_allowed_branch_id: int | None
    best_allowed_improvement: float
    best_improvement_action_id: int | None
    best_improvement_branch_id: int | None
    best_improvement: float
    branches: list[BranchContinuation]

    @property
    def selected_action_id(self) -> int:
        """Compatibility alias for legacy diagnostics and scripts."""
        return 0 if self.recommended_action_id is None else self.recommended_action_id

    @property
    def selected_branch_id(self) -> int | None:
        return self.recommended_branch_id

    @property
    def selected_reason(self) -> str:
        return self.recommendation_reason


def topology_penalty(
    state: GridFMState,
    depth: int = 0,
    switch_penalty: float = CONTINUATION_SWITCH_PENALTY,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """Return the shared physical penalty plus lookahead switching cost."""
    if int(depth) < 0:
        raise ValueError("depth must be non-negative.")
    if float(switch_penalty) < 0.0:
        raise ValueError("switch_penalty must be non-negative.")
    return float(
        state_security_penalty(
            state,
            physics_config=physics_config or DEFAULT_PHYSICS_CONFIG,
            weights=CONTINUATION_GRID_UTILITY_WEIGHTS,
        )
        + float(switch_penalty) * float(depth)
    )


def _node_penalty(node: MCTSNode, physics_config: PhysicsConfig) -> float:
    state = node.env.current_state
    if state is None:
        return 1e9
    return topology_penalty(
        state=state,
        depth=node.depth,
        physics_config=physics_config,
    )


def _node_metrics(node: MCTSNode) -> dict[str, Any]:
    state = node.env.current_state
    return {} if state is None else dict(state.metrics)


def _best_reachable_state_from_subtree(
    start_node: MCTSNode,
    first_action_id: int,
    physics_config: PhysicsConfig,
) -> tuple[float, list[int], list[int | None], list[float], dict[str, Any]]:
    best_penalty = _node_penalty(start_node, physics_config)
    best_actions = [int(first_action_id)]
    best_branches = [start_node.branch_id_from_parent]
    best_rewards = [float(start_node.reward_from_parent)]
    best_metrics = _node_metrics(start_node)
    stack: list[
        tuple[
            MCTSNode,
            tuple[int, ...],
            tuple[int | None, ...],
            tuple[float, ...],
        ]
    ] = [
        (
            start_node,
            (int(first_action_id),),
            (start_node.branch_id_from_parent,),
            (float(start_node.reward_from_parent),),
        )
    ]

    while stack:
        node, action_path, branch_path, reward_path = stack.pop()
        penalty = _node_penalty(node, physics_config)
        if penalty < best_penalty:
            best_penalty = penalty
            best_actions = list(action_path)
            best_branches = list(branch_path)
            best_rewards = list(reward_path)
            best_metrics = _node_metrics(node)
        for child_action_id, child in node.children.items():
            stack.append(
                (
                    child,
                    (*action_path, int(child_action_id)),
                    (*branch_path, child.branch_id_from_parent),
                    (*reward_path, float(child.reward_from_parent)),
                )
            )

    return (
        float(best_penalty),
        best_actions,
        best_branches,
        best_rewards,
        best_metrics,
    )


def _is_branch_allowed(
    *,
    root_num_hard: int,
    root_has_hard: bool,
    best_final_metrics: dict[str, Any],
    improvement: float,
    confidence_ok: bool,
    min_hard_improvement: float,
    min_soft_improvement: float,
) -> tuple[bool, str]:
    final_hard = int(best_final_metrics.get("num_hard_overloaded_branches", 999))
    final_max = float(best_final_metrics.get("max_loading_percent", 999.0))

    if not confidence_ok:
        return False, "low_mcts_confidence"
    if root_has_hard:
        if final_hard >= root_num_hard:
            return False, (
                f"hard_not_reduced final_hard={final_hard}, root_hard={root_num_hard}"
            )
        if improvement < min_hard_improvement:
            return False, (
                f"hard_improvement_too_small {improvement:.3f} "
                f"< {min_hard_improvement:.3f}"
            )
        return True, (
            f"hard_reduced final_hard={final_hard}, final_max={final_max:.3f}, "
            f"improvement={improvement:.3f}"
        )
    if final_hard > 0:
        return False, f"created_hard_overload final_hard={final_hard}"
    if improvement < min_soft_improvement:
        return False, (
            f"soft_improvement_too_small {improvement:.3f} < {min_soft_improvement:.3f}"
        )
    return True, (
        f"useful_soft_improvement final_max={final_max:.3f}, "
        f"improvement={improvement:.3f}"
    )


def analyze_root_branches(
    result: MCTSResult,
    min_hard_improvement: float = 50.0,
    min_soft_improvement: float = 15.0,
    min_visits: int = 5,
    min_visit_fraction: float = 0.01,
    physics_config: PhysicsConfig | None = None,
) -> ContinuationDecision:
    """Analyze which searched root actions satisfy the continuation heuristic."""
    config = physics_config or DEFAULT_PHYSICS_CONFIG
    root = result.root
    root_state = root.env.current_state
    if root_state is None:
        return _decision(
            result=result,
            branches=[],
            root_penalty=1e9,
            root_has_hard=True,
            root_num_hard=999,
            recommendation=None,
            reason="no_root_state",
        )

    root_penalty = topology_penalty(root_state, physics_config=config)
    root_num_hard = int(root_state.metrics.get("num_hard_overloaded_branches", 0))
    root_has_hard = root_num_hard > 0
    total_visits = max(sum(child.visit_count for child in root.children.values()), 1)
    branches: list[BranchContinuation] = []

    for action_id, child in root.children.items():
        best_penalty, seq_actions, seq_branches, seq_rewards, best_metrics = (
            _best_reachable_state_from_subtree(
                start_node=child,
                first_action_id=int(action_id),
                physics_config=config,
            )
        )
        visits = int(child.visit_count)
        policy = float(visits / total_visits)
        improvement = float(root_penalty - best_penalty)
        confidence_ok = visits >= int(min_visits) and policy >= float(
            min_visit_fraction
        )
        allow, reason = _is_branch_allowed(
            root_num_hard=root_num_hard,
            root_has_hard=root_has_hard,
            best_final_metrics=best_metrics,
            improvement=improvement,
            confidence_ok=confidence_ok,
            min_hard_improvement=min_hard_improvement,
            min_soft_improvement=min_soft_improvement,
        )
        branches.append(
            BranchContinuation(
                action_id=int(action_id),
                branch_id=child.branch_id_from_parent,
                visits=visits,
                policy=policy,
                immediate_reward=float(child.reward_from_parent),
                best_penalty=float(best_penalty),
                improvement=improvement,
                best_sequence_action_ids=seq_actions,
                best_sequence_branch_ids=seq_branches,
                best_sequence_rewards=seq_rewards,
                best_final_metrics=best_metrics,
                confidence_ok=confidence_ok,
                allow=allow,
                reason=reason,
            )
        )

    allowed = [branch for branch in branches if branch.allow]
    if root_has_hard:
        allowed.sort(key=lambda item: (item.visits, item.improvement), reverse=True)
        recommendation_reason = "best_allowed_by_visits"
    else:
        allowed.sort(key=lambda item: (item.improvement, item.visits), reverse=True)
        recommendation_reason = "best_allowed_by_improvement"

    recommendation = allowed[0] if allowed else None
    return _decision(
        result=result,
        branches=branches,
        root_penalty=root_penalty,
        root_has_hard=root_has_hard,
        root_num_hard=root_num_hard,
        recommendation=recommendation,
        reason=(
            recommendation_reason
            if recommendation is not None
            else "no_allowed_continuation"
        ),
    )


def _decision(
    *,
    result: MCTSResult,
    branches: list[BranchContinuation],
    root_penalty: float,
    root_has_hard: bool,
    root_num_hard: int,
    recommendation: BranchContinuation | None,
    reason: str,
) -> ContinuationDecision:
    ordered = sorted(
        branches,
        key=lambda item: (item.allow, item.visits, item.improvement),
        reverse=True,
    )
    best_improvement = max(
        branches,
        key=lambda item: (item.improvement, item.visits),
        default=None,
    )
    return ContinuationDecision(
        allowed_action_ids=tuple(
            branch.action_id for branch in ordered if branch.allow
        ),
        recommended_action_id=(
            None if recommendation is None else int(recommendation.action_id)
        ),
        recommended_branch_id=(
            None if recommendation is None else recommendation.branch_id
        ),
        recommendation_reason=reason,
        root_penalty=float(root_penalty),
        root_has_hard_overload=bool(root_has_hard),
        root_num_hard=int(root_num_hard),
        best_visit_action_id=result.best_action_id,
        best_visit_branch_id=result.best_branch_id,
        best_allowed_action_id=(
            None if recommendation is None else recommendation.action_id
        ),
        best_allowed_branch_id=(
            None if recommendation is None else recommendation.branch_id
        ),
        best_allowed_improvement=(
            0.0 if recommendation is None else float(recommendation.improvement)
        ),
        best_improvement_action_id=(
            None if best_improvement is None else best_improvement.action_id
        ),
        best_improvement_branch_id=(
            None if best_improvement is None else best_improvement.branch_id
        ),
        best_improvement=(
            0.0 if best_improvement is None else float(best_improvement.improvement)
        ),
        branches=ordered,
    )
