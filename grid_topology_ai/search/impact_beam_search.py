from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, TypeVar

import numpy as np

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.physics.utility import state_security_penalty
from grid_topology_ai.termination import TerminationReason

from tqdm import tqdm


class TrajectoryNode(Protocol):
    action_ids: Sequence[int]
    branch_ids: Sequence[int | None]
    safety_score: float
    discounted_score: float
    num_hard_overloaded: int
    solved: bool


NodeT = TypeVar("NodeT", bound=TrajectoryNode)


@dataclass(frozen=True)
class TrajectorySelection:
    node: TrajectoryNode
    pareto_front: tuple[TrajectoryNode, ...]
    candidate_pool: tuple[TrajectoryNode, ...]
    best_physical_safety: float
    selected_safety: float
    selected_switch_count: int
    retained_improvement_fraction: float


def switch_count(node: TrajectoryNode) -> int:
    """Count physical switching actions; stop/handoff actions do not count."""

    return sum(branch_id is not None for branch_id in node.branch_ids)


def _action_key(node: TrajectoryNode) -> tuple[int, ...]:
    return tuple(int(action_id) for action_id in node.action_ids)


def _tie_break_key(node: TrajectoryNode) -> tuple[float, tuple[int, ...]]:
    return (-float(node.discounted_score), _action_key(node))


def _same_objectives(
    left: TrajectoryNode,
    right: TrajectoryNode,
    *,
    tolerance: float,
) -> bool:
    return (
        switch_count(left) == switch_count(right)
        and abs(float(left.safety_score) - float(right.safety_score)) <= tolerance
    )


def _dominates(
    left: TrajectoryNode,
    right: TrajectoryNode,
    *,
    tolerance: float,
) -> bool:
    left_safety = float(left.safety_score)
    right_safety = float(right.safety_score)
    left_switches = switch_count(left)
    right_switches = switch_count(right)

    no_worse = (
        left_safety <= right_safety + tolerance
        and left_switches <= right_switches
    )
    strictly_better = (
        left_safety < right_safety - tolerance
        or left_switches < right_switches
    )
    return no_worse and strictly_better


def pareto_front(
    nodes: Sequence[NodeT],
    *,
    max_hard_overloaded: int,
    tolerance: float = 1e-9,
) -> list[NodeT]:
    """Return nondominated AC-valid trajectories in (final penalty, switches)."""

    eligible = [
        node
        for node in nodes
        if float(node.safety_score) < float("inf")
        and int(node.num_hard_overloaded) <= int(max_hard_overloaded)
    ]

    unique: list[NodeT] = []
    for node in eligible:
        duplicate_index = next(
            (
                index
                for index, other in enumerate(unique)
                if _same_objectives(node, other, tolerance=tolerance)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(node)
            continue

        if _tie_break_key(node) < _tie_break_key(unique[duplicate_index]):
            unique[duplicate_index] = node

    front = [
        node
        for node in unique
        if not any(
            other is not node and _dominates(other, node, tolerance=tolerance)
            for other in unique
        )
    ]

    return sorted(
        front,
        key=lambda node: (
            switch_count(node),
            float(node.safety_score),
            *_tie_break_key(node),
        ),
    )


def update_pareto_archive(
    archive: Sequence[NodeT],
    nodes: Sequence[NodeT],
    *,
    max_hard_overloaded: int,
    tolerance: float = 1e-9,
) -> list[NodeT]:
    """Update a compact Pareto archive without retaining every searched env clone."""

    return pareto_front(
        [*archive, *nodes],
        max_hard_overloaded=max_hard_overloaded,
        tolerance=tolerance,
    )


def select_epsilon_optimal_trajectory(
    root: NodeT,
    nodes: Sequence[NodeT],
    *,
    relative_physical_epsilon: float,
    max_hard_overloaded: int,
    tolerance: float = 1e-9,
) -> TrajectorySelection:
    """
    Select the minimum-intervention trajectory among physically near-optimal ones.

    Strictly solved trajectories are handled lexicographically: choose the one
    requiring the fewest switching actions. If none is solved, keep trajectories
    within a relative epsilon of the best physical improvement found by the
    search, then choose the one with the fewest switches.
    """

    epsilon = float(relative_physical_epsilon)
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("relative_physical_epsilon must satisfy 0 <= epsilon < 1")

    front = pareto_front(
        nodes,
        max_hard_overloaded=max_hard_overloaded,
        tolerance=tolerance,
    )
    if not front:
        front = [root]

    solved = [node for node in front if bool(node.solved)]
    if solved:
        pool = sorted(
            solved,
            key=lambda node: (
                switch_count(node),
                float(node.safety_score),
                *_tie_break_key(node),
            ),
        )
        selected = pool[0]
    else:
        best_physical = min(float(node.safety_score) for node in front)
        root_safety = float(root.safety_score)
        available_improvement = max(root_safety - best_physical, 0.0)
        threshold = best_physical + epsilon * available_improvement

        pool = [
            node
            for node in front
            if float(node.safety_score) <= threshold + tolerance
        ]
        if not pool:
            pool = [min(front, key=lambda node: float(node.safety_score))]

        pool = sorted(
            pool,
            key=lambda node: (
                switch_count(node),
                float(node.safety_score),
                *_tie_break_key(node),
            ),
        )
        selected = pool[0]

    best_physical_safety = min(float(node.safety_score) for node in front)
    selected_safety = float(selected.safety_score)
    root_safety = float(root.safety_score)
    available_improvement = max(root_safety - best_physical_safety, 0.0)

    if available_improvement <= tolerance:
        retained_fraction = 1.0
    else:
        retained_fraction = (
            root_safety - selected_safety
        ) / available_improvement
        retained_fraction = min(max(retained_fraction, 0.0), 1.0)

    return TrajectorySelection(
        node=selected,
        pareto_front=tuple(front),
        candidate_pool=tuple(pool),
        best_physical_safety=best_physical_safety,
        selected_safety=selected_safety,
        selected_switch_count=switch_count(selected),
        retained_improvement_fraction=float(retained_fraction),
    )


# ======================================================================================
# Safety metrics
# ======================================================================================


def _active_loadings(state: GridFMState) -> np.ndarray:
    """
    Return loading_percent values only for active branches.
    """

    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    status = state.branch_features[:, status_idx]
    loading = state.branch_features[:, loading_idx]

    return loading[status > 0.0]


def total_overload(
    state: GridFMState,
    limit: float | None = None,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """
    Sum of overload above the normal limit.

    Example:
        loadings = [80, 105, 130]
        total_overload = 0 + 5 + 30 = 35
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.overload_limit_percent
        if limit is None
        else float(limit)
    )
    loading = _active_loadings(state)
    overload = np.where(
        loading > effective_limit + config.thermal_tolerance_percent,
        loading - effective_limit,
        0.0,
    )

    return float(np.sum(overload))


def total_hard_overload(
    state: GridFMState,
    hard_limit: float | None = None,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """
    Sum of overload above the hard emergency limit.
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent
        if hard_limit is None
        else float(hard_limit)
    )
    loading = _active_loadings(state)
    hard = np.where(
        loading > effective_limit + config.thermal_tolerance_percent,
        loading - effective_limit,
        0.0,
    )

    return float(np.sum(hard))


def squared_hard_overload(
    state: GridFMState,
    hard_limit: float | None = None,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """
    Squared hard overload.

    This strongly penalizes one catastrophic overloaded branch.

    Example:
        215% loading:
            hard excess = 215 - 120 = 95
            squared = 9025

        160% loading:
            hard excess = 160 - 120 = 40
            squared = 1600
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent
        if hard_limit is None
        else float(hard_limit)
    )
    loading = _active_loadings(state)
    hard = np.where(
        loading > effective_limit + config.thermal_tolerance_percent,
        loading - effective_limit,
        0.0,
    )

    return float(np.sum(hard * hard))


def max_hard_excess(
    state: GridFMState,
    hard_limit: float | None = None,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """
    Maximum excess above hard limit.
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent
        if hard_limit is None
        else float(hard_limit)
    )
    max_loading = float(state.metrics["max_loading_percent"])

    if max_loading <= effective_limit + config.thermal_tolerance_percent:
        return 0.0
    return float(max_loading - effective_limit)


def max_overload_excess(
    state: GridFMState,
    limit: float | None = None,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """
    Maximum excess above normal loading limit.
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.overload_limit_percent
        if limit is None
        else float(limit)
    )
    max_loading = float(state.metrics["max_loading_percent"])

    if max_loading <= effective_limit + config.thermal_tolerance_percent:
        return 0.0
    return float(max_loading - effective_limit)


def safety_score(
    state: GridFMState,
    physics_config: PhysicsConfig | None = None,
) -> float:
    """Return the canonical lower-is-better physical-state penalty."""

    return state_security_penalty(
        state,
        physics_config=physics_config,
    )


# ======================================================================================
# Data classes
# ======================================================================================


@dataclass(frozen=True)
class ImpactBeamSearchConfig:
    """
    Configuration for impact-aware beam search.

    This planner is intended as a physics-based teacher search.

    It differs from simple beam search:
    - simple beam search ranks candidate actions by current branch loading;
    - this planner applies candidate actions, runs power flow, and ranks by
      actual safety impact.

    Parameters
    ----------
    max_depth:
        Maximum number of topology switching steps.

    beam_width:
        Number of best partial trajectories kept after each depth.

    candidate_pool_size:
        Cheap prefilter size before expensive power-flow evaluation.
        If 0, all valid switch actions are evaluated.

    top_k_actions:
        Number of impact-tested children kept per node.

    gamma:
        Discount factor for cumulative impact score.

    include_stop_action:
        Whether do_nothing/stop may be considered when hard overload is gone.

    allow_hard_count_increase:
        If False, the planner filters out actions that increase the number of
        hard-overloaded branches whenever at least one non-worsening action exists.

    relative_physical_epsilon:
        Maximum fraction of the best discovered physical improvement that may be
        traded for a shorter switching sequence. Zero recovers exact physical
        minimization; 0.01 retains at least 99% of the discovered improvement.

    switch_penalty:
        Small search-score cost for each topology switching action.

    failure_penalty:
        Penalty for power-flow failure.

    solved_bonus:
        Bonus for fully removing all overloads.
    """

    max_depth: int = 4
    beam_width: int = 20
    candidate_pool_size: int = 120
    top_k_actions: int = 30
    gamma: float = 0.95

    include_stop_action: bool = True
    allow_hard_count_increase: bool = False
    relative_physical_epsilon: float = 0.01

    switch_penalty: float = 5.0
    failure_penalty: float = 1_000_000.0
    solved_bonus: float = 5000.0

    show_progress: bool = False
    progress_update_every: int = 1

    def __post_init__(self) -> None:
        epsilon = float(self.relative_physical_epsilon)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(
                "relative_physical_epsilon must satisfy 0 <= epsilon < 1"
            )


@dataclass
class ImpactBeamSearchNode:
    """
    One partial trajectory in impact-aware beam search.
    """

    env: TopologySwitchingEnv

    action_ids: list[int] = field(default_factory=list)
    branch_ids: list[int | None] = field(default_factory=list)

    rewards: list[float] = field(default_factory=list)
    impact_scores: list[float] = field(default_factory=list)

    cumulative_score: float = 0.0
    discounted_score: float = 0.0

    safety_score: float = 0.0

    max_loading_percent: float = 0.0
    num_overloaded: int = 0
    num_hard_overloaded: int = 0

    total_overload: float = 0.0
    total_hard_overload: float = 0.0
    squared_hard_overload: float = 0.0

    depth: int = 0
    done: bool = False
    solved: bool = False
    termination_reason: TerminationReason | None = None

    last_step_result: TopologyStepResult | None = None

    def short_sequence(self) -> str:
        parts: list[str] = []

        for branch_id in self.branch_ids:
            if branch_id is None:
                parts.append("stop")
            else:
                parts.append(str(branch_id))

        return " -> ".join(parts) if parts else "(root)"


@dataclass(frozen=True)
class ImpactBeamSearchResult:
    scenario_id: int
    best_node: ImpactBeamSearchNode
    final_beam: list[ImpactBeamSearchNode]
    pareto_front: list[ImpactBeamSearchNode]
    config: ImpactBeamSearchConfig
    evaluated_actions: int
    best_physical_safety: float
    selected_safety: float
    selected_switch_count: int
    retained_improvement_fraction: float


# ======================================================================================
# Planner
# ======================================================================================


class ImpactBeamSearchPlanner:
    """
    Impact-aware beam search planner.

    This is not the final AlphaZero agent.
    It is a reliable physics-based teacher used to bootstrap the neural policy.

    Main idea:
        At each state, candidate actions are actually simulated with power flow.
        The planner then ranks actions by canonical physical-state improvement,
        not by current branch loading.

    Safety guard:
        If hard overloads exist, the planner avoids actions that increase the
        number of hard-overloaded branches whenever possible.
    """

    def __init__(
        self,
        config: ImpactBeamSearchConfig,
        physics_config: PhysicsConfig | None = None,
    ):
        self.config = config
        self.physics_config = physics_config or DEFAULT_PHYSICS_CONFIG

        self.status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

        self.evaluated_actions = 0
        self.root_num_hard_overloaded = 0

        self._progress_bar = None
        self._current_depth = 0

    def _estimated_actions_per_node(self) -> int | None:
        """Estimate expensive evaluations for one expanded node."""

        candidate_limit = int(self.config.candidate_pool_size)
        screened_limit = getattr(self, "lodf_screen_top_k", None)

        if screened_limit is not None and int(screened_limit) > 0:
            if candidate_limit <= 0:
                candidate_limit = int(screened_limit)
            else:
                candidate_limit = min(candidate_limit, int(screened_limit))

        if candidate_limit <= 0:
            return None

        if self.config.include_stop_action:
            candidate_limit += 1

        return candidate_limit

    def _estimated_progress_total(self) -> int | None:
        """
        Estimate the number of expensive action evaluations.

        Depth 1 expands only the root node. Later depths expand up to beam_width
        nodes. Screened planners expose their pre-AC limit through
        lodf_screen_top_k, so the estimate follows the number of actions that can
        actually reach env.step() instead of the larger cheap candidate pool.
        """

        actions_per_node = self._estimated_actions_per_node()

        if actions_per_node is None:
            return None

        if self.config.max_depth <= 0:
            return 0

        estimated_nodes = 1 + max(self.config.beam_width, 1) * max(
            self.config.max_depth - 1,
            0,
        )

        return int(actions_per_node * estimated_nodes)

    def _start_progress(self) -> None:
        if not self.config.show_progress:
            return

        if tqdm is None:
            print(
                "Progress bar requested, but tqdm is not installed. "
                "Install it with: python -m pip install tqdm"
            )
            return

        self._progress_bar = tqdm(
            total=self._estimated_progress_total(),
            desc="Impact beam search",
            unit="eval",
            dynamic_ncols=True,
            leave=True,
        )

    def _update_progress(
        self,
        postfix: dict | None = None,
    ) -> None:
        if self._progress_bar is None:
            return

        update_every = max(int(self.config.progress_update_every), 1)

        if self.evaluated_actions % update_every != 0:
            return

        delta = int(self.evaluated_actions) - int(self._progress_bar.n)

        if delta > 0:
            self._progress_bar.update(delta)

        if postfix:
            self._progress_bar.set_postfix(postfix)

    def _set_progress_postfix(self, postfix: dict) -> None:
        if self._progress_bar is None:
            return

        self._progress_bar.set_postfix(postfix)

    def _close_progress(self) -> None:
        if self._progress_bar is None:
            return

        delta = int(self.evaluated_actions) - int(self._progress_bar.n)

        if delta > 0:
            self._progress_bar.update(delta)

        if self._progress_bar.total is not None:
            self._progress_bar.total = int(self.evaluated_actions)
            self._progress_bar.refresh()

        self._progress_bar.close()
        self._progress_bar = None

    def search(
            self,
            env: TopologySwitchingEnv,
            scenario_id: int,
    ) -> ImpactBeamSearchResult:
        self.evaluated_actions = 0
        self._start_progress()

        try:
            root_env = env.clone()
            root_env.reset(int(scenario_id))

            root_state = root_env.current_state

            if root_state is None:
                raise RuntimeError("Environment reset returned no state.")

            self.root_num_hard_overloaded = int(
                root_state.metrics["num_hard_overloaded_branches"]
            )

            root = self._make_node_from_state(
                env=root_env,
                depth=0,
                action_ids=[],
                branch_ids=[],
                rewards=[],
                impact_scores=[],
                cumulative_score=0.0,
                discounted_score=0.0,
                done=bool(root_env.done),
                solved=bool(root_env.solved),
                termination_reason=root_env.termination_reason,
                last_step_result=None,
            )

            beam: list[ImpactBeamSearchNode] = [root]
            pareto_archive: list[ImpactBeamSearchNode] = [root]

            for _depth in range(self.config.max_depth):
                self._current_depth = int(_depth) + 1

                candidates: list[ImpactBeamSearchNode] = []

                self._set_progress_postfix(
                    {
                        "depth": self._current_depth,
                        "beam": len(beam),
                        "evaluated": self.evaluated_actions,
                    }
                )

                for node in beam:
                    if node.done:
                        candidates.append(node)
                        continue

                    expanded = self._expand_best_impact_actions(node)

                    if not expanded:
                        candidates.append(node)
                        continue

                    candidates.extend(expanded)

                if not candidates:
                    break

                pareto_archive = update_pareto_archive(
                    pareto_archive,
                    candidates,
                    max_hard_overloaded=self.root_num_hard_overloaded,
                )

                candidates = self._sort_nodes(candidates)
                beam = candidates[: self.config.beam_width]

                if beam:
                    self._set_progress_postfix(
                        {
                            "depth": self._current_depth,
                            "beam": len(beam),
                            "best_safety": f"{beam[0].safety_score:.1f}",
                            "hard": beam[0].num_hard_overloaded,
                            "max": f"{beam[0].max_loading_percent:.1f}%",
                        }
                    )

                if beam and beam[0].solved:
                    break

            selection = select_epsilon_optimal_trajectory(
                root,
                pareto_archive,
                relative_physical_epsilon=self.config.relative_physical_epsilon,
                max_hard_overloaded=self.root_num_hard_overloaded,
            )

            best_node = selection.node
            final_beam = list(selection.candidate_pool)
            if not final_beam:
                final_beam = [best_node]

            result = ImpactBeamSearchResult(
                scenario_id=int(scenario_id),
                best_node=best_node,
                final_beam=final_beam[: self.config.beam_width],
                pareto_front=list(selection.pareto_front),
                config=self.config,
                evaluated_actions=int(self.evaluated_actions),
                best_physical_safety=float(selection.best_physical_safety),
                selected_safety=float(selection.selected_safety),
                selected_switch_count=int(selection.selected_switch_count),
                retained_improvement_fraction=float(
                    selection.retained_improvement_fraction
                ),
            )

            return result

        finally:
            self._close_progress()

    # ----------------------------------------------------------------------------------
    # Node construction
    # ----------------------------------------------------------------------------------

    def _make_node_from_state(
        self,
        env: TopologySwitchingEnv,
        depth: int,
        action_ids: list[int],
        branch_ids: list[int | None],
        rewards: list[float],
        impact_scores: list[float],
        cumulative_score: float,
        discounted_score: float,
        done: bool,
        solved: bool,
        termination_reason: TerminationReason | None,
        last_step_result: TopologyStepResult | None,
    ) -> ImpactBeamSearchNode:
        state = env.current_state

        if state is None:
            raise RuntimeError("Cannot create node without current state.")

        return ImpactBeamSearchNode(
            env=env,
            action_ids=action_ids,
            branch_ids=branch_ids,
            rewards=rewards,
            impact_scores=impact_scores,
            cumulative_score=float(cumulative_score),
            discounted_score=float(discounted_score),
            safety_score=safety_score(
                state,
                physics_config=self.physics_config,
            ),
            max_loading_percent=float(state.metrics["max_loading_percent"]),
            num_overloaded=int(state.metrics["num_overloaded_branches"]),
            num_hard_overloaded=int(state.metrics["num_hard_overloaded_branches"]),
            total_overload=total_overload(
                state,
                physics_config=self.physics_config,
            ),
            total_hard_overload=total_hard_overload(
                state,
                physics_config=self.physics_config,
            ),
            squared_hard_overload=squared_hard_overload(
                state,
                physics_config=self.physics_config,
            ),
            depth=int(depth),
            done=bool(done),
            solved=bool(solved),
            termination_reason=termination_reason,
            last_step_result=last_step_result,
        )

    # ----------------------------------------------------------------------------------
    # Expansion
    # ----------------------------------------------------------------------------------

    def _expand_best_impact_actions(
        self,
        node: ImpactBeamSearchNode,
    ) -> list[ImpactBeamSearchNode]:
        candidate_actions = self._candidate_actions(node.env)

        if not candidate_actions:
            return []

        evaluated_children: list[ImpactBeamSearchNode] = []

        for action in candidate_actions:
            child = self._expand_node(node, action)

            if child is None:
                continue

            evaluated_children.append(child)

        if not evaluated_children:
            return []

        evaluated_children = self._apply_safety_guards(
            parent=node,
            children=evaluated_children,
        )

        evaluated_children = self._sort_nodes(evaluated_children)

        if self.config.top_k_actions > 0:
            evaluated_children = evaluated_children[: self.config.top_k_actions]

        return evaluated_children

    def _candidate_actions(
            self,
            env: TopologySwitchingEnv,
    ) -> list[GridFMAction]:
        """
        Build the cheap candidate pool.

        Loading-based prefiltering applies only to branch
        openings. Other topology actions remain available for
        the actual power-flow impact evaluation.
        """
        state = env.current_state

        if state is None:
            return []

        valid_actions = env.valid_actions()
        num_hard = int(
            state.metrics.get(
                "num_hard_overloaded_branches",
                0,
            )
        )

        stop_actions = [
            action
            for action in valid_actions
            if action.kind == "stop"
        ]
        topology_actions = [
            action
            for action in valid_actions
            if action.kind != "stop"
        ]

        loading_priorities: dict[int, float] = {}
        always_keep: list[GridFMAction] = []

        for action in topology_actions:
            loading = env.action_space.loading_priority(
                state,
                action,
            )

            if loading is None:
                always_keep.append(action)
                continue

            loading_priorities[
                int(action.action_id)
            ] = float(loading)

        loading_actions = sorted(
            [
                action
                for action in topology_actions
                if int(action.action_id)
                   in loading_priorities
            ],
            key=lambda action: loading_priorities[
                int(action.action_id)
            ],
            reverse=True,
        )

        always_keep.sort(
            key=lambda action: int(action.action_id)
        )

        if self.config.candidate_pool_size > 0:
            loading_actions = loading_actions[
                              : self.config.candidate_pool_size
                              ]

        selected: list[GridFMAction] = []

        if (
                self.config.include_stop_action
                and num_hard == 0
        ):
            selected.extend(stop_actions)

        selected.extend(loading_actions)
        selected.extend(always_keep)

        return selected

    def _expand_node(
        self,
        node: ImpactBeamSearchNode,
        action: GridFMAction,
    ) -> ImpactBeamSearchNode | None:
        child_env = node.env.clone()

        before_state = child_env.current_state

        if before_state is None:
            return None

        before_safety = safety_score(
            before_state,
            physics_config=self.physics_config,
        )

        try:
            step_result = child_env.step(action.action_id)
        except Exception:
            return None

        self.evaluated_actions += 1

        self._update_progress(
            postfix={
                "depth": self._current_depth,
                "evaluated": self.evaluated_actions,
            },
        )

        child_depth = node.depth + 1

        if not step_result.power_flow_success or step_result.next_state is None:
            impact_score = -float(self.config.failure_penalty)

            if action.kind != "stop":
                impact_score -= float(
                    self.config.switch_penalty
                )

            discounted = (float(self.config.gamma) ** node.depth) * impact_score

            return ImpactBeamSearchNode(
                env=child_env,
                action_ids=[*node.action_ids, int(action.action_id)],
                branch_ids=[*node.branch_ids, action.branch_id],
                rewards=[*node.rewards, float(step_result.reward)],
                impact_scores=[*node.impact_scores, float(impact_score)],
                cumulative_score=float(node.cumulative_score + impact_score),
                discounted_score=float(node.discounted_score + discounted),
                safety_score=float("inf"),
                max_loading_percent=float("inf"),
                num_overloaded=10**9,
                num_hard_overloaded=10**9,
                total_overload=float("inf"),
                total_hard_overload=float("inf"),
                squared_hard_overload=float("inf"),
                depth=child_depth,
                done=True,
                solved=False,
                termination_reason=TerminationReason.POWER_FLOW_FAILED,
                last_step_result=step_result,
            )

        after_state = step_result.next_state
        after_safety = safety_score(
            after_state,
            physics_config=self.physics_config,
        )

        impact_score = float(before_safety - after_safety)

        if action.kind != "stop":
            impact_score -= float(
                self.config.switch_penalty
            )

        if bool(step_result.solved):
            impact_score += float(self.config.solved_bonus)

        before_hard = int(before_state.metrics["num_hard_overloaded_branches"])
        after_hard = int(after_state.metrics["num_hard_overloaded_branches"])

        # Strongly discourage increasing the number of hard-overloaded branches.
        if after_hard > before_hard:
            impact_score -= 500.0 * float(after_hard - before_hard)

        # Mild bonus for reducing hard-overload count.
        if after_hard < before_hard:
            impact_score += 50.0 * float(before_hard - after_hard)

        discounted = (float(self.config.gamma) ** node.depth) * impact_score

        return self._make_node_from_state(
            env=child_env,
            depth=child_depth,
            action_ids=[*node.action_ids, int(action.action_id)],
            branch_ids=[*node.branch_ids, action.branch_id],
            rewards=[*node.rewards, float(step_result.reward)],
            impact_scores=[*node.impact_scores, float(impact_score)],
            cumulative_score=float(node.cumulative_score + impact_score),
            discounted_score=float(node.discounted_score + discounted),
            done=bool(step_result.done),
            solved=bool(step_result.solved),
            termination_reason=step_result.info.get("termination_reason"),
            last_step_result=step_result,
        )

    # ----------------------------------------------------------------------------------
    # Safety guards and sorting
    # ----------------------------------------------------------------------------------

    def _apply_safety_guards(
        self,
        parent: ImpactBeamSearchNode,
        children: list[ImpactBeamSearchNode],
    ) -> list[ImpactBeamSearchNode]:
        """
        Apply hard safety guards before sorting/pruning.

        Rule 1:
            If at least one child does not increase the hard-overload count
            compared with the parent, discard children that do increase it.

        Rule 2:
            If the parent is already at or below the initial hard-overload count,
            and at least one child remains at or below the initial count, discard
            children that exceed the initial count.

        These rules prevent the teacher from learning:
            "reduce max loading by spreading hard overloads over more branches".
        """

        if self.config.allow_hard_count_increase:
            return children

        if not children:
            return children

        non_worsening = [
            child
            for child in children
            if child.num_hard_overloaded <= parent.num_hard_overloaded
        ]

        if non_worsening:
            children = non_worsening

        within_initial_limit = [
            child
            for child in children
            if child.num_hard_overloaded <= self.root_num_hard_overloaded
        ]

        if within_initial_limit:
            children = within_initial_limit

        return children

    def _sort_nodes(
        self,
        nodes: list[ImpactBeamSearchNode],
    ) -> list[ImpactBeamSearchNode]:
        """
        Sort nodes by canonical final physical-state quality during exploration.

        Priority:
        1. solved states;
        2. avoid hard-overload count above initial state;
        3. lower canonical physical-state penalty;
        4. higher discounted improvement;
        5. shorter sequence.

        Final teacher selection is performed separately from the Pareto archive,
        so beam exploration remains focused on physical quality and can pass
        through locally worse intermediate states.
        """

        return sorted(
            nodes,
            key=lambda node: (
                int(node.solved),
                -int(
                    max(
                        int(node.num_hard_overloaded)
                        - int(self.root_num_hard_overloaded),
                        0,
                    )
                ),
                -float(node.safety_score),
                float(node.discounted_score),
                -int(node.depth),
            ),
            reverse=True,
        )


class LODFScreenedImpactBeamSearchPlanner(ImpactBeamSearchPlanner):
    """Impact beam search with LODF pre-screening before authoritative AC PF."""

    def __init__(
        self,
        config: ImpactBeamSearchConfig,
        lodf_screen_top_k: int,
        lodf_min_candidate_count: int = 1,
        physics_config: PhysicsConfig | None = None,
        lodf_structure_cache=None,
    ) -> None:
        super().__init__(config, physics_config=physics_config)
        self.lodf_screen_top_k = int(lodf_screen_top_k)
        self.lodf_min_candidate_count = int(lodf_min_candidate_count)
        self.lodf_structure_cache = lodf_structure_cache

    def _candidate_actions(self, env: TopologySwitchingEnv) -> list[GridFMAction]:
        from grid_topology_ai.search.screening import rank_actions_by_lodf_screening

        base_actions = super()._candidate_actions(env)
        if self.lodf_screen_top_k <= 0 or env.current_state is None:
            return base_actions

        stop_actions = [
            action for action in base_actions if action.action_type == "do_nothing"
        ]
        switch_actions = [
            action
            for action in base_actions
            if action.action_type == "switch_off_branch"
        ]
        if (
            len(switch_actions) < self.lodf_min_candidate_count
            or len(switch_actions) <= self.lodf_screen_top_k
        ):
            return base_actions

        try:
            ranked = rank_actions_by_lodf_screening(
                state=env.current_state,
                actions=switch_actions,
                physics_config=self.physics_config,
                structure_cache=self.lodf_structure_cache,
            )
        except Exception:
            ranked = switch_actions
        return [*stop_actions, *ranked[: self.lodf_screen_top_k]]
