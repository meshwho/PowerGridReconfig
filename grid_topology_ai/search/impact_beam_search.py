from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.termination import TerminationReason

from tqdm import tqdm
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
    """
    Emergency-oriented safety score.

    Lower is better.

    Design goal:
    - do not blindly switch off the most overloaded line;
    - do not create a single catastrophic 200%+ overload;
    - do not reduce max loading by spreading hard overloads over many branches;
    - prefer actions that reduce the total emergency severity.

    Components:
    - squared hard overload: strongly punishes catastrophic peaks;
    - number of hard-overloaded branches: prevents spreading emergency overloads;
    - total hard overload: measures emergency severity;
    - max hard excess: keeps peak loading under control;
    - total normal overload: secondary soft overload term;
    - number of overloaded branches: discourages spreading violations;
    - voltage violations: keeps voltage safety in the score.
    """

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    num_overloaded = int(state.metrics["num_overloaded_branches"])
    num_hard = int(state.metrics["num_hard_overloaded_branches"])

    hard_sq = squared_hard_overload(state, physics_config=config)
    hard_sum = total_hard_overload(state, physics_config=config)
    over_sum = total_overload(state, physics_config=config)

    max_hard = max_hard_excess(state, physics_config=config)
    max_over = max_overload_excess(state, physics_config=config)

    voltage_violation = float(state.metrics.get("total_voltage_violation", 0.0))

    score = (
        3.0 * hard_sq
        + 1500.0 * float(num_hard)
        + 50.0 * hard_sum
        + 30.0 * max_hard
        + 5.0 * over_sum
        + 100.0 * float(num_overloaded)
        + 2.0 * max_over
        + 5000.0 * voltage_violation
    )

    return float(score)


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

    switch_penalty:
        Small cost for each topology switching action.

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

    switch_penalty: float = 5.0
    failure_penalty: float = 1_000_000.0
    solved_bonus: float = 5000.0

    show_progress: bool = False
    progress_update_every: int = 1


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
    config: ImpactBeamSearchConfig
    evaluated_actions: int


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
        The planner then ranks actions by emergency safety improvement, not by
        current branch loading.

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

        self.loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")
        self.status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")

        self.evaluated_actions = 0
        self.root_num_hard_overloaded = 0

        self._progress_bar = None
        self._current_depth = 0

    def _estimated_progress_total(self) -> int | None:
        """
        Estimate the number of expensive action evaluations.

        For beam search:
        - depth 1 expands only the root node;
        - later depths expand up to beam_width nodes.

        Estimated total:
            candidate_pool_size * (1 + beam_width * (max_depth - 1))

        This matches the common case when every node has at least candidate_pool_size
        valid candidate actions.
        """

        if self.config.candidate_pool_size <= 0:
            return None

        if self.config.max_depth <= 0:
            return 0

        estimated_nodes = 1 + max(self.config.beam_width, 1) * max(
            self.config.max_depth - 1,
            0,
        )

        return int(self.config.candidate_pool_size * estimated_nodes)


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
            unit="pf",
            dynamic_ncols=True,
            leave=True,
        )

    def _update_progress(
            self,
            n: int = 1,
            postfix: dict | None = None,
    ) -> None:
        if self._progress_bar is None:
            return

        update_every = max(int(self.config.progress_update_every), 1)

        if self.evaluated_actions % update_every == 0:
            self._progress_bar.update(n)

            if postfix:
                self._progress_bar.set_postfix(postfix)

    def _set_progress_postfix(self, postfix: dict) -> None:
        if self._progress_bar is None:
            return

        self._progress_bar.set_postfix(postfix)

    def _close_progress(self) -> None:
        if self._progress_bar is None:
            return

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
            completed: list[ImpactBeamSearchNode] = []

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
                        completed.append(node)
                        candidates.append(node)
                        continue

                    expanded = self._expand_best_impact_actions(node)

                    if not expanded:
                        completed.append(node)
                        candidates.append(node)
                        continue

                    candidates.extend(expanded)

                    for child in expanded:
                        if child.done:
                            completed.append(child)

                if not candidates:
                    break

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

            all_final = completed + beam

            if not all_final:
                best_node = root
                final_beam = [root]
            else:
                final_beam = self._sort_nodes(all_final)
                best_node = final_beam[0]

            result = ImpactBeamSearchResult(
                scenario_id=int(scenario_id),
                best_node=best_node,
                final_beam=final_beam[: self.config.beam_width],
                config=self.config,
                evaluated_actions=int(self.evaluated_actions),
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
        Cheap candidate prefilter.

        Important:
            This function does not decide which action is good.
            It only decides which actions are worth expensive power-flow evaluation.

        Default behavior:
            - stop action is allowed only if no hard overload exists;
            - switch actions are prefiltered by current loading;
            - actual ranking happens only after power-flow simulation.
        """

        state = env.current_state

        if state is None:
            return []

        valid_actions = env.valid_actions()

        num_hard = int(state.metrics.get("num_hard_overloaded_branches", 0))

        stop_actions: list[GridFMAction] = []
        switch_actions: list[GridFMAction] = []

        for action in valid_actions:
            if action.action_type == "do_nothing":
                stop_actions.append(action)
            elif action.action_type == "switch_off_branch":
                switch_actions.append(action)

        switch_actions = sorted(
            switch_actions,
            key=lambda action: float(
                state.branch_features[action.branch_pos, self.loading_idx]
            ),
            reverse=True,
        )

        if self.config.candidate_pool_size > 0:
            switch_actions = switch_actions[: self.config.candidate_pool_size]

        selected: list[GridFMAction] = []

        if self.config.include_stop_action and num_hard == 0:
            selected.extend(stop_actions)

        selected.extend(switch_actions)

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
            n=1,
            postfix={
                "depth": self._current_depth,
                "evaluated": self.evaluated_actions,
            },
        )

        child_depth = node.depth + 1

        if not step_result.power_flow_success or step_result.next_state is None:
            impact_score = -float(self.config.failure_penalty)

            if action.action_type == "switch_off_branch":
                impact_score -= float(self.config.switch_penalty)

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

        if action.action_type == "switch_off_branch":
            impact_score -= float(self.config.switch_penalty)

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
        Sort nodes by emergency safety.

        Priority:
        1. solved states;
        2. avoid hard-overload count above initial state;
        3. lower safety score;
        4. higher discounted improvement;
        5. shorter sequence.

        The safety_score already combines:
        - squared hard overload;
        - number of hard overloaded branches;
        - total hard overload;
        - max loading excess;
        - total overload;
        - number of overloaded branches;
        - voltage violation.
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
