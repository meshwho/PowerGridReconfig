from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeVar

import numpy as np
from tqdm import tqdm

from grid_topology_ai.actions import GridFMAction
from grid_topology_ai.config import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.environment import TopologyStepResult, TopologySwitchingEnv
from grid_topology_ai.physics.objective import assess_physical_state
from grid_topology_ai.physics.redispatch import (
    MinimalRedispatchResult,
    run_minimal_ac_redispatch,
)
from grid_topology_ai.physics.utility import state_security_penalty
from grid_topology_ai.self_play.artifacts import file_content_identity
from grid_topology_ai.state import BRANCH_FEATURE_COLUMNS, GridFMState
from grid_topology_ai.termination import TerminationReason


class TrajectoryNode(Protocol):
    action_ids: Sequence[int]
    branch_ids: Sequence[int | None]
    safety_score: float
    discounted_score: float
    num_hard_overloaded: int
    solved: bool


NodeT = TypeVar("NodeT", bound=TrajectoryNode)


def switch_count(node: TrajectoryNode) -> int:
    """Count physical switching actions; stop/handoff actions do not count."""

    return sum(branch_id is not None for branch_id in node.branch_ids)


def _action_key(node: TrajectoryNode) -> tuple[int, ...]:
    return tuple(int(action_id) for action_id in node.action_ids)


def _tie_break_key(node: TrajectoryNode) -> tuple[float, tuple[int, ...]]:
    return (-float(node.discounted_score), _action_key(node))


def _topology_candidate_key(node: TrajectoryNode) -> tuple[object, ...]:
    return (
        float(node.safety_score),
        switch_count(node),
        *_tie_break_key(node),
    )


def update_top_j_candidate_archive(
    archive: Mapping[int, Sequence[NodeT]],
    nodes: Sequence[NodeT],
    *,
    per_switch_count: int,
) -> dict[int, list[NodeT]]:
    """Keep the lowest-J AC-valid trajectories independently for each switch count."""

    limit = int(per_switch_count)
    if limit <= 0:
        raise ValueError("per_switch_count must be positive")

    updated = {
        int(num_switches): list(bucket)
        for num_switches, bucket in archive.items()
    }

    for node in nodes:
        if not math.isfinite(float(node.safety_score)):
            continue

        num_switches = switch_count(node)
        bucket = updated.setdefault(num_switches, [])
        action_key = _action_key(node)
        duplicate_index = next(
            (
                index
                for index, other in enumerate(bucket)
                if _action_key(other) == action_key
            ),
            None,
        )
        node_key = _topology_candidate_key(node)

        if duplicate_index is None:
            bucket.append(node)
        else:
            other = bucket[duplicate_index]
            if node_key < _topology_candidate_key(other):
                bucket[duplicate_index] = node

        bucket.sort(key=_topology_candidate_key)
        del bucket[limit:]

    return updated


# ======================================================================================
# Safety metrics
# ======================================================================================


def _active_loadings(state: GridFMState) -> np.ndarray:
    """Return loading_percent values only for active branches."""

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
    """Sum of overload above the normal limit."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = config.overload_limit_percent if limit is None else float(limit)
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
    """Sum of overload above the hard emergency limit."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent if hard_limit is None else float(hard_limit)
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
    """Squared hard overload."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent if hard_limit is None else float(hard_limit)
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
    """Maximum excess above hard limit."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = (
        config.hard_overload_limit_percent if hard_limit is None else float(hard_limit)
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
    """Maximum excess above normal loading limit."""

    config = physics_config or DEFAULT_PHYSICS_CONFIG
    effective_limit = config.overload_limit_percent if limit is None else float(limit)
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
    """Configuration for impact-aware beam search."""

    max_depth: int = 4
    beam_width: int = 20
    candidate_pool_size: int = 120
    top_k_actions: int = 30
    redispatch_candidates_per_switch_count: int = 5
    gamma: float = 0.95

    allow_hard_count_increase: bool = False

    switch_penalty: float = 5.0
    failure_penalty: float = 1_000_000.0
    solved_bonus: float = 5000.0

    show_progress: bool = False
    progress_update_every: int = 1

    def __post_init__(self) -> None:
        if int(self.redispatch_candidates_per_switch_count) <= 0:
            raise ValueError("redispatch_candidates_per_switch_count must be positive")


@dataclass
class ImpactBeamSearchNode:
    """One partial trajectory in impact-aware beam search."""

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
    redispatch_candidates: list[ImpactBeamSearchNode] = field(default_factory=list)


# ======================================================================================
# Planner
# ======================================================================================


class ImpactBeamSearchPlanner:
    """Impact-aware beam search planner."""

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
        self._redispatch_candidate_archive: dict[
            int, list[ImpactBeamSearchNode]
        ] = {}

        self._progress_bar = None
        self._current_depth = 0

    def _estimated_actions_per_node(self) -> int | None:
        candidate_limit = int(self.config.candidate_pool_size)
        screened_limit = getattr(self, "lodf_screen_top_k", None)

        if screened_limit is not None and int(screened_limit) > 0:
            if candidate_limit <= 0:
                candidate_limit = int(screened_limit)
            else:
                candidate_limit = min(candidate_limit, int(screened_limit))

        if candidate_limit <= 0:
            return None

        return candidate_limit

    def _estimated_progress_total(self) -> int | None:
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
            self._redispatch_candidate_archive = update_top_j_candidate_archive(
                {},
                [root],
                per_switch_count=self.config.redispatch_candidates_per_switch_count,
            )

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

            redispatch_candidates = [
                node
                for num_switches in sorted(self._redispatch_candidate_archive)
                for node in self._redispatch_candidate_archive[num_switches]
            ]
            if not redispatch_candidates:
                redispatch_candidates = [root]

            ranked_candidates = sorted(redispatch_candidates, key=_topology_candidate_key)
            best_node = ranked_candidates[0]
            final_beam = ranked_candidates[: self.config.beam_width]

            return ImpactBeamSearchResult(
                scenario_id=int(scenario_id),
                best_node=best_node,
                final_beam=final_beam,
                config=self.config,
                evaluated_actions=int(self.evaluated_actions),
                redispatch_candidates=redispatch_candidates,
            )

        finally:
            self._close_progress()

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

        self._redispatch_candidate_archive = update_top_j_candidate_archive(
            self._redispatch_candidate_archive,
            evaluated_children,
            per_switch_count=self.config.redispatch_candidates_per_switch_count,
        )

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
        state = env.current_state

        if state is None:
            return []

        topology_actions = [
            action for action in env.valid_actions() if action.kind != "stop"
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

            loading_priorities[int(action.action_id)] = float(loading)

        loading_actions = sorted(
            [
                action
                for action in topology_actions
                if int(action.action_id) in loading_priorities
            ],
            key=lambda action: loading_priorities[int(action.action_id)],
            reverse=True,
        )

        always_keep.sort(key=lambda action: int(action.action_id))

        if self.config.candidate_pool_size > 0:
            loading_actions = loading_actions[: self.config.candidate_pool_size]

        return [*loading_actions, *always_keep]

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

        if action.kind != "stop":
            impact_score -= float(self.config.switch_penalty)

        if bool(step_result.solved):
            impact_score += float(self.config.solved_bonus)

        before_hard = int(before_state.metrics["num_hard_overloaded_branches"])
        after_hard = int(after_state.metrics["num_hard_overloaded_branches"])

        if after_hard > before_hard:
            impact_score -= 500.0 * float(after_hard - before_hard)

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

    def _apply_safety_guards(
        self,
        parent: ImpactBeamSearchNode,
        children: list[ImpactBeamSearchNode],
    ) -> list[ImpactBeamSearchNode]:
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
        return ranked[: self.lodf_screen_top_k]


def make_one_hot_policy(action_id: int) -> dict[int, float]:
    return {int(action_id): 1.0}


def make_policy_from_final_beam(
    result: ImpactBeamSearchResult,
    temperature: float,
) -> tuple[dict[int, float], dict[int, int]]:
    best_node = result.best_node

    if not best_node.action_ids:
        return {}, {}

    best_action_id = int(best_node.action_ids[0])

    if temperature <= 1e-12:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    best_safety = float(best_node.safety_score)

    weights_by_action: dict[int, float] = {}
    counts_by_action: dict[int, int] = {}

    for node in result.final_beam:
        if not node.action_ids:
            continue

        action_id = int(node.action_ids[0])
        safety_gap = max(float(node.safety_score) - best_safety, 0.0)
        weight = float(np.exp(-safety_gap / float(temperature)))

        weights_by_action[action_id] = weights_by_action.get(action_id, 0.0) + weight
        counts_by_action[action_id] = counts_by_action.get(action_id, 0) + 1

    total = float(sum(weights_by_action.values()))

    if total <= 0.0:
        return make_one_hot_policy(best_action_id), {best_action_id: 1}

    policy = {
        int(action_id): float(weight / total)
        for action_id, weight in weights_by_action.items()
    }

    return policy, counts_by_action


def _safe_short_sequence(best_node) -> str:
    if hasattr(best_node, "short_sequence"):
        return str(best_node.short_sequence())

    parts = []

    for branch_id in getattr(best_node, "branch_ids", []):
        parts.append("stop" if branch_id is None else str(branch_id))

    return " -> ".join(parts) if parts else "(root)"


_TEACHER_SELECTION_MODE = "top_j_per_switch_then_redispatch_pareto"
_TERMINAL_REDISPATCH_RELATIVE_EPSILON = 0.01
_TERMINAL_REDISPATCH_ABSOLUTE_EPSILON_MW = 1.0
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _TerminalCandidate:
    node: Any
    redispatch_l1_mw: float
    redispatch_result: MinimalRedispatchResult | None = None


def _terminal_action_key(node: Any) -> tuple[int, ...]:
    return tuple(int(action_id) for action_id in node.action_ids)


def _terminal_candidate_key(candidate: _TerminalCandidate) -> tuple[object, ...]:
    return (
        switch_count(candidate.node),
        float(candidate.redispatch_l1_mw),
        float(candidate.node.safety_score),
        _terminal_action_key(candidate.node),
    )


def _same_terminal_objectives(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    return (
        switch_count(left.node) == switch_count(right.node)
        and abs(left.redispatch_l1_mw - right.redispatch_l1_mw) <= _TOLERANCE
    )


def _terminal_dominates(
    left: _TerminalCandidate,
    right: _TerminalCandidate,
) -> bool:
    left_switches = switch_count(left.node)
    right_switches = switch_count(right.node)
    left_redispatch = float(left.redispatch_l1_mw)
    right_redispatch = float(right.redispatch_l1_mw)
    no_worse = (
        left_switches <= right_switches
        and left_redispatch <= right_redispatch + _TOLERANCE
    )
    strictly_better = (
        left_switches < right_switches
        or left_redispatch < right_redispatch - _TOLERANCE
    )
    return no_worse and strictly_better


def _terminal_pareto_front(
    candidates: Sequence[_TerminalCandidate],
) -> list[_TerminalCandidate]:
    unique: list[_TerminalCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, other in enumerate(unique)
                if _same_terminal_objectives(candidate, other)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(candidate)
            continue
        if _terminal_candidate_key(candidate) < _terminal_candidate_key(
            unique[duplicate_index]
        ):
            unique[duplicate_index] = candidate

    front = [
        candidate
        for candidate in unique
        if not any(
            other is not candidate and _terminal_dominates(other, candidate)
            for other in unique
        )
    ]
    return sorted(front, key=_terminal_candidate_key)


def _with_handoff(node: Any) -> Any:
    if node.action_ids and int(node.action_ids[-1]) == 0:
        return node
    return replace(
        node,
        action_ids=[*node.action_ids, 0],
        branch_ids=[*node.branch_ids, None],
        done=True,
        solved=False,
        termination_reason=TerminationReason.HANDOFF_TO_REDISPATCH,
    )


def _terminal_candidate(
    node: Any,
    *,
    redispatch_result: MinimalRedispatchResult | None = None,
) -> _TerminalCandidate | None:
    state = node.env.current_state
    if state is None:
        return None
    assessment = assess_physical_state(state.metrics)
    if assessment.physically_secure:
        return _TerminalCandidate(node=node, redispatch_l1_mw=0.0)

    if redispatch_result is None:
        redispatch_result = run_minimal_ac_redispatch(node.env.backend, state)
    if not redispatch_result.validated or redispatch_result.redispatch_l1_mw is None:
        return None
    redispatch_l1_mw = float(redispatch_result.redispatch_l1_mw)
    if not math.isfinite(redispatch_l1_mw) or redispatch_l1_mw < 0.0:
        return None
    return _TerminalCandidate(
        node=_with_handoff(node),
        redispatch_l1_mw=redispatch_l1_mw,
        redispatch_result=redispatch_result,
    )


def _select_terminal_candidate(
    candidates: Sequence[_TerminalCandidate],
    *,
    relative_epsilon: float,
    absolute_epsilon_mw: float,
) -> tuple[_TerminalCandidate, list[_TerminalCandidate], list[_TerminalCandidate]]:
    front = _terminal_pareto_front(candidates)
    if not front:
        raise ValueError(
            "Terminal redispatch selection requires at least one candidate."
        )
    best_redispatch = min(candidate.redispatch_l1_mw for candidate in front)
    threshold = best_redispatch * (1.0 + float(relative_epsilon)) + float(
        absolute_epsilon_mw
    )
    pool = [
        candidate
        for candidate in front
        if candidate.redispatch_l1_mw <= threshold + _TOLERANCE
    ]
    selected = min(pool, key=_terminal_candidate_key)
    return selected, front, sorted(pool, key=_terminal_candidate_key)


def _redispatch_aware_selection(
    result,
    *,
    task_config: dict[str, Any],
    initial_redispatch_result: MinimalRedispatchResult | None = None,
) -> tuple[Any, dict[str, object]]:
    terminal_candidates: list[_TerminalCandidate] = []
    for node in result.redispatch_candidates:
        candidate = _terminal_candidate(
            node,
            redispatch_result=(
                initial_redispatch_result if not node.action_ids else None
            ),
        )
        if candidate is not None:
            terminal_candidates.append(candidate)

    diagnostics: dict[str, object] = {
        "terminal_redispatch_relative_epsilon": float(
            task_config["terminal_redispatch_relative_epsilon"]
        ),
        "terminal_redispatch_absolute_epsilon_mw": float(
            task_config["terminal_redispatch_absolute_epsilon_mw"]
        ),
        "teacher_terminal_selection_applied": False,
        "teacher_redispatch_candidate_count": int(len(result.redispatch_candidates)),
        "teacher_terminal_candidate_count": int(len(terminal_candidates)),
        "teacher_terminal_pareto_front_size": 0,
    }

    if not terminal_candidates:
        return result, diagnostics

    selected, terminal_front, terminal_pool = _select_terminal_candidate(
        terminal_candidates,
        relative_epsilon=float(task_config["terminal_redispatch_relative_epsilon"]),
        absolute_epsilon_mw=float(
            task_config["terminal_redispatch_absolute_epsilon_mw"]
        ),
    )
    diagnostics["teacher_terminal_selection_applied"] = True
    diagnostics["teacher_terminal_pareto_front_size"] = int(len(terminal_front))
    diagnostics["teacher_selected_redispatch_l1_mw"] = float(
        selected.redispatch_l1_mw
    )
    diagnostics["_selected_redispatch_result"] = selected.redispatch_result

    updated = replace(
        result,
        best_node=selected.node,
        final_beam=[
            candidate.node
            for candidate in terminal_pool[: result.config.beam_width]
        ],
    )
    return updated, diagnostics


def _selection_provenance(
    result,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "teacher_selection_mode": _TEACHER_SELECTION_MODE,
        "redispatch_candidates_per_switch_count": int(
            result.config.redispatch_candidates_per_switch_count
        ),
        "teacher_selected_J": float(result.best_node.safety_score),
        "teacher_selected_switch_count": int(switch_count(result.best_node)),
        **diagnostics,
    }


_RAW_SOURCE_FILES = ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet")
_RUNTIME_FIELDS = frozenset({"disable_cache"})
_TEACHER_RUN_ID_CACHE: dict[tuple[str, str], str] = {}


def _normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def teacher_source_identity(
    raw_dir: str | Path, transitions_path: str | Path
) -> dict[str, Any]:
    """Describe teacher inputs by content, independent of filesystem location."""
    raw_root = Path(raw_dir)
    transitions = Path(transitions_path)
    return {
        "raw_files": {
            name: file_content_identity(raw_root / name)
            for name in _RAW_SOURCE_FILES
        },
        "transitions": file_content_identity(transitions),
    }


def semantic_teacher_task_config(task_config: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize(
        {
            str(key): value
            for key, value in task_config.items()
            if str(key) not in _RUNTIME_FIELDS
        }
    )


def _legacy_teacher_source_matches(
    existing: object,
    current: object,
) -> bool:
    if existing == current:
        return True
    if not isinstance(existing, Mapping) or not isinstance(current, Mapping):
        return False

    legacy_files = existing.get("files")
    current_raw = current.get("raw_files")
    current_transitions = current.get("transitions")
    if (
        not isinstance(legacy_files, Mapping)
        or not isinstance(current_raw, Mapping)
        or not isinstance(current_transitions, Mapping)
    ):
        return False

    legacy_raw_sizes: dict[str, int] = {}
    legacy_other_sizes: list[int] = []
    for raw_path, metadata in legacy_files.items():
        if not isinstance(metadata, Mapping):
            return False
        try:
            size = int(metadata["size"])
        except (KeyError, TypeError, ValueError):
            return False
        name = str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]
        if name in _RAW_SOURCE_FILES:
            if name in legacy_raw_sizes:
                return False
            legacy_raw_sizes[name] = size
        else:
            legacy_other_sizes.append(size)

    try:
        current_raw_sizes = {
            name: int(current_raw[name]["size"])
            for name in _RAW_SOURCE_FILES
        }
        transition_size = int(current_transitions["size"])
    except (KeyError, TypeError, ValueError):
        return False

    return (
        legacy_raw_sizes == current_raw_sizes
        and legacy_other_sizes == [transition_size]
    )


def _write_teacher_checkpoint_identity(
    config_path: Path,
    identity: Mapping[str, Any],
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(
            dict(identity),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def ensure_teacher_checkpoint_config(
    config_path: Path, config: Mapping[str, Any]
) -> None:
    """Persist a portable resume identity and reject semantic mismatches."""
    task_config = config.get("task_config")
    if not isinstance(task_config, Mapping):
        raise ValueError("Teacher checkpoint config is missing task_config.")
    identity = _normalize(
        {
            "source_identity": config.get("source_identity"),
            "scenario_ids": config.get("scenario_ids"),
            "task_config": semantic_teacher_task_config(task_config),
        }
    )
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        same_scenarios = existing.get("scenario_ids") == identity["scenario_ids"]
        same_task = existing.get("task_config") == identity["task_config"]
        same_source = _legacy_teacher_source_matches(
            existing.get("source_identity"),
            identity["source_identity"],
        )
        if not (same_scenarios and same_task and same_source):
            raise RuntimeError(
                "Teacher checkpoint configuration does not match the current "
                "semantic inputs or settings. Use the original data/settings "
                "or a different output directory."
            )
        if existing != identity:
            _write_teacher_checkpoint_identity(config_path, identity)
        return
    _write_teacher_checkpoint_identity(config_path, identity)


def load_teacher_task_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")
    task_config = payload.get("task_config", payload)
    if not isinstance(task_config, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")
    return dict(task_config)


def _existing_teacher_run_id(output_dir: Path) -> str | None:
    checkpoint_path = output_dir / "teacher_checkpoint.jsonl"
    if not checkpoint_path.is_file():
        return None

    run_ids: set[str] = set()
    with checkpoint_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows = payload.get("rows")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                value = row.get("run_id")
                if isinstance(value, str) and value.strip():
                    run_ids.add(value.strip())

    if len(run_ids) > 1:
        raise RuntimeError(
            "Teacher checkpoint contains multiple run_id values and cannot be "
            "resumed safely."
        )
    return next(iter(run_ids), None)


def teacher_run_id(states_dir: str | Path, task_config: Mapping[str, Any]) -> str:
    """Return a relocation-stable teacher run ID, preserving legacy resumes."""
    output_dir = Path(states_dir).parent
    semantic_task = semantic_teacher_task_config(task_config)
    task_key = json.dumps(
        semantic_task,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    cache_key = (str(output_dir), task_key)
    cached = _TEACHER_RUN_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    existing = _existing_teacher_run_id(output_dir)
    if existing is not None:
        _TEACHER_RUN_ID_CACHE[cache_key] = existing
        return existing

    config_path = output_dir / "teacher_checkpoint_config.json"
    source_identity = None
    scenario_ids = None
    if config_path.is_file():
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(checkpoint_config, Mapping):
            source_identity = checkpoint_config.get("source_identity")
            scenario_ids = checkpoint_config.get("scenario_ids")

    payload = {
        "source_identity": source_identity,
        "scenario_ids": scenario_ids,
        "task_config": semantic_task,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    run_id = f"impact_teacher_{hashlib.sha256(encoded).hexdigest()[:24]}"
    _TEACHER_RUN_ID_CACHE[cache_key] = run_id
    return run_id
