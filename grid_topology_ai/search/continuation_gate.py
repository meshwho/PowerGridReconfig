from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG, PhysicsConfig
from grid_topology_ai.data_adapter import GridFMState
from grid_topology_ai.grid_utility import (
    CONTINUATION_GRID_UTILITY_WEIGHTS,
    CONTINUATION_SWITCH_PENALTY,
    state_security_penalty,
)
from grid_topology_ai.search.mcts import MCTSNode, MCTSResult


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
                f"hard_not_reduced final_hard={final_hard}, "
                f"root_hard={root_num_hard}"
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
            f"soft_improvement_too_small {improvement:.3f} "
            f"< {min_soft_improvement:.3f}"
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
        allowed_action_ids=tuple(branch.action_id for branch in ordered if branch.allow),
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


def make_do_nothing_action() -> GridFMAction:
    return GridFMAction(
        action_id=0,
        action_type="do_nothing",
        branch_id=None,
        branch_pos=None,
    )
