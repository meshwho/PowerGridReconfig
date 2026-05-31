from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grid_topology_ai.action_space import GridFMAction
from grid_topology_ai.data_adapter import BRANCH_FEATURE_COLUMNS, GridFMState
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
    selected_action_id: int
    selected_branch_id: int | None
    selected_reason: str

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


def _active_loadings(state: GridFMState) -> np.ndarray:
    status_idx = BRANCH_FEATURE_COLUMNS.index("br_status")
    loading_idx = BRANCH_FEATURE_COLUMNS.index("loading_percent")

    status = state.branch_features[:, status_idx]
    loading = state.branch_features[:, loading_idx]

    return loading[status > 0.0]


def topology_penalty(
    state: GridFMState,
    depth: int = 0,
    switch_penalty: float = 8.0,
) -> float:
    """
    Lower is better.

    This penalty is used only for lookahead branch comparison.
    It is intentionally operational:
      - hard overloads dominate;
      - soft overloads matter;
      - extra switching is penalized;
      - voltage violation is included.
    """

    loading = _active_loadings(state)

    total_overload = float(np.sum(np.maximum(loading - 100.0, 0.0)))
    hard_overload = float(np.sum(np.maximum(loading - 120.0, 0.0)))

    num_overloaded = int(state.metrics.get("num_overloaded_branches", 0))
    num_hard = int(state.metrics.get("num_hard_overloaded_branches", 0))

    max_loading = float(state.metrics.get("max_loading_percent", 0.0))
    voltage_violation = float(state.metrics.get("total_voltage_violation", 0.0))

    return float(
        1000.0 * num_hard
        + 30.0 * hard_overload
        + 80.0 * num_overloaded
        + 4.0 * total_overload
        + 5.0 * max(0.0, max_loading - 100.0)
        + 500.0 * voltage_violation
        + switch_penalty * float(depth)
    )


def _node_penalty(node: MCTSNode) -> float:
    state = node.env.current_state

    if state is None:
        return 1e9

    return topology_penalty(
        state=state,
        depth=node.depth,
    )


def _node_metrics(node: MCTSNode) -> dict[str, Any]:
    state = node.env.current_state

    if state is None:
        return {}

    return dict(state.metrics)


def _best_reachable_state_from_subtree(
    start_node: MCTSNode,
    first_action_id: int,
) -> tuple[float, list[int], list[int | None], list[float], dict[str, Any]]:
    """
    Iterative DFS over already-built MCTS subtree.

    This is intentionally non-recursive and avoids repeated list copying where possible.
    """

    best_penalty = _node_penalty(start_node)
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

        penalty = _node_penalty(node)

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
            return (
                False,
                f"hard_not_reduced final_hard={final_hard}, root_hard={root_num_hard}",
            )

        if improvement < min_hard_improvement:
            return (
                False,
                f"hard_improvement_too_small {improvement:.3f} < {min_hard_improvement:.3f}",
            )

        return (
            True,
            f"hard_reduced final_hard={final_hard}, final_max={final_max:.3f}, improvement={improvement:.3f}",
        )

    if final_hard > 0:
        return False, f"created_hard_overload final_hard={final_hard}"

    if improvement < min_soft_improvement:
        return (
            False,
            f"soft_improvement_too_small {improvement:.3f} < {min_soft_improvement:.3f}",
        )

    return (
        True,
        f"useful_soft_improvement final_max={final_max:.3f}, improvement={improvement:.3f}",
    )


def analyze_root_branches(
    result: MCTSResult,
    min_hard_improvement: float = 50.0,
    min_soft_improvement: float = 15.0,
    min_visits: int = 5,
    min_visit_fraction: float = 0.01,
) -> ContinuationDecision:
    """
    Decide executed action using lookahead gate.

    Important:
    - MCTS visit counts are still used as confidence.
    - Improvement is used as a gate, not as the only selection criterion.
    - Among allowed branches, select the branch with the highest visit count.

    This prevents:
      - meaningless first moves;
      - endless topology switching after hard overloads are already cleared;
      - selecting a low-confidence branch only because it had one lucky leaf.
    """

    root = result.root
    root_state = root.env.current_state

    if root_state is None:
        return ContinuationDecision(
            selected_action_id=0,
            selected_branch_id=None,
            selected_reason="no_root_state",
            root_penalty=1e9,
            root_has_hard_overload=True,
            root_num_hard=999,
            best_visit_action_id=result.best_action_id,
            best_visit_branch_id=result.best_branch_id,
            best_allowed_action_id=None,
            best_allowed_branch_id=None,
            best_allowed_improvement=0.0,
            best_improvement_action_id=None,
            best_improvement_branch_id=None,
            best_improvement=0.0,
            branches=[],
        )

    root_penalty = topology_penalty(root_state, depth=0)

    root_num_hard = int(root_state.metrics.get("num_hard_overloaded_branches", 0))
    root_has_hard = root_num_hard > 0

    total_visits = sum(child.visit_count for child in root.children.values())
    total_visits = max(total_visits, 1)

    branches: list[BranchContinuation] = []

    for action_id, child in root.children.items():
        best_penalty, seq_actions, seq_branches, seq_rewards, best_metrics = (
            _best_reachable_state_from_subtree(
                start_node=child,
                first_action_id=int(action_id),
            )
        )

        visits = int(child.visit_count)
        policy = float(visits / total_visits)
        improvement = float(root_penalty - best_penalty)

        confidence_ok = visits >= int(min_visits) and policy >= float(min_visit_fraction)

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

    branches_by_improvement = sorted(
        branches,
        key=lambda item: (item.improvement, item.visits),
        reverse=True,
    )

    best_improvement_branch = branches_by_improvement[0] if branches_by_improvement else None

    allowed = [branch for branch in branches if branch.allow]

    if root_has_hard:
        allowed_sorted = sorted(
            allowed,
            key=lambda item: (item.visits, item.improvement),
            reverse=True,
        )
        selected_reason = "best_allowed_by_visits"
    else:
        allowed_sorted = sorted(
            allowed,
            key=lambda item: (item.improvement, item.visits),
            reverse=True,
        )
        selected_reason = "best_allowed_by_improvement"

    if allowed_sorted:
        selected = allowed_sorted[0]

        return ContinuationDecision(
            selected_action_id=int(selected.action_id),
            selected_branch_id=selected.branch_id,
            selected_reason=selected_reason,
            root_penalty=float(root_penalty),
            root_has_hard_overload=root_has_hard,
            root_num_hard=root_num_hard,
            best_visit_action_id=result.best_action_id,
            best_visit_branch_id=result.best_branch_id,
            best_allowed_action_id=selected.action_id,
            best_allowed_branch_id=selected.branch_id,
            best_allowed_improvement=float(selected.improvement),
            best_improvement_action_id=(
                None if best_improvement_branch is None else best_improvement_branch.action_id
            ),
            best_improvement_branch_id=(
                None if best_improvement_branch is None else best_improvement_branch.branch_id
            ),
            best_improvement=(
                0.0 if best_improvement_branch is None else float(best_improvement_branch.improvement)
            ),
            branches=sorted(
                branches,
                key=lambda item: (item.allow, item.visits, item.improvement),
                reverse=True,
            ),
        )

    # ------------------------------------------------------------------
    # Hard-overload fallback
    # ------------------------------------------------------------------
    # If hard overloads are still present, do not stop just because the
    # strict gate did not find a fully convincing continuation.
    #
    # Stopping with hard overload teaches the policy to do nothing in an
    # unsafe state. For self-play this is especially harmful.
    #
    # Therefore:
    #   - if hard overload exists;
    #   - and no branch passed the strict allow rules;
    #   - execute the most trusted non-stop branch as a fallback.
    #
    # This keeps exploration alive on new/OOD GridFM scenarios.
    if root_has_hard:
        non_stop_branches = [
            branch
            for branch in branches
            if int(branch.action_id) != 0 and branch.branch_id is not None
        ]

        if non_stop_branches:
            fallback = sorted(
                non_stop_branches,
                key=lambda item: (
                    item.confidence_ok,
                    item.visits,
                    item.improvement,
                ),
                reverse=True,
            )[0]

            return ContinuationDecision(
                selected_action_id=int(fallback.action_id),
                selected_branch_id=fallback.branch_id,
                selected_reason="fallback_hard_overload_best_non_stop_by_visits",
                root_penalty=float(root_penalty),
                root_has_hard_overload=root_has_hard,
                root_num_hard=root_num_hard,
                best_visit_action_id=result.best_action_id,
                best_visit_branch_id=result.best_branch_id,
                best_allowed_action_id=None,
                best_allowed_branch_id=None,
                best_allowed_improvement=0.0,
                best_improvement_action_id=(
                    None
                    if best_improvement_branch is None
                    else best_improvement_branch.action_id
                ),
                best_improvement_branch_id=(
                    None
                    if best_improvement_branch is None
                    else best_improvement_branch.branch_id
                ),
                best_improvement=(
                    0.0
                    if best_improvement_branch is None
                    else float(best_improvement_branch.improvement)
                ),
                branches=sorted(
                    branches,
                    key=lambda item: (item.allow, item.visits, item.improvement),
                    reverse=True,
                ),
            )

    return ContinuationDecision(
        selected_action_id=0,
        selected_branch_id=None,
        selected_reason="no_useful_topology_continuation",
        root_penalty=float(root_penalty),
        root_has_hard_overload=root_has_hard,
        root_num_hard=root_num_hard,
        best_visit_action_id=result.best_action_id,
        best_visit_branch_id=result.best_branch_id,
        best_allowed_action_id=None,
        best_allowed_branch_id=None,
        best_allowed_improvement=0.0,
        best_improvement_action_id=(
            None if best_improvement_branch is None else best_improvement_branch.action_id
        ),
        best_improvement_branch_id=(
            None if best_improvement_branch is None else best_improvement_branch.branch_id
        ),
        best_improvement=(
            0.0 if best_improvement_branch is None else float(best_improvement_branch.improvement)
        ),
        branches=sorted(
            branches,
            key=lambda item: (item.allow, item.visits, item.improvement),
            reverse=True,
        ),
    )


def make_do_nothing_action() -> GridFMAction:
    return GridFMAction(
        action_id=0,
        action_type="do_nothing",
        branch_id=None,
        branch_pos=None,
    )