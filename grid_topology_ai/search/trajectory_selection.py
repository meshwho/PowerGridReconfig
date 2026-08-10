from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar


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
