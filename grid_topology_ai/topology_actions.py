from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Literal


SlotKind = Literal[
    "stop",
    "branch_status",
]

ActionKind = Literal[
    "stop",
    "set_branch_status",
]

ActionType = Literal[
    "do_nothing",
    "switch_off_branch",
    "switch_on_branch",
]

def _non_negative_int(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    parsed = int(value)

    if parsed < 0:
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    return parsed

def _binary_status(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be either 0 or 1."
        )

    parsed = float(value)

    if (
        not math.isfinite(parsed)
        or parsed not in (0.0, 1.0)
    ):
        raise ValueError(
            f"{name} must be either 0 or 1."
        )

    return int(parsed)

@dataclass(frozen=True, slots=True)
class ActionSlot:
    """
    One stable position in the policy vector.

    A slot identifies the controlled object. It does not
    describe the state-dependent command applied to it.
    """

    action_id: int
    kind: SlotKind
    target_id: int | None
    target_pos: int | None

    def __post_init__(self) -> None:
        action_id = _non_negative_int(
            "action_id",
            self.action_id,
        )

        if self.kind == "stop":
            if action_id != 0:
                raise ValueError(
                    "The stop slot must use action_id 0."
                )

            if (
                self.target_id is not None
                or self.target_pos is not None
            ):
                raise ValueError(
                    "The stop slot must not have a target."
                )

            return

        if self.kind != "branch_status":
            raise ValueError(
                f"Unsupported action slot kind: {self.kind!r}."
            )

        if (
            self.target_id is None
            or self.target_pos is None
        ):
            raise ValueError(
                "A branch-status slot requires target_id "
                "and target_pos."
            )

        target_id = _non_negative_int(
            "target_id",
            self.target_id,
        )
        target_pos = _non_negative_int(
            "target_pos",
            self.target_pos,
        )

        expected_action_id = 1 + target_pos

        if action_id != expected_action_id:
            raise ValueError(
                "Branch-status action_id must equal "
                f"1 + target_pos. Expected "
                f"{expected_action_id}, got {action_id}."
            )

        object.__setattr__(
            self,
            "action_id",
            action_id,
        )
        object.__setattr__(
            self,
            "target_id",
            target_id,
        )
        object.__setattr__(
            self,
            "target_pos",
            target_pos,
        )


@dataclass(frozen=True, slots=True)
class ActionSpaceConfig:
    require_connected_after_switch: bool = True
    min_loading_for_switch_percent: float = 0.0
    closeable_branch_ids: tuple[int, ...] = ()
    enable_cache: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.require_connected_after_switch,
            bool,
        ):
            raise ValueError(
                "require_connected_after_switch must be "
                "a boolean."
            )

        if not isinstance(
            self.enable_cache,
            bool,
        ):
            raise ValueError(
                "enable_cache must be a boolean."
            )

        threshold = (
            self.min_loading_for_switch_percent
        )

        if isinstance(threshold, bool):
            raise ValueError(
                "min_loading_for_switch_percent must be "
                "a finite non-negative number."
            )

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(
                "min_loading_for_switch_percent must be "
                "a finite non-negative number."
            ) from None

        if (
            not math.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError(
                "min_loading_for_switch_percent must be "
                "a finite non-negative number."
            )

        object.__setattr__(
            self,
            "min_loading_for_switch_percent",
            threshold,
        )
        try:
            closeable_branch_ids = tuple(
                _non_negative_int(
                    "closeable_branch_ids item",
                    branch_id,
                )
                for branch_id
                in self.closeable_branch_ids
            )
        except TypeError:
            raise ValueError(
                "closeable_branch_ids must be an "
                "iterable of non-negative integers."
            ) from None

        if (
            len(set(closeable_branch_ids))
            != len(closeable_branch_ids)
        ):
            raise ValueError(
                "closeable_branch_ids must not contain "
                "duplicate branch IDs."
            )

        object.__setattr__(
            self,
            "closeable_branch_ids",
            tuple(sorted(closeable_branch_ids)),
        )


@dataclass(frozen=True, slots=True)
class GridFMAction:
    """
    One executable topology action.

    Branch actions use one stable policy slot per branch.
    The current branch status determines whether the command
    opens or closes that branch.
    """

    action_id: int
    action_type: ActionType
    branch_id: int | None = None
    branch_pos: int | None = None
    target_status: int | None = None

    def __post_init__(self) -> None:
        action_id = _non_negative_int(
            "action_id",
            self.action_id,
        )

        if self.action_type == "do_nothing":
            if action_id != 0:
                raise ValueError(
                    "do_nothing must use action_id 0."
                )

            if (
                self.branch_id is not None
                or self.branch_pos is not None
                or self.target_status is not None
            ):
                raise ValueError(
                    "do_nothing must not have a branch "
                    "target or target_status."
                )

            object.__setattr__(
                self,
                "action_id",
                action_id,
            )
            return

        if self.action_type not in {
            "switch_off_branch",
            "switch_on_branch",
        }:
            raise ValueError(
                f"Unsupported action type: "
                f"{self.action_type!r}."
            )

        if (
            self.branch_id is None
            or self.branch_pos is None
        ):
            raise ValueError(
                "A branch-status action requires "
                "branch_id and branch_pos."
            )

        branch_id = _non_negative_int(
            "branch_id",
            self.branch_id,
        )
        branch_pos = _non_negative_int(
            "branch_pos",
            self.branch_pos,
        )

        expected_action_id = 1 + branch_pos

        if action_id != expected_action_id:
            raise ValueError(
                "Branch action_id must equal "
                f"1 + branch_pos. Expected "
                f"{expected_action_id}, got {action_id}."
            )

        expected_target_status = (
            0
            if self.action_type == "switch_off_branch"
            else 1
        )

        target_status = (
            expected_target_status
            if self.target_status is None
            else _binary_status(
                "target_status",
                self.target_status,
            )
        )

        if target_status != expected_target_status:
            raise ValueError(
                f"{self.action_type} requires "
                f"target_status={expected_target_status}."
            )

        object.__setattr__(
            self,
            "action_id",
            action_id,
        )
        object.__setattr__(
            self,
            "branch_id",
            branch_id,
        )
        object.__setattr__(
            self,
            "branch_pos",
            branch_pos,
        )
        object.__setattr__(
            self,
            "target_status",
            target_status,
        )

    @property
    def kind(self) -> ActionKind:
        if self.action_type == "do_nothing":
            return "stop"

        return "set_branch_status"

    @property
    def target_id(self) -> int | None:
        return self.branch_id

    @property
    def target_pos(self) -> int | None:
        return self.branch_pos


def build_branch_action_slots(
    branch_ids: Iterable[int],
) -> tuple[ActionSlot, ...]:
    """
    Build the stable stop-plus-branch policy layout.

    Slot 0 is stop. Slot 1 + branch_pos controls the branch
    stored at that position.
    """

    slots: list[ActionSlot] = [
        ActionSlot(
            action_id=0,
            kind="stop",
            target_id=None,
            target_pos=None,
        )
    ]

    for branch_pos, branch_id in enumerate(
        branch_ids
    ):
        slots.append(
            ActionSlot(
                action_id=1 + branch_pos,
                kind="branch_status",
                target_id=int(branch_id),
                target_pos=branch_pos,
            )
        )

    return tuple(slots)

def branch_status_signature(
    branch_ids: Iterable[int],
    branch_status: Iterable[float],
) -> tuple[tuple[int, int], ...]:
    """
    Return the canonical branch topology represented by
    stable branch IDs and their current statuses.
    """

    ids = tuple(
        _non_negative_int(
            "branch_id",
            branch_id,
        )
        for branch_id in branch_ids
    )
    statuses = tuple(
        _binary_status(
            "branch_status",
            status,
        )
        for status in branch_status
    )

    if len(ids) != len(statuses):
        raise ValueError(
            "branch_ids and branch_status must have "
            "the same length."
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            "branch_ids must be unique."
        )

    return tuple(zip(ids, statuses))