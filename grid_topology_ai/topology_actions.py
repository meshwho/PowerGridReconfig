from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Literal


SlotKind = Literal[
    "stop",
    "branch_status",
]

ActionType = Literal[
    "do_nothing",
    "switch_off_branch",
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


@dataclass(frozen=True)
class GridFMAction:
    """
    One executable topology action.

    This is the current compatibility form. Directional
    branch-status commands will be added in the next stage.
    """

    action_id: int
    action_type: ActionType
    branch_id: int | None = None
    branch_pos: int | None = None


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