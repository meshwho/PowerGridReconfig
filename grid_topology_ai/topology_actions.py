from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
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

STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT = (
    "stop_plus_branch_status_v1"
)


def _fingerprint_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()

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



    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": int(self.action_id),
            "kind": str(self.kind),
            "target_id": (
                None
                if self.target_id is None
                else int(self.target_id)
            ),
            "target_pos": (
                None
                if self.target_pos is None
                else int(self.target_pos)
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
    ) -> "ActionSlot":
        if not isinstance(data, Mapping):
            raise ValueError(
                "Action slot must be a mapping."
            )

        required = {
            "action_id",
            "kind",
            "target_id",
            "target_pos",
        }

        unknown = set(data) - required

        if unknown:
            raise ValueError(
                "Unknown action slot fields: "
                f"{sorted(unknown)}."
            )

        missing = required - set(data)

        if missing:
            raise ValueError(
                "Missing action slot fields: "
                f"{sorted(missing)}."
            )

        return cls(
            action_id=data["action_id"],
            kind=data["kind"],
            target_id=data["target_id"],
            target_pos=data["target_pos"],
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

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "require_connected_after_switch": bool(
                self.require_connected_after_switch
            ),
            "min_loading_for_switch_percent": float(
                self.min_loading_for_switch_percent
            ),
            "closeable_branch_ids": [
                int(branch_id)
                for branch_id in self.closeable_branch_ids
            ],
        }

    @classmethod
    def from_contract_mapping(
        cls,
        data: Mapping[str, object],
    ) -> "ActionSpaceConfig":
        if not isinstance(data, Mapping):
            raise ValueError(
                "Topology action config must be a mapping."
            )

        required = {
            "require_connected_after_switch",
            "min_loading_for_switch_percent",
            "closeable_branch_ids",
        }

        unknown = set(data) - required

        if unknown:
            raise ValueError(
                "Unknown topology action settings: "
                f"{sorted(unknown)}."
            )

        missing = required - set(data)

        if missing:
            raise ValueError(
                "Missing topology action settings: "
                f"{sorted(missing)}."
            )

        closeable_branch_ids = data[
            "closeable_branch_ids"
        ]

        if not isinstance(
            closeable_branch_ids,
            (list, tuple),
        ):
            raise ValueError(
                "closeable_branch_ids must be a list."
            )

        return cls(
            require_connected_after_switch=data[
                "require_connected_after_switch"
            ],
            min_loading_for_switch_percent=data[
                "min_loading_for_switch_percent"
            ],
            closeable_branch_ids=tuple(
                closeable_branch_ids
            ),
        )

    def contract_fingerprint(self) -> str:
        return _fingerprint_json(
            self.to_contract_dict()
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

def action_layout_to_list(
    action_layout: Iterable[ActionSlot],
) -> list[dict[str, object]]:
    slots = tuple(action_layout)

    if not slots:
        raise ValueError(
            "Action layout must not be empty."
        )

    action_ids = [
        int(slot.action_id)
        for slot in slots
    ]
    expected_action_ids = list(
        range(len(slots))
    )

    if action_ids != expected_action_ids:
        raise ValueError(
            "Action layout IDs must be contiguous and "
            f"ordered from 0. Observed {action_ids}."
        )

    return [
        slot.to_dict()
        for slot in slots
    ]


def action_layout_from_value(
    value: object,
) -> tuple[ActionSlot, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid action layout JSON."
            ) from exc

    if not isinstance(value, list):
        raise ValueError(
            "Action layout must be a list."
        )

    slots = tuple(
        ActionSlot.from_mapping(item)
        for item in value
    )

    action_layout_to_list(slots)

    return slots


def action_layout_fingerprint(
    action_layout: Iterable[ActionSlot],
) -> str:
    return _fingerprint_json(
        action_layout_to_list(action_layout)
    )


def require_branch_status_policy_layout(
    action_layout: Iterable[ActionSlot],
) -> str:
    slots = tuple(action_layout)

    action_layout_to_list(slots)

    if slots[0].kind != "stop":
        raise ValueError(
            "The current policy head requires stop at "
            "action_id 0."
        )

    if any(
        slot.kind != "branch_status"
        for slot in slots[1:]
    ):
        raise ValueError(
            "The current policy head supports only "
            "branch-status action slots."
        )

    return STOP_PLUS_BRANCH_STATUS_POLICY_LAYOUT

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