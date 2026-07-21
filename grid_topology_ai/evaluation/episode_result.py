from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from grid_topology_ai.config.physics import PhysicsConfig
from grid_topology_ai.evaluation.metrics import compute_safety_score
from grid_topology_ai.physical_objective import assess_physical_state
from grid_topology_ai.termination import (
    TerminationReason,
    termination_reason_value,
    validate_outcome_invariants,
)


@dataclass(slots=True)
class EvaluationEpisodeTrace:
    actions: list[int] = field(default_factory=list)
    branches: list[int | None] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    raw_policies: list[dict[int, float]] = field(default_factory=list)
    executed_policies: list[dict[int, float]] = field(default_factory=list)
    allowed_action_ids: list[list[int]] = field(default_factory=list)
    total_reward: float = 0.0
    discounted_return: float = 0.0
    constraint_changed_policy_steps: int = 0
    empty_constrained_support_count: int = 0

    @property
    def constraint_exhausted(self) -> bool:
        return self.empty_constrained_support_count > 0


def build_evaluation_episode_row(
    *,
    scenario_id: int,
    policy_mode: str,
    env: Any,
    trace: EvaluationEpisodeTrace,
    physics_config: PhysicsConfig | None,
) -> dict[str, Any]:
    effective_reason = (
        TerminationReason.CONSTRAINT_EXHAUSTED
        if trace.constraint_exhausted
        else env.termination_reason
    )
    effective_done = bool(env.done or trace.constraint_exhausted)
    effective_solved = bool(env.solved)
    physical = _physical_result_fields(
        env=env,
        effective_done=effective_done,
        effective_solved=effective_solved,
        effective_reason=effective_reason,
    )

    row = {
        "scenario_id": int(scenario_id),
        "policy_mode": str(policy_mode),
        "steps": len(trace.actions),
        "policy_decisions": len(trace.executed_policies),
        "use_continuation_gate": policy_mode == "constrained",
        "actions": str(trace.actions),
        "branches": str(trace.branches),
        "rewards": str([round(value, 4) for value in trace.rewards]),
        "raw_root_policies_json": json.dumps(
            trace.raw_policies,
            sort_keys=True,
        ),
        "root_policies_json": json.dumps(
            trace.executed_policies,
            sort_keys=True,
        ),
        "allowed_action_ids_json": json.dumps(trace.allowed_action_ids),
        "constraint_changed_policy": bool(
            trace.constraint_changed_policy_steps
        ),
        "constraint_changed_policy_steps": int(
            trace.constraint_changed_policy_steps
        ),
        "constraint_exhausted": trace.constraint_exhausted,
        "empty_constrained_support_count": int(
            trace.empty_constrained_support_count
        ),
        "total_reward": float(trace.total_reward),
        "discounted_return": float(trace.discounted_return),
        "done": effective_done,
        "solved": effective_solved,
        "termination_reason": termination_reason_value(effective_reason),
        **physical,
    }
    row["safety_score"] = compute_safety_score(
        row,
        physics_config=physics_config,
    )
    return row


def _physical_result_fields(
    *,
    env: Any,
    effective_done: bool,
    effective_solved: bool,
    effective_reason: TerminationReason | None,
) -> dict[str, Any]:
    final_state = env.current_state
    if final_state is None:
        return {
            "final_max_loading_percent": float("nan"),
            "final_num_overloaded_branches": -1,
            "final_num_hard_overloaded_branches": -1,
            "final_num_outaged_branches": -1,
            "thermal_solved": False,
            "thermal_feasible": False,
            "power_flow_converged": False,
            "all_values_finite": False,
            "topology_connected": False,
            "hard_overload_free": False,
            "voltage_feasible": False,
            "generator_p_feasible": False,
            "generator_q_feasible": False,
            "angle_difference_feasible": False,
            "physically_secure": False,
            "num_generator_p_violations": -1,
            "num_generator_q_violations": -1,
            "num_angle_difference_violations": -1,
            "total_generator_p_violation_mw": float("nan"),
            "total_generator_q_violation_mvar": float("nan"),
            "total_angle_difference_violation_degrees": float("nan"),
            "total_voltage_violation": float("nan"),
            "num_low_voltage_buses": -1,
            "num_high_voltage_buses": -1,
            "total_thermal_overload_mva": float("nan"),
            "safe_handoff": False,
            "unsafe_terminal_state": effective_done,
        }

    assessment = assess_physical_state(final_state.metrics)
    validate_outcome_invariants(
        solved=effective_solved,
        termination_reason=effective_reason,
        physically_secure=assessment.physically_secure,
    )
    safe_handoff = (
        effective_reason is TerminationReason.HANDOFF_TO_REDISPATCH
        and assessment.hard_overload_free
        and not assessment.physically_secure
    )
    return {
        "final_max_loading_percent": float(
            final_state.metrics["max_loading_percent"]
        ),
        "final_num_overloaded_branches": int(
            final_state.metrics["num_overloaded_branches"]
        ),
        "final_num_hard_overloaded_branches": int(
            final_state.metrics["num_hard_overloaded_branches"]
        ),
        "final_num_outaged_branches": int(
            final_state.metrics["num_outaged_branches"]
        ),
        "thermal_solved": assessment.thermal_solved,
        "thermal_feasible": assessment.thermal_feasible,
        "power_flow_converged": assessment.power_flow_converged,
        "all_values_finite": assessment.all_values_finite,
        "topology_connected": assessment.topology_connected,
        "hard_overload_free": assessment.hard_overload_free,
        "voltage_feasible": assessment.voltage_feasible,
        "generator_p_feasible": assessment.generator_p_feasible,
        "generator_q_feasible": assessment.generator_q_feasible,
        "angle_difference_feasible": assessment.angle_difference_feasible,
        "physically_secure": assessment.physically_secure,
        "num_generator_p_violations": assessment.num_generator_p_violations,
        "num_generator_q_violations": assessment.num_generator_q_violations,
        "num_angle_difference_violations": (
            assessment.num_angle_difference_violations
        ),
        "total_generator_p_violation_mw": (
            assessment.total_generator_p_violation_mw
        ),
        "total_generator_q_violation_mvar": (
            assessment.total_generator_q_violation_mvar
        ),
        "total_angle_difference_violation_degrees": (
            assessment.total_angle_difference_violation_degrees
        ),
        "total_voltage_violation": assessment.total_voltage_violation,
        "num_low_voltage_buses": assessment.num_low_voltage_buses,
        "num_high_voltage_buses": assessment.num_high_voltage_buses,
        "total_thermal_overload_mva": assessment.total_thermal_overload_mva,
        "safe_handoff": safe_handoff,
        "unsafe_terminal_state": bool(
            effective_done
            and not assessment.physically_secure
            and not safe_handoff
        ),
    }
