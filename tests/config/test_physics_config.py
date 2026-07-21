from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from grid_topology_ai.config.physics import (
    DEFAULT_PHYSICS_CONFIG,
    PhysicsConfig,
    QLimitPolicy,
    ZeroRateAPolicy,
    resolve_physics_config,
)


def test_numpy_integral_settings_have_canonical_fingerprint() -> None:
    first = PhysicsConfig(pf_alg=np.int64(3), max_iterations=np.int32(30))
    second = PhysicsConfig()
    assert type(first.pf_alg) is int
    assert type(first.max_iterations) is int
    assert first.fingerprint() == second.fingerprint()


def test_config_is_frozen_and_slotted() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_PHYSICS_CONFIG.pf_alg = 1  # type: ignore[misc]

    assert not hasattr(DEFAULT_PHYSICS_CONFIG, "__dict__")


def test_mapping_round_trip_preserves_canonical_config_and_fingerprint() -> None:
    config = PhysicsConfig(
        base_mva=110.0,
        pf_alg=4,
        pf_tolerance=1e-9,
        max_iterations=45,
        q_limit_policy=QLimitPolicy.VALIDATE_ONLY,
        zero_rate_a_policy=ZeroRateAPolicy.ERROR,
        overload_limit_percent=115.0,
        hard_overload_limit_percent=135.0,
        thermal_tolerance_percent=0.01,
        voltage_tolerance_pu=0.002,
        generator_p_tolerance_mw=0.03,
        generator_q_tolerance_mvar=0.04,
        angle_tolerance_degrees=0.05,
    )

    restored = PhysicsConfig.from_mapping(config.to_dict())

    assert restored == config
    assert restored.fingerprint() == config.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_mva", 110.0),
        ("pf_alg", 1),
        ("pf_tolerance", 1e-9),
        ("max_iterations", 31),
        ("q_limit_policy", QLimitPolicy.VALIDATE_ONLY),
        ("zero_rate_a_policy", ZeroRateAPolicy.ERROR),
        ("overload_limit_percent", 101.0),
        ("hard_overload_limit_percent", 121.0),
        ("thermal_tolerance_percent", 0.01),
        ("voltage_tolerance_pu", 0.002),
        ("generator_p_tolerance_mw", 0.03),
        ("generator_q_tolerance_mvar", 0.04),
        ("angle_tolerance_degrees", 0.05),
    ],
)
def test_every_configurable_physics_setting_changes_fingerprint(
    field: str,
    value: object,
) -> None:
    changed = replace(DEFAULT_PHYSICS_CONFIG, **{field: value})

    assert changed.fingerprint() != DEFAULT_PHYSICS_CONFIG.fingerprint()


@pytest.mark.parametrize("value", [True, False])
def test_boolean_algorithm_is_rejected(value: bool) -> None:
    with pytest.raises(ValueError, match="pf_alg"):
        PhysicsConfig(pf_alg=value)


@pytest.mark.parametrize("value", [0, 5, 3.0, "3", None])
def test_non_exact_or_unsupported_algorithm_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="pf_alg"):
        PhysicsConfig(pf_alg=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 0, -1, 30.0, "30", None])
def test_non_positive_or_non_exact_iteration_count_is_rejected(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        PhysicsConfig(max_iterations=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    ["base_mva", "pf_tolerance", "overload_limit_percent"],
)
@pytest.mark.parametrize(
    "value",
    [True, 0.0, -1.0, float("nan"), float("inf"), "1.0"],
)
def test_positive_finite_settings_reject_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_PHYSICS_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "thermal_tolerance_percent",
        "voltage_tolerance_pu",
        "generator_p_tolerance_mw",
        "generator_q_tolerance_mvar",
        "angle_tolerance_degrees",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, -1.0, float("nan"), float("inf"), "0.0"],
)
def test_nonnegative_finite_tolerances_reject_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_PHYSICS_CONFIG, **{field: value})


def test_hard_overload_limit_cannot_be_below_overload_limit() -> None:
    with pytest.raises(ValueError, match="hard_overload_limit_percent"):
        PhysicsConfig(
            overload_limit_percent=120.0,
            hard_overload_limit_percent=119.0,
        )


@pytest.mark.parametrize(
    "field",
    ["q_limit_policy", "island_policy", "zero_rate_a_policy"],
)
def test_unsupported_policy_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_PHYSICS_CONFIG, **{field: "unsupported"})


def test_from_mapping_rejects_non_mapping_and_unknown_settings() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        PhysicsConfig.from_mapping([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown physics settings"):
        PhysicsConfig.from_mapping({"pf_alg": 3, "hidden_threshold": 1})


@pytest.mark.parametrize("legacy", [3, np.int64(3), 3.0, "3", " 3 "])
def test_legacy_algorithm_resolves_to_the_same_canonical_config(
    legacy: object,
) -> None:
    resolved = resolve_physics_config(None, legacy)  # type: ignore[arg-type]

    assert resolved == DEFAULT_PHYSICS_CONFIG
    assert resolved.fingerprint() == DEFAULT_PHYSICS_CONFIG.fingerprint()


@pytest.mark.parametrize("legacy", [True, False, 3.5, "3.0", "x", object()])
def test_invalid_legacy_algorithm_is_rejected(legacy: object) -> None:
    with pytest.raises(ValueError, match="legacy pf_alg"):
        resolve_physics_config(None, legacy)  # type: ignore[arg-type]


def test_none_legacy_algorithm_returns_default_singleton() -> None:
    assert resolve_physics_config(None, None) is DEFAULT_PHYSICS_CONFIG


def test_explicit_matching_config_is_preserved() -> None:
    config = PhysicsConfig(pf_alg=2)

    assert resolve_physics_config(config, 2) is config


def test_explicit_default_config_conflicts_with_legacy_algorithm(tmp_path) -> None:
    from grid_topology_ai.config.evaluation import EvaluationConfig
    from grid_topology_ai.evaluation.checkpoint import EvaluationRequest

    with pytest.raises(ValueError, match="conflicts"):
        EvaluationRequest(
            raw_dir=tmp_path / "raw",
            transitions_csv=tmp_path / "transitions.csv",
            checkpoint=tmp_path / "checkpoint.pt",
            config=EvaluationConfig(pf_alg=1),
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )
