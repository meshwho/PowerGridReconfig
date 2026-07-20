import numpy as np
import pytest

from grid_topology_ai.config.physics import PhysicsConfig


def test_numpy_integral_settings_have_canonical_fingerprint() -> None:
    first = PhysicsConfig(pf_alg=np.int64(3), max_iterations=np.int32(30))
    second = PhysicsConfig()
    assert type(first.pf_alg) is int
    assert type(first.max_iterations) is int
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize("value", [True, False])
def test_boolean_algorithm_is_rejected(value: bool) -> None:
    with pytest.raises(ValueError):
        PhysicsConfig(pf_alg=value)


def test_explicit_default_config_conflicts_with_legacy_algorithm(tmp_path) -> None:
    from grid_topology_ai.config.evaluation import EvaluationConfig
    from grid_topology_ai.config.physics import DEFAULT_PHYSICS_CONFIG
    from grid_topology_ai.evaluation.checkpoint import EvaluationRequest

    with pytest.raises(ValueError, match="conflicts"):
        EvaluationRequest(
            raw_dir=tmp_path / "raw",
            transitions_csv=tmp_path / "transitions.csv",
            checkpoint=tmp_path / "checkpoint.pt",
            config=EvaluationConfig(pf_alg=1),
            physics_config=DEFAULT_PHYSICS_CONFIG,
        )
