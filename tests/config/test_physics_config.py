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
