from __future__ import annotations

import numpy as np
import pytest

from grid_topology_ai.training.checkpoints import extract_normalization_stats


def _payload() -> dict[str, object]:
    return {
        "bus_feature_mean": [1.0, 2.0],
        "bus_feature_std": [3.0, 4.0],
        "branch_feature_mean": [5.0, 6.0, 7.0],
        "branch_feature_std": [8.0, 9.0, 10.0],
    }


def test_extract_normalization_stats_returns_float32_copies() -> None:
    payload = _payload()
    stats = extract_normalization_stats(payload, source="init.pt")
    assert all(value.dtype == np.float32 for value in stats.values())
    np.testing.assert_array_equal(stats["bus_feature_mean"], np.array([1.0, 2.0], dtype=np.float32))
    stats["bus_feature_mean"][0] = 99.0
    assert payload["bus_feature_mean"][0] == 1.0  # type: ignore[index]


def test_missing_normalization_key_is_rejected() -> None:
    payload = _payload()
    del payload["branch_feature_std"]
    with pytest.raises(ValueError, match="branch_feature_std.*init.pt"):
        extract_normalization_stats(payload, source="init.pt")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_normalization_is_rejected(bad: float) -> None:
    payload = _payload()
    payload["bus_feature_mean"] = [1.0, bad]
    with pytest.raises(ValueError, match="finite"):
        extract_normalization_stats(payload, source="init.pt")


def test_zero_std_is_rejected() -> None:
    payload = _payload(); payload["bus_feature_std"] = [1.0, 0.0]
    with pytest.raises(ValueError, match="strictly positive"):
        extract_normalization_stats(payload, source="init.pt")


def test_negative_std_is_rejected() -> None:
    payload = _payload(); payload["branch_feature_std"] = [1.0, -1.0, 2.0]
    with pytest.raises(ValueError, match="strictly positive"):
        extract_normalization_stats(payload, source="init.pt")


def test_mean_std_shape_mismatch_is_rejected() -> None:
    payload = _payload(); payload["bus_feature_std"] = [1.0]
    with pytest.raises(ValueError, match="shape"):
        extract_normalization_stats(payload, source="init.pt")


def test_multidimensional_stats_are_rejected() -> None:
    payload = _payload(); payload["bus_feature_mean"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="1D"):
        extract_normalization_stats(payload, source="init.pt")


def test_empty_stats_are_rejected() -> None:
    payload = _payload(); payload["bus_feature_mean"] = []
    with pytest.raises(ValueError, match="empty"):
        extract_normalization_stats(payload, source="init.pt")


def test_extraction_does_not_mutate_checkpoint_payload() -> None:
    payload = _payload()
    before = {key: list(value) for key, value in payload.items()}  # type: ignore[arg-type]
    extract_normalization_stats(payload, source="init.pt")
    assert payload == before
