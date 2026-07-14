from scripts.self_play.run_iteration import accept_candidate


POLICY = {
    "metric": "solve_rate",
    "min_improvement": 0.0,
    "max_simple_solve_rate_drop": 0.05,
    "reject_if_failed_scenarios_above": 0,
}


def metrics(
    solve_rate: float,
    *,
    simple: float = 0.50,
    failed: int = 0,
) -> dict[str, float | int]:
    return {
        "solve_rate": solve_rate,
        "solve_rate_simple": simple,
        "failed_scenarios": failed,
    }


def test_accepts_strict_improvement() -> None:
    assert accept_candidate(
        new_metrics=metrics(0.51),
        best_metrics=metrics(0.50),
        policy=POLICY,
    )


def test_rejects_exact_tie() -> None:
    assert not accept_candidate(
        new_metrics=metrics(0.50),
        best_metrics=metrics(0.50),
        policy=POLICY,
    )


def test_rejects_regression() -> None:
    assert not accept_candidate(
        new_metrics=metrics(0.49),
        best_metrics=metrics(0.50),
        policy=POLICY,
    )


def test_rejects_excessive_simple_drop() -> None:
    assert not accept_candidate(
        new_metrics=metrics(0.51, simple=0.74),
        best_metrics=metrics(0.50, simple=0.80),
        policy=POLICY,
    )


def test_rejects_failed_scenarios() -> None:
    assert not accept_candidate(
        new_metrics=metrics(0.51, failed=1),
        best_metrics=metrics(0.50),
        policy=POLICY,
    )