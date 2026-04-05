import numpy as np

from promo_uplift.policy import budget_policy_mask, estimate_incremental_policy_value, uplift_curve


def test_budget_policy_mask_targets_expected_share() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    mask = budget_policy_mask(scores, budget=0.5)
    assert mask.sum() == 2


def test_policy_value_positive_for_helpful_policy() -> None:
    y = np.array([1, 0, 0, 0], dtype=float)
    treatment = np.array([1, 0, 1, 0], dtype=float)
    policy = np.array([1, 0, 0, 0], dtype=int)
    value = estimate_incremental_policy_value(
        y=y,
        treatment=treatment,
        policy=policy,
        propensity=0.5,
        conversion_value=10.0,
        treatment_cost=1.0,
    )
    assert value > 0


def test_uplift_curve_returns_finite_area() -> None:
    curve = uplift_curve(
        y=np.array([1, 0, 1, 0], dtype=float),
        treatment=np.array([1, 0, 1, 0], dtype=float),
        score=np.array([0.9, 0.1, 0.8, 0.0], dtype=float),
        propensity=0.5,
    )
    assert np.isfinite(curve.auuc)
