from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CurveResult:
    share_targeted: np.ndarray
    incremental_gain: np.ndarray
    auuc: float


def uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    propensity: float,
) -> CurveResult:
    order = np.argsort(-score)
    y_sorted = y[order]
    t_sorted = treatment[order]
    treated_weight = np.cumsum(y_sorted * t_sorted / propensity)
    control_weight = np.cumsum(y_sorted * (1 - t_sorted) / (1.0 - propensity))
    gain = treated_weight - control_weight
    share = np.arange(1, len(score) + 1) / len(score)
    auuc = float(np.trapezoid(gain / len(score), share))
    return CurveResult(share_targeted=share, incremental_gain=gain / len(score), auuc=auuc)


def qini_coefficient(curve: CurveResult) -> float:
    random_baseline = curve.share_targeted * curve.incremental_gain[-1]
    return float(np.trapezoid(curve.incremental_gain - random_baseline, curve.share_targeted))


def budget_policy_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    n_target = max(1, int(len(scores) * budget))
    threshold_indices = np.argsort(-scores)[:n_target]
    mask = np.zeros(len(scores), dtype=int)
    mask[threshold_indices] = 1
    return mask


def estimate_incremental_policy_value(
    y: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    propensity: float,
    conversion_value: float,
    treatment_cost: float,
) -> float:
    weighted_outcome = (treatment * y / propensity) - ((1 - treatment) * y / (1.0 - propensity))
    reward = policy * (conversion_value * weighted_outcome - treatment_cost)
    return float(reward.mean())


def evaluate_policy_grid(
    df: pd.DataFrame,
    score_columns: dict[str, np.ndarray],
    outcome: str,
    budgets: list[float],
    conversion_value: float,
    treatment_cost: float,
    propensity: float,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    y = df[outcome].to_numpy(dtype=float)
    treatment = df["treatment"].to_numpy(dtype=float)
    records: list[dict[str, float | str]] = []

    for budget in budgets:
        random_scores = rng.random(len(df))
        policy_inputs = {
            "uplift_best": score_columns["uplift_best"],
            "naive_response": score_columns["naive_response"],
            "random": random_scores,
            "treat_none": np.zeros(len(df)),
            "treat_all": np.ones(len(df)),
        }
        for policy_name, scores in policy_inputs.items():
            if policy_name == "treat_all":
                mask = np.ones(len(df), dtype=int)
            elif policy_name == "treat_none":
                mask = np.zeros(len(df), dtype=int)
            else:
                mask = budget_policy_mask(scores, budget)
            value = estimate_incremental_policy_value(
                y=y,
                treatment=treatment,
                policy=mask,
                propensity=propensity,
                conversion_value=conversion_value,
                treatment_cost=treatment_cost,
            )
            records.append({"policy": policy_name, "budget": budget, "incremental_value_per_user": value})

    return pd.DataFrame.from_records(records)
