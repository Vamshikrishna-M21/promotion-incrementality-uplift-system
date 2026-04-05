from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(slots=True)
class ATEResult:
    outcome: str
    control_rate: float
    treatment_rate: float
    ate: float
    relative_lift: float
    standard_error: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "outcome": self.outcome,
            "control_rate": self.control_rate,
            "treatment_rate": self.treatment_rate,
            "ate": self.ate,
            "relative_lift": self.relative_lift,
            "standard_error": self.standard_error,
            "z_stat": self.z_stat,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def difference_in_means(df: pd.DataFrame, outcome: str, treatment_col: str = "treatment") -> ATEResult:
    treated = df.loc[df[treatment_col] == 1, outcome].to_numpy(dtype=float)
    control = df.loc[df[treatment_col] == 0, outcome].to_numpy(dtype=float)

    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Both treatment and control groups must be present.")

    treatment_rate = treated.mean()
    control_rate = control.mean()
    ate = treatment_rate - control_rate
    var_t = treatment_rate * (1.0 - treatment_rate) / len(treated)
    var_c = control_rate * (1.0 - control_rate) / len(control)
    standard_error = float(np.sqrt(var_t + var_c))
    z_stat = ate / standard_error if standard_error > 0 else 0.0
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    ci_delta = 1.96 * standard_error
    relative_lift = ate / control_rate if control_rate > 0 else np.nan
    return ATEResult(
        outcome=outcome,
        control_rate=float(control_rate),
        treatment_rate=float(treatment_rate),
        ate=float(ate),
        relative_lift=float(relative_lift),
        standard_error=standard_error,
        z_stat=float(z_stat),
        p_value=p_value,
        ci_low=float(ate - ci_delta),
        ci_high=float(ate + ci_delta),
    )


def summarize_experiment(
    df: pd.DataFrame,
    outcomes: list[str],
    treatment_col: str = "treatment",
) -> list[ATEResult]:
    return [difference_in_means(df, outcome=outcome, treatment_col=treatment_col) for outcome in outcomes]

