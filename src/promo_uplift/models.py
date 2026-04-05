from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )


def _gbdt_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=200, random_state=42)


@dataclass(slots=True)
class FittedUpliftModel:
    name: str
    outcome: str
    score_column: str
    treated_model: object | None = None
    control_model: object | None = None
    joint_model: object | None = None

    def predict_uplift(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        if self.name.startswith("two_model"):
            treated_prob = self.treated_model.predict_proba(frame[features])[:, 1]
            control_prob = self.control_model.predict_proba(frame[features])[:, 1]
            return treated_prob - control_prob

        design = add_treatment_interactions(frame[features], np.ones(len(frame), dtype=int))
        treated_prob = self.joint_model.predict_proba(design)[:, 1]
        design = add_treatment_interactions(frame[features], np.zeros(len(frame), dtype=int))
        control_prob = self.joint_model.predict_proba(design)[:, 1]
        return treated_prob - control_prob

    def predict_treated_response(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        if self.name.startswith("two_model"):
            return self.treated_model.predict_proba(frame[features])[:, 1]
        design = add_treatment_interactions(frame[features], np.ones(len(frame), dtype=int))
        return self.joint_model.predict_proba(design)[:, 1]


def add_treatment_interactions(features: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    design = features.copy()
    design["treatment"] = treatment
    for column in features.columns:
        design[f"{column}_x_treatment"] = features[column] * treatment
    return design


def fit_two_model(
    train_df: pd.DataFrame,
    features: list[str],
    outcome: str,
    base_model: object,
    name: str,
) -> FittedUpliftModel:
    treated_model = clone(base_model)
    control_model = clone(base_model)
    treated_mask = train_df["treatment"] == 1
    control_mask = ~treated_mask
    treated_model.fit(train_df.loc[treated_mask, features], train_df.loc[treated_mask, outcome])
    control_model.fit(train_df.loc[control_mask, features], train_df.loc[control_mask, outcome])
    return FittedUpliftModel(
        name=name,
        outcome=outcome,
        score_column=f"{name}_uplift",
        treated_model=treated_model,
        control_model=control_model,
    )


def fit_interaction_logistic(train_df: pd.DataFrame, features: list[str], outcome: str) -> FittedUpliftModel:
    design = add_treatment_interactions(train_df[features], train_df["treatment"].to_numpy())
    model = _logistic_pipeline()
    model.fit(design, train_df[outcome])
    return FittedUpliftModel(
        name="interaction_logistic",
        outcome=outcome,
        score_column="interaction_logistic_uplift",
        joint_model=model,
    )


def fit_candidate_models(train_df: pd.DataFrame, features: list[str], outcome: str) -> list[FittedUpliftModel]:
    return [
        fit_two_model(train_df, features, outcome, _logistic_pipeline(), "two_model_logistic"),
        fit_two_model(train_df, features, outcome, _gbdt_model(), "two_model_gbdt"),
        fit_interaction_logistic(train_df, features, outcome),
    ]

