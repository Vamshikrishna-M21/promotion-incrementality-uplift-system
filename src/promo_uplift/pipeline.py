from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from promo_uplift.ab_testing import summarize_experiment
from promo_uplift.config import load_config
from promo_uplift.data import DatasetPaths, build_criteo_sample, download_file, load_split
from promo_uplift.models import fit_candidate_models
from promo_uplift.plots import save_ate_plot, save_policy_plot, save_uplift_curve
from promo_uplift.policy import evaluate_policy_grid, qini_coefficient, uplift_curve


def _dataset_paths(config: dict) -> DatasetPaths:
    dataset_cfg = config["dataset"]
    return DatasetPaths(
        raw_path=Path(dataset_cfg["raw_path"]),
        processed_dir=Path(dataset_cfg["processed_dir"]),
    )


def _select_best_model(valid_df: pd.DataFrame, models: list, features: list[str], outcome: str) -> tuple[object, pd.DataFrame]:
    propensity = float(valid_df["treatment"].mean())
    leaderboard: list[dict[str, float | str]] = []
    curves = {}
    for model in models:
        uplift_scores = model.predict_uplift(valid_df, features)
        curve = uplift_curve(
            y=valid_df[outcome].to_numpy(dtype=float),
            treatment=valid_df["treatment"].to_numpy(dtype=float),
            score=uplift_scores,
            propensity=propensity,
        )
        curves[model.name] = curve
        leaderboard.append(
            {
                "model": model.name,
                "auuc": curve.auuc,
                "qini": qini_coefficient(curve),
            }
        )
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("auuc", ascending=False).reset_index(drop=True)
    best_name = leaderboard_df.iloc[0]["model"]
    best_model = next(model for model in models if model.name == best_name)
    return best_model, leaderboard_df


def run_pipeline(config_path: str | Path) -> dict:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    modeling_cfg = config["modeling"]
    policy_cfg = config["policy"]
    reporting_cfg = config["reporting"]

    paths = _dataset_paths(config)
    raw_path = download_file(dataset_cfg["url"], paths.raw_path)
    sample_paths = build_criteo_sample(
        raw_path=raw_path,
        output_dir=paths.processed_dir,
        sample_size=dataset_cfg["sample_size"],
        total_rows_hint=dataset_cfg["total_rows_hint"],
        chunk_size=dataset_cfg["chunk_size"],
        seed=dataset_cfg["random_seed"],
    )

    train_df = load_split(sample_paths.train_path)
    valid_df = load_split(sample_paths.valid_path)
    test_df = load_split(sample_paths.test_path)
    features = modeling_cfg["features"]
    outcome = modeling_cfg["outcome"]

    experiment_summary = summarize_experiment(test_df, outcomes=[outcome, modeling_cfg["visit_outcome"]])
    experiment_df = pd.DataFrame([result.as_dict() for result in experiment_summary])

    models = fit_candidate_models(train_df, features=features, outcome=outcome)
    best_model, leaderboard_df = _select_best_model(valid_df, models, features=features, outcome=outcome)

    propensity = float(test_df["treatment"].mean())
    test_curves = {}
    for model in models:
        test_curves[model.name] = uplift_curve(
            y=test_df[outcome].to_numpy(dtype=float),
            treatment=test_df["treatment"].to_numpy(dtype=float),
            score=model.predict_uplift(test_df, features),
            propensity=propensity,
        )

    best_uplift_scores = best_model.predict_uplift(test_df, features)
    naive_scores = best_model.predict_treated_response(test_df, features)
    policy_df = evaluate_policy_grid(
        df=test_df,
        score_columns={
            "uplift_best": best_uplift_scores,
            "naive_response": naive_scores,
        },
        outcome=outcome,
        budgets=policy_cfg["budgets"],
        conversion_value=policy_cfg["conversion_value"],
        treatment_cost=policy_cfg["treatment_cost"],
        propensity=propensity,
        random_seed=dataset_cfg["random_seed"],
    )

    artifact_dir = Path(reporting_cfg["output_dir"])
    plots_dir = artifact_dir / "figures"
    metrics_dir = artifact_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_ate_plot(experiment_df, plots_dir / "ate_summary.png")
    save_uplift_curve(test_curves, plots_dir / "uplift_curves.png")
    save_policy_plot(policy_df, plots_dir / "policy_value.png")

    summary = {
        "config_path": str(config_path),
        "sample_size": int(len(train_df) + len(valid_df) + len(test_df)),
        "propensity": propensity,
        "best_model": str(best_model.name),
        "ate_summary": experiment_df.to_dict(orient="records"),
        "model_leaderboard": leaderboard_df.to_dict(orient="records"),
        "policy_results": policy_df.to_dict(orient="records"),
    }

    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report_lines = [
        "# Promotion Incrementality Report",
        "",
        f"- Sample size: {summary['sample_size']:,}",
        f"- Test-set treatment propensity: {propensity:.4f}",
        f"- Best uplift model by validation AUUC: {best_model.name}",
        "",
        "## Randomized experiment results",
    ]
    for row in summary["ate_summary"]:
        report_lines.append(
            f"- {row['outcome']}: ATE={row['ate']:.6f}, CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}], p={row['p_value']:.4g}"
        )
    report_lines.extend(["", "## Model leaderboard"])
    for row in summary["model_leaderboard"]:
        report_lines.append(f"- {row['model']}: AUUC={row['auuc']:.6f}, Qini={row['qini']:.6f}")

    with (artifact_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    return summary
