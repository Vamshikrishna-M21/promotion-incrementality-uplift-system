from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from promo_uplift.policy import CurveResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_ate_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary_df["outcome"], summary_df["ate"], color=["#1f77b4", "#ff7f0e"])
    ax.errorbar(
        summary_df["outcome"],
        summary_df["ate"],
        yerr=[
            summary_df["ate"] - summary_df["ci_low"],
            summary_df["ci_high"] - summary_df["ate"],
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Randomized Incremental Lift")
    ax.set_ylabel("Absolute treatment effect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_uplift_curve(curves: dict[str, CurveResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in curves.items():
        ax.plot(curve.share_targeted, curve.incremental_gain, label=f"{name} (AUUC={curve.auuc:.4f})")
    ax.set_title("Uplift Curves on Holdout Data")
    ax.set_xlabel("Share of users targeted")
    ax.set_ylabel("Incremental conversions per user")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_policy_plot(policy_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy_name, group in policy_df.groupby("policy"):
        ax.plot(group["budget"], group["incremental_value_per_user"], marker="o", label=policy_name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("ROI-Aware Policy Value by Budget")
    ax.set_xlabel("Targeting budget share")
    ax.set_ylabel("Incremental value per eligible user")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
