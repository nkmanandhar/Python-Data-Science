"""Aggregate EDA and diagnostic figures; never publish student-level predictions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from .features import FEATURES


def save(out, name):
    plt.tight_layout()
    plt.savefig(out / name, dpi=150)
    plt.close("all")


def eda(d, out):
    summary = d.groupby(["code_presentation", "code_module"]).agg(
        enrolments=("target", "size"), risk_rate=("target", "mean"),
        mean_clicks=("clicks", "mean"), zero_click_rate=("clicks", lambda x: x.eq(0).mean()))
    summary.to_csv(out / "cohort_summary.csv")
    d.isna().mean().rename("missing_fraction").to_csv(out / "feature_missingness.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.crosstab(d.code_presentation, d.final_result, normalize="index").plot.bar(stacked=True, ax=ax)
    ax.set(ylabel="Fraction of eligible enrolments", title="Day-28 cohort outcomes")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(out, "outcomes.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in d.groupby("target"):
        ax.hist(np.log1p(group.clicks), bins=40, alpha=0.55, density=True,
                label="Fail/Withdrawn" if label else "Pass/Distinction")
    ax.set(xlabel="log(1 + clicks), days 0–28", ylabel="Density", title="Early activity and later outcomes")
    ax.legend()
    save(out, "activity.png")
    plt.figure(figsize=(8, 5))
    heat = d.groupby(["code_module", "code_presentation"]).target.mean().unstack()
    plt.imshow(heat, vmin=0, vmax=1, aspect="auto", cmap="YlOrRd")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat)), heat.index)
    plt.colorbar(label="Fail/Withdrawn rate")
    plt.title("Outcome variation across modules and presentations")
    save(out, "cohort_heatmap.png")


def diagnostics(model, test, p, threshold, audit, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    PrecisionRecallDisplay.from_predictions(test.target, p, ax=axes[0, 0])
    RocCurveDisplay.from_predictions(test.target, p, ax=axes[0, 1])
    CalibrationDisplay.from_predictions(test.target, p, n_bins=10, strategy="quantile", ax=axes[1, 0])
    ConfusionMatrixDisplay.from_predictions(test.target, p >= threshold, ax=axes[1, 1], colorbar=False)
    save(out, "evaluation.png")
    importance = permutation_importance(model, test[FEATURES], test.target,
                                        scoring="average_precision", n_repeats=10, random_state=42, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "ap_decrease": importance.importances_mean,
                  "std": importance.importances_std}).sort_values("ap_decrease", ascending=False).to_csv(
                      out / "permutation_importance.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, attr in zip(axes, ["gender", "age_band", "disability"]):
        sub = audit.loc[audit.attribute.eq(attr) & ~audit.suppressed]
        ax.bar(sub.group, sub.recall, color="#246b8e")
        ax.set(title=attr, ylim=(0, 1), ylabel="Recall for risk class")
    save(out, "fairness.png")
