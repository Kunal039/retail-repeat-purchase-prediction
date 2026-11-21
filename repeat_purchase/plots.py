"""Figures for exploratory analysis and model evaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import NUMERIC_FEATURES, TARGET

PALETTE = ["#9aa5b1", "#2f6db5"]
sns.set_theme(style="whitegrid")


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def eda(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sns.countplot(x=TARGET, data=df, ax=ax[0], palette=PALETTE, hue=TARGET, legend=False)
    ax[0].set_title("Class balance (repeat purchase in 30 days)")
    sns.boxplot(x=TARGET, y="visits_per_month", data=df, ax=ax[1], palette=PALETTE, hue=TARGET, legend=False)
    ax[1].set_title("Visits per month vs target")
    sns.barplot(x="membership", y=TARGET, data=df, ax=ax[2], order=["None", "Silver", "Gold"], color=PALETTE[1])
    ax[2].set_title("Repeat-purchase rate by membership")
    _save(fig, out / "eda.png")

    fig, ax = plt.subplots(figsize=(8, 6.2))
    sns.heatmap(df[NUMERIC_FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": .8})
    ax.set_title("Correlation matrix of numerical features")
    _save(fig, out / "correlation.png")


def model_comparison(results: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    results.set_index("Model")[["Accuracy", "F1-Score", "ROC-AUC"]].plot(
        kind="bar", ax=ax, color=["#9aa5b1", "#5b8fd1", "#1f4e8c"], width=0.75)
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.set_title("Model comparison on held-out test set")
    plt.xticks(rotation=25, ha="right"); ax.legend(loc="lower right")
    _save(fig, out / "model_comparison.png")


def confusion_and_roc(final, y_pred, curves: dict, X_test, y_test, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0], cmap="Blues", colorbar=False,
                                            display_labels=["No repeat", "Repeat"])
    ax[0].set_title("Confusion matrix - final model"); ax[0].grid(False)
    for name, est in curves.items():
        RocCurveDisplay.from_estimator(est, X_test, y_test, ax=ax[1], name=name)
    RocCurveDisplay.from_estimator(final, X_test, y_test, ax=ax[1], name="Final Logistic Regression")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_title("ROC curves")
    _save(fig, out / "confusion_roc.png")


def feature_importance(importance: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    importance.sort_values().plot(kind="barh", ax=ax, color=PALETTE[1])
    ax.set_title("Top-10 feature importances (tuned Random Forest)")
    _save(fig, out / "feature_importance.png")
