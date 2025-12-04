"""
visualization.py
----------------
Purpose:
- Generate and save key evaluation plots:
  • Confusion Matrix
  • ROC Curve
  • Feature Importance (tree-based: built-in, LogReg: permutation)
  • Weekly engagement trend
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.inspection import permutation_importance  # for LogReg importance


def plot_confusion_matrix(cm, model_name, save_path="confusion_matrix_{}.png"):
    """
    Plot and save the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path.format(model_name))
    plt.close()
    print(f"Confusion Matrix saved to: {save_path.format(model_name)}")


def plot_roc(pipeline, X_test, y_test, model_name, save_path="roc_curve_{}.png"):
    """
    Plot and save the ROC curve (only if probability predictions are available).
    """
    if hasattr(pipeline, "predict_proba"):
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
        plt.title(f"{model_name} ROC Curve")
        plt.savefig(save_path.format(model_name))
        plt.close()
        print(f"ROC Curve saved to: {save_path.format(model_name)}")
    else:
        print(f"{model_name} does not support predict_proba → skipping ROC plot")


def plot_feature_importance(
    pipeline,
    X_test,
    y_test,
    feature_names,
    model_name: str,
    save_path: str = "figures/feature_importance_{}.png",
    top_n: int = 15,
    max_label_length: int = 50,
):
    """
    Generate a clean, publication-ready feature importance plot.
    - Tree models (RF/XGB): use built-in Gini importance
    - Logistic Regression: use permutation importance with error bars
    - Long feature names are intelligently truncated
    - Horizontal bars for perfect readability
    """
    print(f"Generating feature importance plot for {model_name} (top {top_n})")
    model = pipeline.named_steps["model"]

    # === 1. Get feature importances ===
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = pd.Series(model.feature_importances_, index=feature_names)
        std_aligned = None
        title_suffix = "Feature Importance (Gini)"
    else:
        # Logistic Regression → permutation importance
        print(f"  → Computing permutation importance for {model_name}...")
        perm = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = pd.Series(perm.importances_mean, index=X_test.columns)
        # Align standard deviation to the sorted top_n features
        std_aligned = (
            pd.Series(perm.importances_std, index=X_test.columns)
            .loc[importances.index[:top_n]]
            .values
        )
        title_suffix = "Permutation Importance"

    # === 2. Select top N and sort ===
    importances = importances.sort_values(ascending=False).head(top_n)

    # === 3. Shorten long feature names ===
    def shorten_label(label: str) -> str:
        return (
            label
            if len(label) <= max_label_length
            else label[: max_label_length - 3] + "..."
        )

    labels = [shorten_label(lab) for lab in importances.index]

    # === 4. Plot horizontal bar chart ===
    plt.figure(figsize=(11, max(6.5, top_n * 0.48)))
    y_pos = range(len(importances))

    # Main bars
    bars = plt.barh(y_pos, importances.values, color="#4c72b0", alpha=0.85, height=0.6)

    # Error bars (only for LogReg) — beautiful and non-intrusive
    if std_aligned is not None:
        plt.errorbar(
            importances.values,
            y_pos,
            xerr=std_aligned,
            fmt="none",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            alpha=0.7,
            color="black",
        )

    # === 5. Styling ===
    plt.yticks(y_pos, labels, fontsize=11)
    plt.xlabel("Importance", fontsize=12)
    plt.title(
        f"{model_name.upper()} - Top {top_n} {title_suffix}",
        fontsize=14,
        pad=20,
        fontweight="bold",
    )
    plt.gca().invert_yaxis()  # Most important at the top
    plt.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
    plt.tight_layout()

    # === 6. Save ===
    path = save_path.format(model_name)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Feature importance plot saved: {path}")


# Optional plot: weekly engagement trend
def plot_weekly_trend(weekly_df, save_path="weekly_access_trend.png"):
    trend = weekly_df.groupby("week")["label"].mean()
    plt.figure(figsize=(8, 5))
    trend.plot()
    plt.title("Weekly Access Trend (Mean Label)")
    plt.xlabel("Week")
    plt.ylabel("Mean Continued Access")
    plt.savefig(save_path)
    plt.close()
    print(f"Weekly Trend saved to: {save_path}")
