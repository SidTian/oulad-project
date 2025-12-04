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
    model_name,
    save_path="figures/feature_importance_{}.png",
):
    """
    Plot feature importance.
    - Tree models (RF/XGB): use built-in feature_importances_
    - Logistic Regression: use permutation importance
    """
    print(f"Generating feature importance plot for {model_name}")
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):  # RF / XGBoost
        print(
            f"{model_name} is a tree-based model → using built-in feature importances"
        )
        print("Number of features:", len(feature_names))
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        importances.plot(kind="bar")
        plt.title(f"{model_name} Feature Importance")
        plt.savefig(save_path.format(model_name))
        plt.close()
    else:  # Logistic Regression
        print(f"{model_name} is Logistic Regression → using permutation importance")
        original_feature_names = X_test.columns.tolist()
        print("Number of original features:", len(original_feature_names))
        print("Sample feature names:", original_feature_names[:5])
        perm = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        print("Permutation importance computed")
        importances = pd.Series(
            perm.importances_mean, index=original_feature_names
        ).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        importances.plot(kind="bar", yerr=perm.importances_std)
        plt.title(f"{model_name} Permutation Importance")
        plt.savefig(save_path.format(model_name))
        plt.close()
    print(f"Feature Importance saved to: {save_path.format(model_name)}")


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
