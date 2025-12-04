"""
evaluate.py
-----------
Purpose:
- Compute accuracy, precision, recall, F1-score, and ROC-AUC
- Also output the confusion matrix
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate a single trained model on the test set.

    Args:
        pipeline: fitted pipeline returned from train.py
        X_test:   test features
        y_test:   true labels for the test set

    Returns:
        metrics dict and confusion matrix
    """
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f"Evaluation metrics: {metrics}")
    print(f"Confusion Matrix:\n{cm}")

    return metrics, cm
