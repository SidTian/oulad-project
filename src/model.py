"""
model.py
--------
Purpose:
- Define the classification models used in the project (Logistic Regression, Random Forest, XGBoost)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    """
    Return a dictionary of model instances (unfitted).

    Returns:
        dict: {model_name: estimator}
    """
    models = {
        "logreg": LogisticRegression(
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
            max_iter=1000,
        ),
        "rf": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=100,
            max_depth=10,  # Prevent overfitting
        ),
        "xgb": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=1,
            n_estimators=100,
        ),
    }
    return models
