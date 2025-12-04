"""
train.py
--------
Purpose:
- Train all models with stratified train/test split and 5-fold cross-validation
- Return fitted pipelines, CV scores, and hold-out test sets for evaluation
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import set_seed


def train_and_validate(models, X, y):
    set_seed(42)

    # Sanity check: ensure both classes are present
    if len(y.unique()) < 2:
        raise ValueError(
            "y contains only one class. Fix labels in feature_engineering.py."
        )

    # Preprocessing: scale numerical features, one-hot encode categorical features
    categorical_cols = [
        "gender",
        "age_band",
        "highest_education",
        "region",
        "disability",
        "imd_band",
    ]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Additional warning if only one class exists after split (should not happen with stratify)
    if len(y.unique()) < 2:
        print("Warning: y contains only one class after split. Check label generation.")
        return {}

    # Train each model with preprocessing pipeline and perform 5-fold CV
    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        results[name] = {
            "pipeline": pipeline,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "X_test": X_test,
            "y_test": y_test,
        }
        print(f"{name} CV F1-macro: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return results
