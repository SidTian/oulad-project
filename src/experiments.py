"""
experiments.py
--------------
Purpose:
- Orchestrate the entire pipeline from raw data to trained models and visualizations
- One-command execution of the full workflow
"""

from src.utils import set_seed
from src.data_loader import load_oulad
from src.preprocess import preprocess_data
from src.feature_engineering import build_features
from src.statistics import compute_descriptive_stats
from src.model import get_models
from src.train import train_and_validate
from src.evaluate import evaluate_model
from src.visualization import (
    plot_confusion_matrix,
    plot_roc,
    plot_feature_importance,
    plot_weekly_trend,
)

import os
import pandas as pd
import json


def run_pipeline(
    raw_dir="data/raw/", processed_dir="data/processed/", figures_dir="figures/"
):
    set_seed(42)  # Ensure reproducibility

    # Step 1: Load raw data
    data = load_oulad(raw_dir)
    student_vle = data["student_vle"]
    vle = data["vle"]
    student_info = data["student_info"]

    # Step 2: Preprocessing & merging
    merged_df = preprocess_data(
        student_vle, vle, student_info, output_dir=processed_dir
    )

    # Step 3: Feature engineering (weekly features + label)
    X, y = build_features(merged_df, output_dir=processed_dir)

    # Step 4: Descriptive statistics
    weekly_df = pd.read_csv(os.path.join(processed_dir, "student_weekly_features.csv"))
    stats = compute_descriptive_stats(weekly_df)
    # Save statistics for reporting
    with open(os.path.join(processed_dir, "stats.json"), "w") as f:
        json.dump(stats, f, default=str, indent=2)

    # Step 5: Get model definitions
    models = get_models()

    # Step 6: Train and cross-validate all models
    results = train_and_validate(models, X, y)

    # Step 7: Evaluation and visualization for each model
    os.makedirs(figures_dir, exist_ok=True)

    for name, res in results.items():
        pipeline = res["pipeline"]
        X_test = res["X_test"]
        y_test = res["y_test"]

        # Evaluate on test set
        metrics, cm = evaluate_model(pipeline, X_test, y_test)

        # Get feature names after preprocessing (for importance plots)
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        # Visualization
        plot_confusion_matrix(
            cm, name, save_path=os.path.join(figures_dir, "confusion_matrix_{}.png")
        )
        plot_roc(
            pipeline,
            X_test,
            y_test,
            name,
            save_path=os.path.join(figures_dir, "roc_curve_{}.png"),
        )
        plot_feature_importance(
            pipeline,
            X_test,
            y_test,
            feature_names,
            name,
            save_path=os.path.join(figures_dir, "feature_importance_{}.png"),
        )

    # Global visualization: weekly engagement trend
    plot_weekly_trend(
        weekly_df, save_path=os.path.join(figures_dir, "weekly_access_trend.png")
    )

    print("Full pipeline completed successfully!")
    print("Check the 'data/processed/' and 'figures/' directories for outputs.")


if __name__ == "__main__":
    run_pipeline()
