"""
feature_engineering.py
----------------------
Purpose:
- Convert daily click records into week-level behavioral features
- Generate the target label: whether the student is active in the next consecutive week
"""

import pandas as pd
import os


def build_features(df, output_dir="data/processed/"):
    # Step 1: Create week number and filter out negative dates (pre-course records)
    df = df[df["date"] >= 0].copy()
    df["week"] = df["date"] // 7
    df["week"] = df["week"].clip(lower=0)

    # Aggregate weekly behavioral features
    weekly_behavior = (
        df.groupby(["id_student", "code_module", "code_presentation", "week"])
        .agg(
            sum_click=("sum_click", "sum"),
            num_resources=("id_site", "nunique"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )

    # Compute proportion of clicks per activity type
    type_clicks = (
        df.groupby(
            ["id_student", "code_module", "code_presentation", "week", "activity_type"]
        )["sum_click"]
        .sum()
        .reset_index()
    )
    type_pivot = type_clicks.pivot_table(
        index=["id_student", "code_module", "code_presentation", "week"],
        columns="activity_type",
        values="sum_click",
        fill_value=0,
    ).reset_index()
    type_cols = type_pivot.columns[4:]
    total_clicks = type_pivot[type_cols].sum(axis=1)
    type_pivot[type_cols] = type_pivot[type_cols].div(total_clicks, axis=0).fillna(0)
    type_pivot.columns = ["id_student", "code_module", "code_presentation", "week"] + [
        "resource_type_" + col for col in type_cols
    ]

    # Merge behavioral aggregates with activity-type proportions
    weekly_df = pd.merge(
        weekly_behavior,
        type_pivot,
        on=["id_student", "code_module", "code_presentation", "week"],
        how="left",
    )

    # Step 2: Add static demographic features
    demo_features = df[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "gender",
            "age_band",
            "highest_education",
            "region",
            "studied_credits",
            "disability",
            "imd_band",
        ]
    ].drop_duplicates()
    weekly_df = pd.merge(
        weekly_df,
        demo_features,
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )

    # Step 3: Create target label (1 = active next week, 0 = gap/dropout)
    weekly_df = weekly_df.sort_values(
        ["id_student", "code_module", "code_presentation", "week"]
    )
    weekly_df["next_week"] = weekly_df.groupby(
        ["id_student", "code_module", "code_presentation"]
    )["week"].shift(-1)
    weekly_df["label"] = (
        (weekly_df["next_week"] == weekly_df["week"] + 1).fillna(False)
    ).astype(int)
    weekly_df = weekly_df.drop(columns=["next_week"])

    # Debug: print label distribution
    print("Label distribution:")
    print(weekly_df["label"].value_counts(normalize=True))

    # Step 4: Separate features (X) and target (y)
    resource_cols = [
        col for col in weekly_df.columns if col.startswith("resource_type_")
    ]
    X_cols = (
        ["sum_click", "num_resources", "active_days"]
        + resource_cols
        + [
            "gender",
            "age_band",
            "highest_education",
            "region",
            "studied_credits",
            "disability",
            "imd_band",
        ]
    )
    X = weekly_df[X_cols].copy()
    y = weekly_df["label"].copy()

    # Save weekly feature table
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "student_weekly_features.csv")
    weekly_df.to_csv(output_path, index=False)
    print(f"Weekly features saved to: {output_path}")
    print(f"Weekly table shape: {weekly_df.shape}")

    return X, y
