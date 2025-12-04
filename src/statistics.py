"""
statistics.py
-------------
Purpose:
- Compute descriptive statistics required for the Methods/Results section of the report
- Includes dataset overview, behavioral patterns, demographic distributions, and label balance
- Returns a comprehensive dictionary for easy use in notebooks or reports
"""

import pandas as pd


def compute_descriptive_stats(df):
    """
    Calculate descriptive statistics on the weekly feature dataframe.

    Args:
        df: Weekly-level feature DataFrame (student_weekly_features.csv)

    Returns:
        dict: Dictionary containing all computed statistics
    """
    stats = {}

    # Dataset overview
    stats["num_rows"] = df.shape[0]  # Total number of weekly records
    stats["num_students"] = df["id_student"].nunique()  # Number of unique students
    stats["num_courses"] = df["code_module"].nunique()  # Number of courses
    stats["num_presentations"] = df[
        "code_presentation"
    ].nunique()  # Number of course presentations

    # Behavioral feature summaries
    stats["click_stats"] = df["sum_click"].describe()  # Includes mean, median, etc.
    stats["active_days_stats"] = df["active_days"].describe()
    stats["num_resources_stats"] = df["num_resources"].describe()

    # Engagement over time
    stats["weeks_per_student"] = (
        df.groupby("id_student")["week"].nunique().mean()
    )  # Average number of active weeks per student
    stats["max_week"] = df["week"].max()
    stats["min_week"] = df["week"].min()

    # Target label distribution (%)
    stats["label_distribution"] = (
        df["label"].value_counts(normalize=True) * 100
    )  # Percentage of continued vs. non-continued weeks

    # Demographic distributions (%)
    stats["gender_distribution"] = df["gender"].value_counts(normalize=True) * 100
    stats["age_band_distribution"] = df["age_band"].value_counts(normalize=True) * 100
    stats["education_distribution"] = (
        df["highest_education"].value_counts(normalize=True) * 100
    )
    stats["region_distribution"] = df["region"].value_counts(normalize=True) * 100
    stats["disability_distribution"] = (
        df["disability"].value_counts(normalize=True) * 100
    )

    # Behavioral differences across groups
    stats["click_by_age"] = df.groupby("age_band")["sum_click"].mean()
    stats["click_by_education"] = df.groupby("highest_education")["sum_click"].mean()
    stats["access_rate_by_region"] = df.groupby("region")[
        "label"
    ].mean()  # Next-week continuation rate

    # Average proportion of clicks by resource type
    resource_cols = [col for col in df.columns if col.startswith("resource_type_")]
    stats["resource_type_means"] = df[resource_cols].mean().sort_values(ascending=False)

    # Print results for quick inspection
    print("Descriptive statistics computed successfully:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats
