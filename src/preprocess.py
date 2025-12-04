"""
preprocess.py
-------------
Purpose:
- Clean and merge the three raw OULAD tables (studentVle, vle, studentInfo)
- Produce a clean event-level interaction table ready for feature engineering
"""

import pandas as pd
import os


def preprocess_data(student_vle, vle, student_info, output_dir="data/processed/"):
    """
    Merge and clean the raw datasets into a single event-level dataframe.

    Args:
        student_vle:   DataFrame loaded from studentVle.csv
        vle:           DataFrame loaded from vle.csv
        student_info:  DataFrame loaded from studentInfo.csv
        output_dir:    Directory where the merged file will be saved

    Returns:
        pd.DataFrame: Clean merged interaction table
    """
    # Step 1: Merge student interactions with VLE site metadata
    merged_vle = pd.merge(
        student_vle, vle, on=["code_module", "code_presentation", "id_site"], how="left"
    )

    # Step 2: Merge with student demographic and registration info
    merged_df = pd.merge(
        merged_vle,
        student_info,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    # Step 3: Basic cleaning
    # Convert date to integer and remove pre-course records (negative dates)
    merged_df["date"] = merged_df["date"].astype(int)
    merged_df = merged_df[merged_df["date"] >= 0]

    # Fill missing values with reasonable defaults
    merged_df["disability"] = merged_df["disability"].fillna("N")
    merged_df["imd_band"] = merged_df["imd_band"].fillna("Unknown")
    merged_df["activity_type"] = merged_df["activity_type"].fillna("unknown")

    # Remove exact duplicates if any exist
    merged_df = merged_df.drop_duplicates()

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")
    print(f"Shape after merging and cleaning: {merged_df.shape}")

    return merged_df


def preprocess_tables(student_vle, student_info, vle):
    # TODO: merge tables to produce a single cleaned dataframe
    merged = None
    return merged
