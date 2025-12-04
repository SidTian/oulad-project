"""
data_loader.py
--------------
Purpose:
- Load the core OULAD CSV files (studentVle, studentInfo, vle)
- Read from the 'data/raw/' directory
- Optionally extendable to load config.yaml in the future
"""

import pandas as pd
import os


def load_oulad(raw_dir):
    # TODO: load studentVle.csv, studentInfo.csv, vle.csv
    student_vle_path = os.path.join(raw_dir, "studentVle.csv")
    student_info_path = os.path.join(raw_dir, "studentInfo.csv")
    vle_path = os.path.join(raw_dir, "vle.csv")

    student_vle = pd.read_csv(student_vle_path)
    student_info = pd.read_csv(student_info_path)
    vle = pd.read_csv(vle_path)

    return {"student_vle": student_vle, "student_info": student_info, "vle": vle}
