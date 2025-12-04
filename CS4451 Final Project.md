# CS4451 Final Project

## OULAD Student Engagement Prediction Project

**Early prediction of student dropout / disengagement using weekly behavioral features**

A complete, reproducible end-to-end pipeline built on the Open University Learning Analytics Dataset (OULAD).

## Quick Start (3 commands)

```Bash
# 1. Clone the repository
git clone https://github.com/yourname/oulad-engagement-prediction.git
cd oulad-engagement-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```text
oulad-project/
├── data/
│   ├── raw/                     ← Put only 3 original OULAD files here
│   │   ├── studentInfo.csv
│   │   ├── studentVle.csv
│   │   └── vle.csv
│   └── processed/               ← Auto-generated
│       ├── merged_data.csv
│       ├── student_weekly_features.csv
│       └── stats.json
├── figures/                     ← All plots ready for your paper
├── src/                         ← Clean, fully English, well-commented code
├── config.yaml                  ← (Reserved for future use)
├── requirements.txt
└── README.md
```

## Required Raw Files (place in data/raw/)

You need only these three CSV files from the official OULAD dataset:

- studentInfo.csv
- studentVle.csv
- vle.csv

Download link: https://analyse.kmi.open.ac.uk/open-dataset

## One-Command Full Run

Bash

```
python -m src.experiments
```

This single command will automatically execute the **entire pipeline**:

1. Load raw data
2. Clean and merge tables
3. Build weekly behavioral features + next-week activity label
4. Compute descriptive statistics (saved as data/processed/stats.json)
5. Train three models: Logistic Regression, Random Forest, XGBoost
6. 5-fold CV + hold-out test evaluation
7. Generate and save all figures in the figures/ folder
8. Print evaluation metrics in English

All random seeds are fixed → **100% reproducible results**.

## Output Examples

After running, you will get:

- data/processed/student_weekly_features.csv – final feature table
- data/processed/stats.json – all descriptive statistics for the paper
- figures/confusion_matrix_logreg.png, figures/roc_curve_xgb.png, etc.
- figures/feature_importance_rf.png – ready for the Results section
- figures/weekly_access_trend.png – engagement drop-off curve

## Project Modules (all in English)

| Module                     | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| src/data_loader.py         | Load raw CSV files                         |
| src/preprocess.py          | Merge & clean the three raw tables         |
| src/feature_engineering.py | Create weekly features + next-week label   |
| src/statistics.py          | Descriptive statistics for Methods/Results |
| src/model.py               | Define LogReg / RF / XGBoost               |
| src/train.py               | Train + CV with preprocessing pipeline     |
| src/evaluate.py            | Compute accuracy, F1, ROC-AUC, etc.        |
| src/visualization.py       | Save all required plots                    |
| src/experiments.py         | One-click full pipeline                    |
| src/utils.py               | set_seed for full reproducibility          |

### Dependencies

```Bash
pip install -r requirements.txt
```

## Citation & Dataset

Marta Caro-Martínez et al. (2017) Open University Learning Analytics Dataset (OULAD) https://analyse.kmi.open.ac.uk/open-dataset

## License

MIT License – feel free to use, modify, and share.

------

**Enjoy your research!** If you find this project helpful, a star is greatly appreciated!