"""
Feature Preparation for Module 4 Predictive Modeling.

Builds the modeling matrix from Module 3's cleaned_cohort.csv, with explicit
leakage-prevention decisions documented inline (per Module 4 brief Sec. 2).

PROTECTED ATTRIBUTES (from Module 1 Ethical AI Charter / Module 2 Fairness
Objectives): race, gender, age. These are EXCLUDED from the feature matrix
used to train the model (fairness-through-unawareness is a floor, not a
guarantee -- see fairness_metrics.py for the real audit) but are retained
separately alongside predictions so Fairlearn can measure outcomes by
subgroup after the fact.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

PROCESSED_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "cleaned_cohort.csv"
PROTECTED_ATTRIBUTES = ["race", "gender", "age"]

# Leakage-prone / non-feature columns, with the reason each is excluded
LEAKAGE_DROP = {
    "readmitted": "Original 3-class label readmitted_30d was derived FROM this column -- including it is direct target leakage.",
    "diag_1": "Raw ICD-9 code, superseded by diag_1_category (avoids 717-level high-cardinality leakage/overfit risk).",
    "diag_2": "Same as diag_1.",
    "diag_3": "Same as diag_1.",
    "patient_split_key": "Row identifier used only for the split itself, not a predictive feature.",
}

TARGET = "readmitted_30d"


def load_modeling_frame() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_FILE)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (X, y, protected_df, split_key) where:
      X: encoded, scaled feature matrix (protected attributes excluded)
      y: binary target (readmitted_30d)
      protected_df: race/gender/age, aligned by index to X/y, for fairness auditing
      split_key: patient_split_key, aligned by index, for patient-level splitting
    """
    df = df.copy()

    split_key = df["patient_split_key"]
    protected_df = df[PROTECTED_ATTRIBUTES].copy()
    y = df[TARGET].astype(int)

    drop_cols = list(LEAKAGE_DROP.keys()) + PROTECTED_ATTRIBUTES + [TARGET, "patient_split_key"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Categorical columns -> one-hot encode
    cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
    num_cols = feature_df.select_dtypes(exclude="object").columns.tolist()

    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
    # get_dummies produces bool dtype; cast to int so CSV round-trips as
    # 0/1 rather than the literal strings "True"/"False" (which broke
    # DiceML's counterfactual search when reading X_test.csv back in).
    bool_cols = feature_df.select_dtypes(include="bool").columns
    feature_df[bool_cols] = feature_df[bool_cols].astype(int)

    # Scale numeric columns only (one-hot columns are already 0/1)
    scaler = StandardScaler()
    feature_df[num_cols] = scaler.fit_transform(feature_df[num_cols])

    return feature_df, y, protected_df, split_key, scaler


if __name__ == "__main__":
    df = load_modeling_frame()
    X, y, protected, split_key, scaler = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Target positive rate: {y.mean():.4f}")
    print(f"Protected attributes retained separately: {protected.columns.tolist()}")
    print(f"Dropped for leakage prevention: {list(LEAKAGE_DROP.keys())}")
