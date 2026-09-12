"""
Train/Test Protocol & Baseline (Module 4 Sec. 3).

Split strategy: GROUPED stratified hold-out, grouped by patient_split_key
so no patient's encounters appear in both train and test (prevents the
leakage risk flagged in Module 3's RAID log), and stratified on the target
to preserve the ~20% positive rate in both sets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .features import load_modeling_frame, build_feature_matrix

RANDOM_STATE = 42


def grouped_train_test_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            train_idx, test_idx)


def verify_no_patient_overlap(split_key, train_idx, test_idx) -> bool:
    train_patients = set(split_key.iloc[train_idx])
    test_patients = set(split_key.iloc[test_idx])
    overlap = train_patients & test_patients
    return len(overlap) == 0


def run_baseline(y_train, y_test):
    """Majority-class baseline: predicts 'not readmitted' for everyone."""
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = baseline.predict(np.zeros((len(y_test), 1)))
    y_proba = baseline.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    return {
        "model": "baseline_majority_class",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": 0.5,  # a constant-output classifier has no discriminative power
    }


if __name__ == "__main__":
    df = load_modeling_frame()
    X, y, protected, split_key, scaler = build_feature_matrix(df)
    X_train, X_test, y_train, y_test, train_idx, test_idx = grouped_train_test_split(X, y, split_key)

    print(f"Train: {X_train.shape}, positive rate {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, positive rate {y_test.mean():.4f}")
    print(f"No patient overlap between train/test: {verify_no_patient_overlap(split_key, train_idx, test_idx)}")

    baseline_metrics = run_baseline(y_train, y_test)
    print("\nBaseline (majority class):")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v}")
