"""
Performance Evaluation & Validation artifacts (Module 4 Sec. 5).
Generates confusion matrix and ROC curve figures for all 3 models plus
baseline, and a comparison table, saved to reports/ for the Model
Validation Report.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
COLORS = {"logistic_regression": "#028090", "random_forest": "#00A896", "xgboost": "#01353D"}


def load_test_artifacts():
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()
    protected_test = pd.read_csv(MODEL_DIR / "protected_test.csv")
    return X_test, y_test, protected_test


def plot_confusion_matrices(X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, MODEL_NAMES):
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not readmitted", "Readmitted <30d"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name.replace("_", " ").title())
    plt.tight_layout()
    out = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_roc_curves(X_test, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in MODEL_NAMES:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={roc_auc:.3f})", color=COLORS[name], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 30-Day Readmission Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = REPORTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


if __name__ == "__main__":
    X_test, y_test, protected_test = load_test_artifacts()
    cm_path = plot_confusion_matrices(X_test, y_test)
    roc_path = plot_roc_curves(X_test, y_test)
    print(f"Saved: {cm_path}")
    print(f"Saved: {roc_path}")
