"""
Explainable AI -- Global (SHAP) (Module 4 Sec. 6).
Uses the best-performing model (XGBoost, highest ROC-AUC and recall@20%)
to compute SHAP values on the held-out test set, producing a global
feature-importance summary plot and per-prediction local explanations.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_NAME = "xgboost"


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()
    return model, X_test, y_test


def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def global_feature_importance(shap_values, X_test, top_n=10) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance.head(top_n)


def plot_shap_summary(shap_values, X_test):
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.tight_layout()
    out = REPORTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_local_explanation(explainer, shap_values, X_test, row_idx, tag):
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[row_idx].values,
        feature_names=X_test.columns.tolist(),
    )
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    out = REPORTS_DIR / f"shap_local_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    matplotlib.use("Agg")  # headless backend for standalone CLI runs only;
                            # NOT set at import time so notebooks importing
                            # this module keep their own inline backend.
    model, X_test, y_test = load_artifacts()
    explainer, shap_values = compute_shap_values(model, X_test)

    top_features = global_feature_importance(shap_values, X_test)
    print("Top 10 global features by mean |SHAP value|:")
    print(top_features.to_string(index=False))
    top_features.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)

    summary_path = plot_shap_summary(shap_values, X_test)
    print(f"\nSaved: {summary_path}")

    # Local explanations: one high-risk, one low-risk prediction
    y_proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(y_proba))
    low_risk_idx = int(np.argmin(y_proba))

    high_path = plot_local_explanation(explainer, shap_values, X_test, high_risk_idx, "high_risk")
    low_path = plot_local_explanation(explainer, shap_values, X_test, low_risk_idx, "low_risk")
    print(f"Saved: {high_path} (patient risk={y_proba[high_risk_idx]:.3f})")
    print(f"Saved: {low_path} (patient risk={y_proba[low_risk_idx]:.3f})")
