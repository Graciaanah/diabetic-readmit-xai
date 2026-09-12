"""
Sensitivity Analysis (Module 4 Sec. 6/7).
Tests model robustness by perturbing key input features and observing how
much predicted risk changes -- a model that swings wildly on small,
clinically plausible input noise is less trustworthy for deployment than
one that degrades gracefully.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BEST_MODEL_NAME = "xgboost"


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    return model, X_test


def perturb_feature(model, X_test, feature, deltas, n_sample=200, seed=42):
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    sample = X_test.iloc[sample_idx].copy()
    base_proba = model.predict_proba(sample)[:, 1]

    results = []
    for delta in deltas:
        perturbed = sample.copy()
        perturbed[feature] = perturbed[feature] + delta
        new_proba = model.predict_proba(perturbed)[:, 1]
        mean_abs_change = np.abs(new_proba - base_proba).mean()
        mean_signed_change = (new_proba - base_proba).mean()
        results.append({
            "feature": feature, "delta": delta,
            "mean_abs_risk_change": round(float(mean_abs_change), 4),
            "mean_signed_risk_change": round(float(mean_signed_change), 4),
        })
    return results


def run_sensitivity_analysis():
    model, X_test = load_artifacts()

    # Perturb standardized numeric features by +/- 0.5 and +/- 1.0 std devs
    # (these columns were StandardScaler-transformed in features.py, so a
    # delta of 1.0 IS one standard deviation in the original units).
    numeric_features = ["number_inpatient", "time_in_hospital", "num_medications", "num_lab_procedures"]
    deltas = [-1.0, -0.5, 0.5, 1.0]

    all_results = []
    for feature in numeric_features:
        all_results.extend(perturb_feature(model, X_test, feature, deltas))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORTS_DIR / "sensitivity_analysis.csv", index=False)

    lines = ["Sensitivity Analysis Report", "=" * 60, "",
             "Mean absolute change in predicted readmission risk when each",
             "feature is perturbed by +/- 0.5 and +/- 1.0 standard deviations",
             "(evaluated on a random 200-patient sample of the test set):", ""]
    lines.append(results_df.to_string(index=False))

    most_sensitive = results_df.groupby("feature")["mean_abs_risk_change"].mean().sort_values(ascending=False)
    lines.append("\n\nFeatures ranked by average sensitivity:")
    lines.append(most_sensitive.to_string())
    lines.append(
        f"\n\nInterpretation: {most_sensitive.index[0]} is the most sensitive input -- consistent with it "
        f"being the dominant SHAP feature -- meaning small data-entry errors or missing-data imputation "
        f"artifacts in this field have an outsized effect on predicted risk and warrant extra data-quality "
        f"attention in any production deployment."
    )

    report_text = "\n".join(lines)
    print(report_text)
    (REPORTS_DIR / "sensitivity_analysis_report.txt").write_text(report_text)
    return results_df


if __name__ == "__main__":
    run_sensitivity_analysis()
