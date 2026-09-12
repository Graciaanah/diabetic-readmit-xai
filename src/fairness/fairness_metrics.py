"""
Fairness Metrics (Module 4 Sec. 6).
Operationalizes the Fairness Objectives from the Module 1/2 documents using
Fairlearn, on the best model's predictions against the held-out test set.
This is a MODEL-level fairness audit -- distinct from Module 3's
representation-bias check, which looked at the raw cohort filtering step
before any model existed.

Metrics: Demographic Parity Difference/Ratio, Equalized Odds Difference,
Disparate Impact Ratio, evaluated per protected attribute (race, gender,
age), with k=5 suppression applied to any subgroup breakdown per the
Module 1 Privacy by Design principle.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate, true_positive_rate, false_positive_rate,
    count,
)
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_NAME = "xgboost"
K_THRESHOLD = 5  # consistent with Module 1/3 k-anonymity threshold
PROTECTED_ATTRIBUTES = ["race", "gender", "age"]


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()
    protected_test = pd.read_csv(MODEL_DIR / "protected_test.csv")
    return model, X_test, y_test, protected_test


def disparate_impact_ratio(y_pred, sensitive_feature) -> pd.Series:
    """
    Disparate impact ratio: selection rate of each subgroup relative to the
    subgroup with the HIGHEST selection rate (the "80% rule" convention --
    a ratio below 0.8 is a common regulatory red flag, e.g. EEOC guidance).
    """
    mf = MetricFrame(metrics=selection_rate, y_true=y_pred, y_pred=y_pred, sensitive_features=sensitive_feature)
    rates = mf.by_group
    max_rate = rates.max()
    if max_rate == 0:
        return rates * 0
    return (rates / max_rate).round(4)


def suppress_small_groups(series_or_frame, group_counts, k=K_THRESHOLD):
    """Apply k-anonymity suppression: any subgroup with <k test-set records is masked."""
    small = group_counts < k
    out = series_or_frame.copy()
    if isinstance(out, pd.DataFrame):
        out[small] = f"<{k} records (suppressed)"
    else:
        out[small] = np.nan
    return out


def audit_attribute(y_test, y_pred, y_proba, protected_series, attr_name):
    counts = protected_series.value_counts()

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "recall_tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "count": count,
        },
        y_true=y_test, y_pred=y_pred, sensitive_features=protected_series,
    )
    by_group = mf.by_group.copy()
    by_group["count"] = protected_series.value_counts().reindex(by_group.index)

    # k-anonymity suppression on the by-group breakdown
    small_mask = by_group["count"] < K_THRESHOLD
    by_group_report = by_group.copy()
    by_group_report.loc[small_mask, ["selection_rate", "recall_tpr", "fpr"]] = np.nan
    by_group_report["count"] = by_group_report["count"].astype(object)
    by_group_report.loc[small_mask, "count"] = f"<{K_THRESHOLD}"

    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=protected_series)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=protected_series)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=protected_series)
    di_ratio = disparate_impact_ratio(y_pred, protected_series)
    di_ratio_report = di_ratio.copy()
    di_ratio_report[small_mask] = np.nan

    summary = {
        "attribute": attr_name,
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_ratio": round(dp_ratio, 4),
        "equalized_odds_difference": round(eo_diff, 4),
        "min_disparate_impact_ratio": round(float(di_ratio_report.min(skipna=True)), 4) if di_ratio_report.notna().any() else None,
        "flag_disparate_impact_below_0.8": bool((di_ratio_report < 0.8).any()),
    }
    return summary, by_group_report, di_ratio_report


def run_fairness_audit():
    model, X_test, y_test, protected_test = load_artifacts()
    y_pred_default = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # PRIMARY decision rule audited: the top-20% risk-ranked shortlist --
    # this is what the deployed tool actually does (Module 1 Goal 2), not
    # a raw 0.5 probability threshold. Auditing the 0.5-threshold decision
    # turned out to be close to vacuous here: the model crosses 0.5 so
    # rarely (see Module 4 train.py results, ~1.9% raw recall) that most
    # subgroups show a selection rate of exactly zero, making demographic
    # parity/disparate impact numbers an artifact of threshold choice
    # rather than a meaningful fairness signal.
    k = int(np.ceil(len(y_test) * 0.20))
    threshold_20pct = np.sort(y_proba)[::-1][k - 1]
    y_pred_shortlist = (y_proba >= threshold_20pct).astype(int)

    report_lines = ["Fairness Metrics Report", "=" * 60, ""]
    report_lines.append(f"Model audited: {BEST_MODEL_NAME}")
    report_lines.append(f"Test set size: {len(y_test)}")
    report_lines.append(f"k-anonymity suppression threshold: {K_THRESHOLD}")
    report_lines.append(
        f"\nPRIMARY decision rule: top-20% risk-ranked shortlist (threshold={threshold_20pct:.4f}), "
        f"matching the deployed tool's actual capacity-constrained follow-up list, not a raw 0.5 cutoff.\n"
    )

    summaries = []
    for attr in PROTECTED_ATTRIBUTES:
        summary, by_group, di_ratio = audit_attribute(y_test, y_pred_shortlist, y_proba, protected_test[attr], attr)
        summaries.append(summary)
        report_lines.append(f"--- {attr} (shortlist decision) ---")
        report_lines.append(f"Demographic Parity Difference: {summary['demographic_parity_difference']}")
        report_lines.append(f"Demographic Parity Ratio: {summary['demographic_parity_ratio']}")
        report_lines.append(f"Equalized Odds Difference: {summary['equalized_odds_difference']}")
        report_lines.append(f"Minimum Disparate Impact Ratio: {summary['min_disparate_impact_ratio']}")
        report_lines.append(f"Flagged (disparate impact < 0.8): {summary['flag_disparate_impact_below_0.8']}")
        report_lines.append("\nPer-subgroup breakdown (selection rate, TPR/recall, FPR, count):")
        report_lines.append(by_group.to_string())
        report_lines.append("")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(REPORTS_DIR / "fairness_metrics_summary.csv", index=False)

    # SECONDARY / supplementary: default 0.5-threshold decision, included
    # for completeness against the brief's literal ask, with the caveat
    # above documented rather than silently omitted.
    report_lines.append("=" * 60)
    report_lines.append("SUPPLEMENTARY: default 0.5-threshold decision (see caveat above)")
    report_lines.append("=" * 60 + "\n")
    default_summaries = []
    for attr in PROTECTED_ATTRIBUTES:
        summary, by_group, di_ratio = audit_attribute(y_test, y_pred_default, y_proba, protected_test[attr], attr)
        default_summaries.append(summary)
        report_lines.append(f"--- {attr} (0.5 threshold) ---")
        report_lines.append(f"Demographic Parity Difference: {summary['demographic_parity_difference']}")
        report_lines.append(f"Equalized Odds Difference: {summary['equalized_odds_difference']}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    (REPORTS_DIR / "fairness_metrics_report.txt").write_text(report_text)
    pd.DataFrame(default_summaries).to_csv(REPORTS_DIR / "fairness_metrics_summary_0.5threshold.csv", index=False)

    print(report_text)
    return summary_df


if __name__ == "__main__":
    run_fairness_audit()
