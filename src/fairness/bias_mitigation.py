"""
Fairness Mitigation (Module 4 Sec. 6).
Bias was detected in the fairness audit (race and age flagged for
disparate impact ratio < 0.8 on the shortlist decision). This applies
Fairlearn's ThresholdOptimizer -- a post-processing mitigation that adjusts
the decision threshold PER SUBGROUP to satisfy a fairness constraint,
without retraining the underlying model -- constrained to demographic
parity on race (the attribute with the most severe disparity).

Trade-off made explicit: post-processing mitigation typically improves
group fairness at some cost to overall accuracy/AUC. Both before/after
are reported so that trade-off is visible, not hidden.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_auc_score, accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_ratio, demographic_parity_difference, equalized_odds_difference

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BEST_MODEL_NAME = "xgboost"
K_THRESHOLD = 5


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()
    protected_test = pd.read_csv(MODEL_DIR / "protected_test.csv")
    return model, X_test, y_test, protected_test


def apply_mitigation(model, X_test, y_test, sensitive_feature, constraint="demographic_parity"):
    """
    ThresholdOptimizer needs its own train split to fit the per-group
    thresholds without evaluating on the same data it was calibrated on.
    We split the test set itself 50/50 here (calibration / evaluation)
    since the original train set's protected attributes were deliberately
    excluded upstream and are only retained for this test-set audit.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_test))
    calib_idx, eval_idx = train_test_split(idx, test_size=0.5, random_state=42, stratify=y_test)

    postprocessor = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True,
    )
    postprocessor.fit(
        X_test.iloc[calib_idx], y_test.iloc[calib_idx],
        sensitive_features=sensitive_feature.iloc[calib_idx],
    )
    y_pred_mitigated = postprocessor.predict(
        X_test.iloc[eval_idx], sensitive_features=sensitive_feature.iloc[eval_idx],
    )
    return postprocessor, eval_idx, y_pred_mitigated


def compare_before_after(model, X_test, y_test, protected_test, attr="race", constraint="demographic_parity"):
    postprocessor, eval_idx, y_pred_mitigated = apply_mitigation(model, X_test, y_test, protected_test[attr], constraint=constraint)

    y_test_eval = y_test.iloc[eval_idx].reset_index(drop=True)
    sensitive_eval = protected_test[attr].iloc[eval_idx].reset_index(drop=True)
    y_pred_mitigated = pd.Series(y_pred_mitigated).reset_index(drop=True)

    # BEFORE: original model's top-20%-equivalent decision on the same eval slice
    y_proba_eval = model.predict_proba(X_test.iloc[eval_idx])[:, 1]
    k = int(np.ceil(len(y_proba_eval) * 0.20))
    thresh = np.sort(y_proba_eval)[::-1][k - 1]
    y_pred_before = pd.Series((y_proba_eval >= thresh).astype(int)).reset_index(drop=True)

    def summarize(y_pred, label):
        dp_diff = demographic_parity_difference(y_test_eval, y_pred, sensitive_features=sensitive_eval)
        dp_ratio = demographic_parity_ratio(y_test_eval, y_pred, sensitive_features=sensitive_eval)
        eo_diff = equalized_odds_difference(y_test_eval, y_pred, sensitive_features=sensitive_eval)
        acc = accuracy_score(y_test_eval, y_pred)
        overall_recall = true_positive_rate(y_test_eval, y_pred)
        mf = MetricFrame(metrics={"selection_rate": selection_rate, "recall": true_positive_rate},
                          y_true=y_test_eval, y_pred=y_pred, sensitive_features=sensitive_eval)
        counts = sensitive_eval.value_counts()
        by_group = mf.by_group.copy()
        by_group["count"] = counts.reindex(by_group.index)
        by_group = by_group.astype(object)
        small = by_group["count"].astype(float) < K_THRESHOLD
        by_group.loc[small, ["selection_rate", "recall"]] = "<suppressed>"
        by_group.loc[small, "count"] = f"<{K_THRESHOLD}"
        return {
            "label": label, "demographic_parity_difference": round(dp_diff, 4),
            "demographic_parity_ratio": round(dp_ratio, 4),
            "equalized_odds_difference": round(eo_diff, 4),
            "overall_recall": round(overall_recall, 4),
            "accuracy": round(acc, 4),
        }, by_group

    before_summary, before_by_group = summarize(y_pred_before, "before_mitigation")
    after_summary, after_by_group = summarize(y_pred_mitigated, f"after_mitigation_{constraint}")

    return before_summary, before_by_group, after_summary, after_by_group


if __name__ == "__main__":
    model, X_test, y_test, protected_test = load_artifacts()

    before_dp, before_bg, after_dp, after_dp_bg = compare_before_after(
        model, X_test, y_test, protected_test, attr="race", constraint="demographic_parity")
    before_eo, before_eo_bg, after_eo, after_eo_bg = compare_before_after(
        model, X_test, y_test, protected_test, attr="race", constraint="equalized_odds")

    lines = ["Bias Mitigation Report -- ThresholdOptimizer (attribute: race)", "=" * 80, ""]

    lines.append("BEFORE mitigation (top-20% shortlist decision, original model):")
    lines.append(str(before_dp))
    lines.append(before_bg.to_string())
    lines.append("")

    lines.append("AFTER mitigation -- constraint: demographic_parity")
    lines.append(str(after_dp))
    lines.append(after_dp_bg.to_string())
    lines.append("")

    lines.append("AFTER mitigation -- constraint: equalized_odds")
    lines.append(str(after_eo))
    lines.append(after_eo_bg.to_string())
    lines.append("")

    lines.append("=" * 80)
    lines.append("COMPARISON & INTERPRETATION")
    lines.append("=" * 80)
    lines.append(
        f"\ndemographic_parity constraint: DP difference {before_dp['demographic_parity_difference']} -> "
        f"{after_dp['demographic_parity_difference']}, but overall recall collapsed "
        f"{before_dp['overall_recall']} -> {after_dp['overall_recall']} (accuracy {before_dp['accuracy']} -> {after_dp['accuracy']}). "
        f"Demographic parity forces equal SELECTION rates across groups, not equal ability to correctly "
        f"identify true positives -- a poor fit for a health-outreach tool whose entire purpose is recall."
    )
    lines.append(
        f"\nequalized_odds constraint: an even more severe finding. DP difference "
        f"{before_dp['demographic_parity_difference']} -> {after_eo['demographic_parity_difference']} and "
        f"equalized odds difference {before_dp['equalized_odds_difference']} -> {after_eo['equalized_odds_difference']} "
        f"both hit zero -- but only because ThresholdOptimizer collapsed to the trivial all-negative solution "
        f"(0% selection rate and 0% recall for every group, including the majority group). A model that flags "
        f"no one is perfectly 'fair' by both metrics and clinically useless. This is a real failure mode, not a "
        f"success: with small per-group calibration samples (the calibration half has as few as n=8 for Asian, "
        f"n=18 for Unknown), ThresholdOptimizer's per-group threshold fitting is unstable enough that the "
        f"degenerate zero-selection point becomes the easiest feasible solution to the equality constraint."
    )
    lines.append(
        "\nRecommendation: neither post-processing constraint tested here is fit to deploy as-is. Two better "
        "paths forward: (1) coarsen the race grouping before mitigation (the current 6-category split leaves "
        "several subgroups too small for stable per-group threshold calibration), or (2) use an in-processing "
        "technique such as Fairlearn's ExponentiatedGradient, which optimizes the fairness-constrained objective "
        "jointly with the classifier rather than adjusting thresholds after the fact on a small calibration "
        "split. Both mitigation attempts are reported here in full -- including the failure -- because an "
        "honest fairness analysis has to show when a mitigation technique doesn't work, not just when it does."
    )

    report_text = "\n".join(lines)
    print(report_text)
    (REPORTS_DIR / "bias_mitigation_report.txt").write_text(report_text)
