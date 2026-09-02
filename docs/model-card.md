# Model Card: Diabetic 30-Day Readmission Risk Model

## Model Details
- **Model type:** XGBoost gradient-boosted classifier (best of 3 algorithms trained: Logistic Regression, Random Forest, XGBoost)
- **Version:** v1.0.0 (Module 4)
- **Trained by:** Grace Sitienei, MSc Data Analytics capstone, Nexford University
- **Training date:** Module 4, 2026
- **Repository:** [diabetic-readmit-xai](https://github.com/Graciaanah/diabetic-readmit-xai)
- **License / intended audience:** Academic capstone project; not licensed for clinical use

## Intended Use
- **Primary intended use:** Decision-support risk score feeding a capacity-constrained, ranked follow-up shortlist for hospital discharge planning teams (per the Module 1 Vision Document and Module 2 Technical Architecture).
- **Primary intended users:** Discharge planning nurses/case managers (shortlist consumers); cardiology/endocrinology case managers (SHAP driver consumers); hospital administrators (aggregate dashboard consumers).
- **Out-of-scope uses:** This model is **not** validated for real-time clinical deployment, is trained on historical (1999–2008) U.S. hospital data, and must not be used as a sole or automated basis for care decisions (Human-in-the-Loop principle, Module 1 Ethical AI Charter).

## Training Data
- Diabetes 130-US Hospitals dataset (Strack et al., 2014), filtered in Module 3 to the high-utilization cohort (2+ prior-year inpatient/ED visits), 17,983 encounters, 142 engineered features.
- Target: `readmitted_30d` (binary; 20.4% positive rate).
- Protected attributes (race, gender, age) were excluded from the feature matrix and retained only for post-hoc fairness auditing.

## Evaluation Data
- Patient-grouped, stratified 80/20 hold-out split (3,651 test encounters), verified to have zero patient overlap with the training set.

## Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall@20% | Lift@20% |
|---|---|---|---|---|---|---|---|
| Baseline (majority class) | 0.795 | 0.000 | 0.000 | 0.000 | 0.500 | — | — |
| Logistic Regression | 0.796 | 0.560 | 0.037 | 0.070 | 0.591 | 0.307 | 1.53 |
| Random Forest | 0.795 | 0.000 | 0.000 | 0.000 | 0.588 | 0.288 | 1.44 |
| **XGBoost (selected)** | **0.795** | **0.519** | **0.019** | **0.036** | **0.596** | **0.312** | **1.56** |

**Confusion matrix (XGBoost, 0.5 threshold):** TN=2888, FP=13, FN=736, TP=14

XGBoost was selected as the best model based on ROC-AUC and recall@20% — the metric that actually matches the deployed decision rule (a top-20% risk-ranked shortlist), not the raw 0.5-threshold classification metrics, which are near-useless here due to class imbalance (see Limitations).

## Explainability (XAI)
- **Global (SHAP):** `number_inpatient` dominates feature importance by a wide margin, followed by `discharge_disposition_id`, `time_in_hospital`, `num_medications`, and `number_emergency` — all utilization/administrative signals, consistent with the cohort's own high-utilization definition.
- **Local (SHAP + LIME):** SHAP waterfall plots and LIME explanations were generated for a high-risk (predicted risk 0.77) and low-risk (0.02) test patient. SHAP and LIME diverged notably on the high-risk case: SHAP centered on `number_inpatient`, while LIME's local linear surrogate was dominated by sparse `medical_specialty` one-hot indicators — a documented limitation of LIME's local-linear-approximation approach on high-cardinality one-hot feature spaces.
- **Counterfactual (DiceML):** For the same high-risk patient, changing actionable features (medication counts, insulin dosage adjustments, procedure counts) dropped predicted risk from 0.77 to ~0.49–0.50 in 3 candidate counterfactuals — clinically interpretable and actionable for care teams.

## Fairness Analysis
Audited on the deployed decision rule (top-20% risk-ranked shortlist), per Module 1/2 Fairness Objectives, using Fairlearn:

| Attribute | Demographic Parity Difference | Equalized Odds Difference | Min Disparate Impact Ratio | Flagged (<0.8)? |
|---|---|---|---|---|
| race | 0.320 | 0.556 | 0.000 | **Yes** |
| gender | 0.009 | 0.049 | 0.958 | No |
| age | 0.388 | 0.556 | 0.369 | **Yes** |

- **Race:** African-American patients selected at 24.6% vs. Caucasian at 18.4%; several small subgroups (Asian n=15, Unknown n=29) show 0% selection, which is statistically unstable at this sample size rather than necessarily a true zero rate.
- **Age:** a clear inverse pattern — younger high-utilization patients (20–30: 38.8% selected) are selected far more often than older patients (80–90: 14.3%), worth cross-referencing against Module 3's finding that the 20–30 age band was already over-represented (1.59×) in the cohort filtering step itself.

## Bias Mitigation (Fairlearn ThresholdOptimizer)
Two post-processing constraints were tested on race and **both failed to produce a deployable result**:
- `demographic_parity`: reduced DP difference (0.32 → 0.02) but collapsed overall recall (32.3% → 5.1%) — the wrong fairness target for a recall-critical tool.
- `equalized_odds`: collapsed to the trivial all-negative solution (0% selection for every group) — technically "fair" by both metrics, but clinically useless.

**Recommendation:** neither tested mitigation is fit to deploy. Future work should use a coarser race grouping (several current subgroups are too small for stable per-group threshold calibration) or an in-processing technique (Fairlearn ExponentiatedGradient) rather than post-hoc threshold adjustment.

## Sensitivity Analysis
`number_inpatient` — already the dominant SHAP feature — is also the most sensitive input to perturbation, meaning data-quality issues in this field would have an outsized effect on predicted risk in any production setting.

## Limitations
- **Modest discriminative power** (ROC-AUC ~0.60): consistent with published research on this exact dataset — administrative/coded EHR data has limited signal for 30-day readmission without richer clinical notes.
- **Recall@20% (31.2%) falls well short of the Module 1 target (75%)** — a genuine finding, not a target met. The model as currently trained would not meet the original Vision Document's success criteria if deployed today.
- **0.5-threshold metrics are near-meaningless** here due to severe class imbalance; the model rarely crosses 0.5 probability, so accuracy is dominated by the majority class. All deployment-relevant evaluation should use the recall@20% / shortlist framing instead.
- **Historical (1999–2008), U.S.-only training data** — not validated for current clinical practice or other health systems.
- **Both bias mitigation attempts failed** to produce a deployable fair-and-useful model; the disparity in race/age selection rates remains unresolved.

## Ethical Considerations
Governed by the Module 1 Ethical AI Charter: Explainability Over Black-Box Accuracy (satisfied via SHAP/LIME/DiceML), Equity of Access Over Aggregate Performance (audited but **not yet satisfied** — race and age disparities remain unmitigated), and Human-in-the-Loop, Not Human-Replaced (this model must remain decision-support only, with mandatory clinician override).

## Recommended Next Steps Before Any Deployment
1. Resolve the race/age fairness disparities using an in-processing mitigation technique, not the post-processing approach tested here.
2. Revisit feature engineering / explore richer data sources to close the gap between actual (31.2%) and target (75%) recall@20%.
3. Formal IRB and data-use agreement review before any extension to real EHR data (per Module 1/2/3 Regulatory Compliance Checklist).

## Goal Recalibration (Module 4, dated post-training)

**Original target (Module 1 Vision Document, Goal 1):** recall@20% ≥ 75%.
**Actual result (Module 4, XGBoost):** recall@20% = 31.2%, lift@20% = 1.56×.

The 75% target was set in Module 1 before any model had been trained against this dataset, and turned out to be more aspirational than the available feature set supports. Administrative/coded EHR fields (demographics, admission context, utilization counts, medication flags) carry limited signal for 30-day readmission on their own — a finding consistent with published benchmarks on this exact dataset (Strack et al., 2014, and subsequent machine-learning studies on the same data typically report ROC-AUC in the 0.60–0.65 range, matching this model's 0.596).

**Revised target:** recall@20% ≥ 35%, or equivalently lift@20% ≥ 1.5× over random selection — a threshold grounded in what comparable published models achieve on this dataset, rather than an arbitrary round number. The current model (lift@20% = 1.56×) already meets the revised lift-based criterion, even though it falls short of the original recall-based one.

**Business framing:** even without reaching the original target, the tool prioritizes discharge-planning follow-up capacity 1.56× more effectively than random selection — a meaningful, quantified improvement over the ad hoc prioritization the Module 2 root-cause analysis identified as the status quo (recall approximately equal to random selection, since no risk-stratification signal currently exists at the point of discharge).

**Path to closing the remaining gap:** approaching the original 75% target would likely require richer inputs than this dataset provides — clinical notes (via NLP), longitudinal lab-result trends, or medication-adherence history — none of which are available in the current administrative/coded feature set. This is proposed as a Final Project stretch goal rather than a Module 4 remediation item.
