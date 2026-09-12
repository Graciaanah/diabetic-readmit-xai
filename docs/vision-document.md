**Business Analytics Project Vision Document**

*Reducing 30-Day Readmissions for High-Utilization Diabetic Patients
Through Discharge Planning Risk Stratification*

Grace Sitienei \| Nexford University \| MSc Data Analytics \| Module 1
Capstone Deliverable

1\. Executive Summary

Diabetic patients with a history of frequent hospital utilization face
disproportionately high 30-day readmission rates, driving both clinical
harm and financial penalty exposure under the CMS Hospital Readmissions
Reduction Program (Centers for Medicare & Medicaid Services, 2023). This
project proposes a risk-stratification tool for hospital discharge
planning teams that identifies diabetic patients with two or more
prior-year inpatient/ED visits at elevated readmission risk, and
prioritizes them for follow-up within the team's existing weekly
capacity.

The initiative integrates two analytics types: a predictive component
(30-day readmission risk score) and a prescriptive component (a
capacity-constrained, ranked follow-up shortlist). The analysis uses the
Diabetes 130-US Hospitals dataset — 101,766 inpatient diabetic
encounters from 130 U.S. hospitals, 1999–2008 (Strack et al., 2014).
This document establishes the vision, ethics, stakeholders, technical
direction, and governance that will guide the project through Modules
2–5 and the Final Project.

2\. Business Case & Problem Definition

Readmissions are a defined financial liability: hospitals with excess
readmissions face direct Medicare reimbursement penalties (Centers for
Medicare & Medicaid Services, 2023), and inconsistent glycemic
management independently drives inpatient cost and morbidity (Strack et
al., 2014). A generic framing — “reduce readmissions for diabetic
patients” — fails to guide a build, since nearly every dataset record
already qualifies as diabetic. This project narrows the problem to:

*Build a readmission risk-stratification tool for hospital discharge
planning teams, to prioritize post-discharge follow-up for diabetic
patients with high prior-utilization burden (2+ inpatient/ED visits in
the preceding year), within a fixed weekly outreach capacity.*

This names a specific sub-population, owner (discharge planning, not
administration or an external payer), and constraint (weekly capacity)
that forces a prioritized ranking rather than a blanket flag. The
dataset clears the 5,000-record minimum by a wide margin (101,766
encounters, 50+ features; Strack et al., 2014). Scope boundary: this is
a retrospective, academic analysis of historical (1999–2008) data, not a
live clinical deployment.

Why This Definition Matters: The Drift Risk

Without an explicit, written-down problem definition, this specific
project has a predictable failure path. The assignment brief warns that
analytics initiatives typically do not fail because a model was
technically wrong, but because no one recorded, before any code was
written, what problem was being solved, for whom, and at what ethical
cost — so the project quietly drifts until it is solving the wrong
problem. For this project, that drift would most likely take two
concrete forms: (1) the cohort silently expanding back from the defined
high-utilization sub-population to “all diabetic patients,” which is not
a real segment but the entire dataset, producing a model with no
meaningful target population; and (2) the deliverable drifting from a
capacity-constrained, ranked shortlist tied to a named owner (the
discharge planning team) toward an unranked, undifferentiated risk flag
with no defined action or user attached to it. Fixing the sub-cohort,
the owner, and the capacity constraint in Section 2 above is what
forecloses both drift paths before Module 2 begins.

3\. Vision, Goals & Ethical Foundations

Vision Statement

*“To equip discharge planning teams with a trusted, data-driven
early-warning system that ensures no high-risk diabetic patient leaves
the hospital without a clear, prioritized path to follow-up care —
reducing preventable readmissions within existing staffing capacity.”*

Strategic Goals (SMART)

- Goal 1 — Risk Identification Accuracy: identify ≥75% of actual 30-day
  readmissions within the top 20% of risk-ranked patients (recall@20%)
  on a held-out test set. Users: discharge nurses/case managers, who
  gain a defensible worklist over ad hoc prioritization.

- Goal 2 — Capacity-Constrained Prioritization: produce a shortlist
  sized to weekly capacity (e.g., top 15% of discharges) with measurable
  lift over random selection. Users: team leads, who gain a tool matched
  to real staffing limits.

- Goal 3 — Explainable Risk Drivers: deliver a top-5 feature-importance
  analysis in clinician-actionable terms. Users: case managers and
  administrators, who gain drivers that inform both individual care and
  quality-improvement work.

Ethical AI Charter

The tool must earn clinical staff trust and must not reproduce or
amplify structural healthcare inequities. Fairness, transparency, and
accountability are design requirements, not a post-hoc check. The
following high-level principles govern this project:

- Explainability Over Black-Box Accuracy — interpretable,
  clinician-facing drivers take priority over marginal accuracy gains.

- Equity of Access Over Aggregate Performance — performance is evaluated
  per subgroup (race, age, gender); strong average performance does not
  excuse subgroup underperformance.

- Human-in-the-Loop, Not Human-Replaced — output is decision support
  only; staff retain full override authority.

- Privacy by Design — patient_nbr and encounter_id are dropped before
  modeling; any reported subgroup breakdown (e.g., in the fairness
  audit) is suppressed or aggregated whenever a cell contains fewer than
  five records, a k-anonymity threshold of k=5, so no combination of
  demographic and clinical fields can be narrowed down to an
  identifiable individual; no linkage to external datasets is permitted
  at any stage.

Fairness Objectives

- Recall and false-negative rate must not vary materially across race,
  age band, or gender.

- Subgroups with disproportionate false-negative rates must be
  documented and reviewed before the model is finalized.

- Proxy features (e.g., payer, admission source) are reviewed for
  indirect encoding of socioeconomic bias.

4\. Stakeholder Analysis & User Personas

- Discharge Planning Nurses/Case Managers (primary users) — need an
  actionable shortlist, not a raw score, under time pressure.

- Cardiology/Endocrinology Case Managers (secondary users) — use
  risk-driver explanations to shape follow-up content (e.g., medication
  reconciliation, HbA1c retesting).

- Hospital Administrators/QI Leads (tertiary users) — consume aggregate
  dashboards tracking readmission trends and HRRP penalty exposure.

- Patients/Caregivers (non-user stakeholders) — most directly affected
  by model accuracy and fairness; interested in continuity of care.

- Ethics Oversight: a Clinical & Data Ethics Review Panel (clinical
  advisory reviewer, data governance/fairness reviewer, and model owner)
  reviews the model against the Fairness Objectives before each
  milestone.

5\. Proposed Analytics Solution & Data Strategy

The predictive component scores 30-day readmission risk using clinical,
demographic, and prior-utilization features; the prescriptive component
converts that score into a capacity-sized, ranked follow-up shortlist,
accompanied by an explainability layer and a per-subgroup fairness
report at every iteration.

Dataset: 130 U.S. hospitals, 1999–2008, 101,766 encounters, 50+ features
covering demographics, admission context, utilization history, labs, and
medications (Strack et al., 2014). Key data-quality decisions: weight
(≈97% missing), payer_code (≈40%), and max_glu_serum (≈95%) will be
dropped rather than imputed; race and diag_1–diag_3 (≈2% missing) will
be imputed; splitting will be patient-level (not encounter-level) to
prevent leakage from repeat patients; high-cardinality ICD-9 diagnosis
codes will be grouped into clinical categories.

Proposed Technical Stack

| **Project Phase**    | **Tool / Platform**                               |
|----------------------|---------------------------------------------------|
| Data pipeline        | Python (pandas); local → AWS S3 for Final Project |
| Modeling             | Python (scikit-learn, XGBoost); Jupyter           |
| Explainability       | SHAP                                              |
| Fairness analysis    | Fairlearn                                         |
| Dashboard (Module 5) | Streamlit or Power BI                             |
| Deployment (Final)   | Streamlit Community Cloud                         |
| Version control      | Git / GitHub                                      |

Proposed Repository Structure

capstone-diabetic-readmission/

├─ data/{raw,processed}/

├─ notebooks/

├─ src/{preprocessing,modeling,fairness}/

├─ reports/

├─ dashboard/

├─ docs/vision-document.md

├─ requirements.txt

└─ README.md

6\. Requirements, Risks, and Constraints

- Functional: cohort filtering; risk scoring; capacity-constrained
  shortlist; per-prediction explainability; subgroup fairness reporting;
  human override.

- Non-functional: outputs interpretable by non-technical staff;
  reproducible given the same data/code; small-cell suppression on
  demographic breakdowns; all preprocessing decisions documented.

- Risks/constraints: severe missingness and high-cardinality codes
  (technical); staff distrust of a “black box” (business, mitigated by
  explainability + override); HIPAA-equivalent handling discipline
  applied despite de-identified data, with IRB/data-use agreement
  required before any real-EHR extension (regulatory); retrospective,
  open-source-only scope (time/budget).

Ethical Risk Register

| **Risk**                            | **Likelihood**           | **Impact** | **Mitigation**                                                                                           |
|-------------------------------------|--------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Bias via race/age/payer proxies     | Medium                   | High       | Subgroup fairness audit; Ethics Panel review                                                             |
| Re-identification risk              | Low                      | High       | No external merging; k-anonymity suppression (k=5) on subgroup reports; identifiers dropped pre-modeling |
| Automation bias / over-reliance     | Medium                   | Medium     | Human-in-the-loop; override with no burden                                                               |
| Regulatory non-compliance if scaled | Low now / High if scaled | High       | IRB & data-use agreement required pre-deployment                                                         |

7\. Quality, Documentation, and Governance Standards

All preprocessing and scoping decisions (cohort filter, dropped/imputed
fields, diagnosis grouping) are logged for reproducibility. Deliverables
follow Nexford University formatting standards with APA citations
throughout.

Versioning Strategy

- Code: Git Flow branching model, merged to main at each module
  milestone.

- Data and models: versioned with DVC (Data Version Control), so any
  result traces back to the exact dataset and pipeline version that
  produced it.

- Releases: Semantic Versioning (v0.1.0 at Module 1, through v1.0.0 at
  Final Project deployment).

Data Governance Principles

- Access: least-privilege access; no re-identification attempts or
  external data merging.

- Usage: data used strictly for this course's stated academic purpose;
  any repurposing requires renewed Ethics Panel review.

- Retention: data retained only for course duration, with a defined
  deletion point at project close-out.

- Known limitations: historical, U.S.-only data validated
  retrospectively only; results demonstrate methodology, not current
  clinical guidance; any live extension requires revalidation and formal
  regulatory review.

References

Centers for Medicare & Medicaid Services. (2023). Hospital Readmissions
Reduction Program (HRRP). U.S. Department of Health and Human Services.
https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S.,
Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on
hospital readmission rates: Analysis of 70,000 clinical database patient
records. BioMed Research International, 2014, Article 781670.
https://doi.org/10.1155/2014/781670
