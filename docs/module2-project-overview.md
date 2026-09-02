**Business Analytics Project Overview Report**

*Diabetic Readmission Risk Stratification: From Vision to Delivery Plan*

Grace Sitienei

MSc Data Analytics — Nexford University

Module 2: Business Analytics Project Overview Report

1\. Define the Business Problem

This project builds directly on the Module 1 Vision Document, carrying
forward the same vertical, problem, and dataset. The core issue is that
hospital discharge planning teams have no data-driven way to identify
which diabetic patients, among those with a high prior-year utilization
burden (two or more inpatient or emergency-department visits), are most
likely to be readmitted within 30 days — and therefore cannot direct
their limited weekly follow-up capacity toward the patients who need it
most.

Applying the problem-identification framework: (define) the problem is
scoped exactly as narrowed in Module 1 — a capacity-constrained,
discharge-planning-owned risk-prioritization gap, not a general “reduce
readmissions” goal; (gather data) the Diabetes 130-US Hospitals dataset
(101,766 encounters, 130 U.S. hospitals, 1999–2008; Strack et al., 2014)
supplies the demographic, admission, utilization, and clinical fields
needed; (analyze) Section 2 traces the problem to its structural cause;
(root cause) the absence of a scoped, owned decision-support tool;
(objectives) the three SMART goals defined in Module 1 — recall@20% ≥
75%, a capacity-sized shortlist with measurable lift over random
selection, and a top-5 explainable driver set.

This alignment is intentional: every objective in this planning document
maps to a goal already justified in Module 1, so nothing here re-argues
the business case — it only makes that case executable by a technical
team.

Scope is deliberately bounded to match Module 1: this is a
retrospective, dataset-driven build using historical (1999–2008) data,
not a live clinical integration. Expected deliverables from this
planning phase are an initialized GitHub repository matching the Module
1 directory structure, a defined sprint board, a technical architecture
diagram, a Technical RAID Log, and the privacy, fairness, and compliance
requirements formalized in Section 6 — each of which a technical team
could pick up tomorrow without needing to re-derive the vision or the
ethical constraints from scratch.

2\. Root Cause Analysis

A 5 Whys analysis was used to trace the readmission problem to its
structural root cause (Figure 1).

<img src="media/fd0d72c5db64bca293bafc2f37d865bdcd78c47c.png"
style="width:4.16667in;height:3.42708in" />

*Figure 1. 5 Whys analysis tracing the readmission problem to its root
cause.*

The chain shows that patients are readmitted because follow-up is not
matched to risk (Why 1–2), because no consistent risk signal exists at
discharge (Why 3), because the EHR data needed to build one has never
been operationalized for this purpose (Why 4). The root cause is the
absence of a scoped, owned, capacity-aware problem definition — exactly
the gap Module 1 closed and this document now makes buildable.

This root cause links directly to the project's KPIs: recall@20%
(Goal 1) directly replaces the ad hoc judgment identified at Why 2 with
a consistent signal; lift over random selection (Goal 2) addresses the
capacity-matching gap at Why 3; and the explainability output (Goal 3)
makes that signal actionable rather than a black-box number, addressing
Why 4.

3\. Business Analytics Project Overview

Objectives, scope, and deliverables carry forward unchanged from Module
1: build a risk-stratification tool for discharge planning teams that
scores 30-day readmission risk for the high-utilization diabetic cohort
and converts that score into a capacity-constrained, ranked follow-up
shortlist. Required data types are demographic (race, gender, age band),
admission context (type, source, length of stay), prior utilization
(inpatient/ED/outpatient visit counts), clinical (labs, HbA1c,
diagnoses), and medication fields, all drawn from the Diabetes 130-US
Hospitals dataset (Strack et al., 2014).

Two analytical methods are integrated: a predictive model producing the
30-day readmission risk score, and a prescriptive layer that ranks and
sizes the follow-up shortlist to real weekly staffing capacity. Figure 2
formalizes the technical architecture first proposed as a stack table in
Module 1, now shown as an end-to-end system diagram.

<img src="media/3447de37e99405cb8b2fe25f7251490f63e3088c.png"
style="width:4.16667in;height:4.07292in" />

*Figure 2. Technical architecture: data source through preprocessing,
predictive modeling, parallel explainability/fairness checks,
prescriptive prioritization, and the discharge-planning dashboard.*

Each component in Figure 2 maps to a specific tool named in the Module 1
Technical Stack (pandas for preprocessing, scikit-learn/XGBoost for
modeling, SHAP for explainability, Fairlearn for fairness auditing,
Streamlit for the dashboard) and to a folder in the Module 1 repository
structure (src/preprocessing/, src/modeling/, src/fairness/,
dashboard/).

4\. Agile Methodology

The project is delivered as four sprints, each corresponding to a course
module, allowing iterative feedback (instructor and, conceptually,
discharge-planning stakeholders) between stages rather than a single
end-to-end build. This mirrors agile's core value of adapting the plan
as each sprint surfaces new information — for example, if Sprint 1
data-quality findings change the viable cohort size, Sprint 2's modeling
scope adjusts accordingly rather than proceeding on stale assumptions.

This iterative structure also protects value delivery under uncertainty:
because Sprint 2's fairness audit could surface a subgroup disparity
that requires revisiting feature choices, the plan treats that as an
expected checkpoint rather than a failure state, consistent with the
Human-in-the-Loop and Equity of Access principles from the Module 1
Ethical AI Charter. Each sprint closes with a defined output reviewed
before the next begins, so scope changes are absorbed one sprint at a
time instead of compounding silently across the whole build.

| **Sprint** | **Module**    | **Focus**            | **Key tickets**                                                               |
|------------|---------------|----------------------|-------------------------------------------------------------------------------|
| Sprint 1   | Module 3      | Data pipeline        | Ingest data; drop high-missingness fields; cohort filter; patient-level split |
| Sprint 2   | Module 4      | Model, XAI, fairness | Train model; tune recall@20%; SHAP layer; Fairlearn audit; shortlist logic    |
| Sprint 3   | Module 5      | Dashboard            | Build dashboard; driver display; fairness report view; stakeholder deck       |
| Sprint 4   | Final Project | Integration          | End-to-end test; deploy; finalize documentation                               |

A GitHub Project Board (Kanban) was created with Backlog, To Do, In
Progress, In Review, and Done columns, with tickets from the table above
entered as issues and labeled by sprint, directly referencing the Module
1 repository structure (each ticket links to the repo folder it produces
work in).

5\. Implementation and Communication Plan

Stakeholder roles carry forward from Module 1: discharge planning nurses
and case managers consume the ranked shortlist and retain override
authority; cardiology/endocrinology case managers consume SHAP driver
explanations to shape follow-up content; hospital administrators consume
the aggregate dashboard for HRRP penalty tracking; the Clinical & Data
Ethics Review Panel reviews fairness results before each sprint's model
is considered final. Communication flows sprint-to-sprint: each sprint's
output (data quality report, model + fairness audit, dashboard) is
reviewed before the next sprint begins, preventing downstream work from
building on an unvalidated upstream stage.

Monitoring KPIs are the same three SMART metrics from Module 1 —
recall@20% ≥ 75%, shortlist lift over random selection, and a documented
top-5 driver set — tracked at the end of Sprint 2 and re-verified before
Sprint 3 dashboard integration. If a KPI is not met at the Sprint 2
checkpoint, the plan calls for revisiting feature engineering or cohort
scope before proceeding to Sprint 3, rather than shipping a dashboard
built on an unvalidated model.

Technical RAID Log (Summary)

A full Technical RAID Log was developed as a companion spreadsheet,
expanding the Module 1 Ethical Risk Register into Risks, Assumptions,
Issues, and Dependencies tied to specific sprints. Key entries are
summarized below.

| **Type**   | **Item**                                                   | **Sprint**                 | **Mitigation**                                       |
|------------|------------------------------------------------------------|----------------------------|------------------------------------------------------|
| Risk       | Algorithmic bias via demographic proxies                   | Sprint 2                   | Fairlearn subgroup audit; Ethics Panel review        |
| Risk       | Data leakage from repeat patient encounters                | Sprint 1                   | Patient-level train/test split                       |
| Assumption | ~15% weekly follow-up capacity is realistic                | Sprint 2                   | Revisit shortlist size after lift results            |
| Issue      | NXU vs. APA citation format ambiguity                      | Module 1 (carried forward) | Confirm with instructor before submission            |
| Dependency | Dashboard build depends on finalized SHAP/Fairlearn output | Sprint 2 → 3               | Freeze model version (SemVer tag) before integration |

6\. Privacy, Fairness & Compliance Requirements

This section operationalizes the Ethical AI Charter and Fairness
Objectives from Module 1, converting stated principles into specific
mechanisms a technical team can implement.

Data Privacy Plan

Patient and encounter identifiers are dropped before modeling; any
subgroup breakdown in the fairness audit or dashboard suppresses or
aggregates cells with fewer than five records (k-anonymity, k=5); no
merging with external datasets is permitted at any stage; raw data
remains isolated from processed, de-identified data used downstream; and
all data is retained only for the course engagement, with a defined
deletion point at close-out, per the Module 1 Data Governance
Principles.

Fairness Metrics

Module 1's qualitative Fairness Objectives are operationalized here as
two specific, Fairlearn-computed metrics: Demographic Parity
(selection-rate difference into the high-risk shortlist across race, age
band, and gender) and Equalized Odds (recall and false-positive rate
parity across the same subgroups), both reported at every model
iteration in Sprint 2, not only at final delivery.

Explainability Framework

SHAP is applied at two levels: globally, to produce the top-5
cohort-wide risk drivers required by Goal 3; and locally, to produce a
per-patient explanation a discharge planner can act on. Output is
rendered as plain-language driver labels rather than raw feature names
or SHAP plots, consistent with the Explainability Over Black-Box
Accuracy principle.

Regulatory Compliance Checklist

| **Regulation**              | **Applicability**                                                          | **Status**                  |
|-----------------------------|----------------------------------------------------------------------------|-----------------------------|
| HIPAA (current)             | Source data de-identified; HIPAA-equivalent discipline applied voluntarily | Applied                     |
| HIPAA (future real-EHR use) | Would require data-use agreement and IRB review                            | Flagged, not yet applicable |
| GDPR                        | Not applicable — U.S.-sourced data, no EU subjects                         | N/A                         |
| CCPA                        | Not applicable — outside CCPA's scope                                      | N/A                         |

References

Centers for Medicare & Medicaid Services. (2023). Hospital Readmissions
Reduction Program (HRRP). U.S. Department of Health and Human Services.
https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S.,
Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on
hospital readmission rates: Analysis of 70,000 clinical database patient
records. BioMed Research International, 2014, Article 781670.
https://doi.org/10.1155/2014/781670
