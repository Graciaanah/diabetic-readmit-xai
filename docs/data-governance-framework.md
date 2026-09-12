# Data Governance Framework

This framework extends the Data Governance Principles from the Module 1 Vision
Document and the Data Privacy Plan from the Module 2 Project Overview Report,
operationalized here as concrete access, usage, and retention rules for the
data pipeline built in Module 3.

## Access Control

| Data asset | Who can access | Mechanism |
|---|---|---|
| `data/raw/` (raw CSVs) | Analyst only, local environment | Never committed to GitHub (`.gitignore`); excluded from the Docker image |
| `data/processed/` (cleaned cohort) | Analyst only, local environment | De-identified before write (see `anonymize.py`); also gitignored |
| `reports/` (validation, bias, audit logs) | Analyst; Ethics Review Panel on request | Committed selectively — aggregate reports only, never row-level data |
| Source code (`src/`) | Public (repository is portfolio-facing) | No credentials, connection strings, or data samples are ever hardcoded |

## Usage Policy

- Data is used strictly for this course's academic capstone purpose, per the
  Module 1 Data Governance Principles. Any repurposing (e.g., toward a real
  hospital deployment) requires renewed review by the Clinical & Data Ethics
  Review Panel defined in the Module 2 report.
- No merging with any external dataset at any pipeline stage — enforced
  structurally (the pipeline has no external join points) and confirmed by
  the Great Expectations schema checks (`validate.py`).
- Direct identifiers (`encounter_id`, `patient_nbr`) are dropped in the
  `anonymize` stage before the data is written to `data/processed/`, so no
  downstream stage (modeling, dashboard) ever has access to them.

## Retention

- Raw and processed data are retained only for the duration of course
  engagement. Both directories are excluded from version control, so the
  retention boundary is the analyst's local machine, not GitHub.
- `reports/privacy_audit.log` and `reports/bias_detection_report.txt` are
  retained as aggregate evidence of governance compliance; they contain no
  row-level data and may be kept indefinitely as documentation.

## Privacy Audit Logging

Every pipeline stage (`ingest`, `clean`, `anonymize`, `transform`) calls
`log_access()` (see `src/preprocessing/audit_log.py`), producing a
timestamped record of what operation ran, on which dataset, and with what
row/column-level metadata — never raw field values. This creates a durable,
reviewable trail of every transformation the data underwent, satisfying the
Module 3 Privacy Audit Logging requirement and giving the Ethics Review
Panel a concrete artifact to review rather than a narrative description.

## k-Anonymity Enforcement

Any reporting table that breaks data down by demographic subgroup (bias
detection report, future Module 4 fairness audits) applies k=5 suppression
via `suppress_small_cells()`: any subgroup with fewer than 5 records has its
values replaced with a suppression marker before the table is written or
displayed, consistent with the Privacy by Design principle in the Module 1
Ethical AI Charter.

## Known Limitations

This governance framework is scoped to an academic, retrospective,
historical dataset. It is explicitly **not** sufficient for real EHR data —
that would require a signed institutional data-use agreement, IRB or ethics
board review, and a formal HIPAA compliance assessment, all flagged as
preconditions in the Module 2 Regulatory Compliance Checklist and not yet
in place.
