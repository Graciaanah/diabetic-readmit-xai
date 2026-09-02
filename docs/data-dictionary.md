# Data Dictionary — Cleaned Cohort Output

Source: `data/processed/cleaned_cohort.csv`, produced by `src/preprocessing/dag.py`
from the raw Diabetes 130-US Hospitals dataset (Strack et al., 2014).

**Rows:** 17,983 (high-utilization diabetic cohort, post-cleaning)
**Grain:** one row per qualifying inpatient encounter

## Identifiers (dropped for privacy — see Anonymization Script)

| Column | Status |
|---|---|
| `encounter_id` | Dropped in `anonymize.py` |
| `patient_nbr` | Dropped in `anonymize.py` |
| `patient_split_key` | Added — one-way hash of `patient_nbr`, used only for patient-level train/test splitting in Module 4 |

## Demographics

| Column | Type | Description |
|---|---|---|
| `race` | string | Patient race; imputed to `"Unknown"` where missing (~2% of raw data) |
| `gender` | string | Patient gender |
| `age` | string | 10-year age band, e.g. `[70-80)` |

## Admission Context

| Column | Type | Description |
|---|---|---|
| `admission_type_id` | int | Coded admission type (see `data/raw/IDS_mapping.csv`) |
| `discharge_disposition_id` | int | Coded discharge disposition; expired/hospice codes excluded from this cohort |
| `admission_source_id` | int | Coded admission source |
| `time_in_hospital` | int | Length of stay in days (1–14) |

## Prior Utilization (cohort-defining fields)

| Column | Type | Description |
|---|---|---|
| `number_outpatient` | int | Outpatient visits, preceding year |
| `number_emergency` | int | Emergency visits, preceding year |
| `number_inpatient` | int | Inpatient visits, preceding year |

**Cohort filter:** `number_inpatient + number_emergency >= 2` (high-utilization definition from the Module 1 Vision Document)

## Clinical

| Column | Type | Description |
|---|---|---|
| `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` | int | Encounter-level counts |
| `medical_specialty` | string | Admitting physician specialty; imputed to `"Unknown"` where missing (~49% of raw data) |
| `diag_1`, `diag_2`, `diag_3` | string | Raw ICD-9 codes; imputed to `"Missing"` where absent |
| `diag_1_category`, `diag_2_category`, `diag_3_category` | string | Coarse clinical groupings (Diabetes, Circulatory, Respiratory, etc.) derived from the raw ICD-9 codes |
| `A1Cresult` | string | HbA1c test result category; imputed to `"Not Tested"` where missing (~83% of raw data) — kept as an explicit category rather than dropped, since Strack et al. (2014) found whether the test was administered at all to be independently predictive of readmission |
| `metformin` … `metformin-pioglitazone` (23 columns) | string | Per-medication dosage-change indicator (`No`/`Steady`/`Up`/`Down`) |
| `change`, `diabetesMed` | string | Medication change flag; diabetes medication flag |

## Target

| Column | Type | Description |
|---|---|---|
| `readmitted` | string | Original 3-class label (`<30`, `>30`, `NO`) |
| `readmitted_30d` | int | Binarized target used for modeling: 1 if `readmitted == "<30"`, else 0 |

## Dropped Fields (documented, not present in output)

| Column | Reason |
|---|---|
| `weight` | ~97% missing |
| `payer_code` | ~40% missing |
| `max_glu_serum` | ~95% missing |

Full missingness rationale: see `docs/vision-document.md`, Section 5.2.
