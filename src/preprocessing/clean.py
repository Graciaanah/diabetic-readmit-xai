"""
Stage 2-3: Cleaning & Transformation.

Implements the data-quality decisions documented in the Module 1 Vision
Document (Sec. 5.2) and Module 2 Project Overview Report:
  - Drop weight, payer_code, max_glu_serum (>90% missing each)
  - Recode '?' placeholders to true NaN
  - Impute race and diag_1-3 (low missingness) rather than drop
  - Apply the high-utilization cohort filter (2+ inpatient/ED visits, prior yr)
  - Group high-cardinality ICD-9 diagnosis codes into clinical categories
  - Binarize the readmitted target to the 30-day framing
"""
import pandas as pd
import numpy as np

from .audit_log import log_access

DROP_FIELDS = ["weight", "payer_code", "max_glu_serum"]
IMPUTE_FIELDS = ["race", "diag_1", "diag_2", "diag_3", "medical_specialty", "A1Cresult"]


def recode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """The source dataset encodes missing values as the literal string '?'."""
    before = (df == "?").sum().sum()
    df = df.replace("?", np.nan)
    log_access("clean", "diabetic_data", step="recode_missing", cells_recoded=int(before))
    return df


def drop_high_missingness_fields(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DROP_FIELDS if c in df.columns]
    df = df.drop(columns=present)
    log_access("clean", "diabetic_data", step="drop_fields", dropped=present)
    return df


def impute_low_missingness_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in IMPUTE_FIELDS:
        if col not in df.columns:
            continue
        if col == "race":
            df[col] = df[col].fillna("Unknown")
        elif col == "medical_specialty":
            df[col] = df[col].fillna("Unknown")
        elif col == "A1Cresult":
            # A missing A1Cresult means the test was not performed, which is
            # itself clinically informative (Strack et al., 2014, found test
            # administration predictive of readmission) -- so it is encoded
            # as its own explicit category, not silently dropped.
            df[col] = df[col].fillna("Not Tested")
        else:  # diag_1-3: ICD-9 codes, impute with an explicit "Missing" bucket
            df[col] = df[col].fillna("Missing")
    log_access("clean", "diabetic_data", step="impute", fields=IMPUTE_FIELDS)
    return df


def drop_deceased_hospice(df: pd.DataFrame, id_mappings: dict) -> pd.DataFrame:
    """
    A patient who died or entered hospice cannot be meaningfully 'readmitted'.
    discharge_disposition_id values for expired/hospice are excluded per the
    Module 1 Vision Document's cohort-definition note.
    """
    disposition_map = id_mappings.get("discharge_disposition_id", {})
    exclude_labels = {"Expired", "Hospice / home", "Hospice / medical facility",
                       "Expired at home. Medicaid only, hospice.",
                       "Expired in a medical facility. Medicaid only, hospice.",
                       "Expired, place unknown. Medicaid only, hospice."}
    exclude_ids = {k for k, v in disposition_map.items()
                   if any(lbl.lower() in v.lower() for lbl in ["expired", "hospice"])}
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(exclude_ids)]
    log_access("clean", "diabetic_data", step="drop_deceased_hospice",
               excluded_ids=sorted(exclude_ids), rows_removed=before - len(df))
    return df


def apply_cohort_filter(df: pd.DataFrame) -> pd.DataFrame:
    """High-utilization cohort: 2+ inpatient OR emergency visits in the prior year."""
    before = len(df)
    mask = (df["number_inpatient"] + df["number_emergency"]) >= 2
    df = df[mask].copy()
    log_access("clean", "diabetic_data", step="cohort_filter",
               rows_before=before, rows_after=len(df))
    return df


# Coarse ICD-9 grouping, per Module 1 Sec. 5.2 ("group into clinical categories")
def _icd9_category(code) -> str:
    if code in ("Missing", None) or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    if code.startswith("V") or code.startswith("E"):
        return "External/Supplemental"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 800 <= val <= 999:
        return "Injury/Poisoning"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def group_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_category"] = df[col].apply(_icd9_category)
    log_access("clean", "diabetic_data", step="group_diagnosis_codes",
               fields=["diag_1", "diag_2", "diag_3"])
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    log_access("clean", "diabetic_data", step="binarize_target",
               positive_rate=round(df["readmitted_30d"].mean(), 4))
    return df


def run_cleaning_pipeline(df: pd.DataFrame, id_mappings: dict) -> pd.DataFrame:
    df = recode_missing(df)
    df = drop_high_missingness_fields(df)
    df = impute_low_missingness_fields(df)
    df = drop_deceased_hospice(df, id_mappings)
    df = apply_cohort_filter(df)
    df = group_diagnosis_codes(df)
    df = binarize_target(df)
    return df
