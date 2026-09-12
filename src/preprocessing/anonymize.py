"""
Anonymization Script.

Implements the Data Privacy Plan from the Module 1 Ethical AI Charter and
Module 2 Privacy, Fairness & Compliance section:
  - Direct identifiers (encounter_id, patient_nbr) are dropped before any
    data leaves this stage.
  - Small-cell suppression (k=5): any reporting breakdown -- demographic
    or otherwise -- with fewer than 5 records in a cell is suppressed.
  - Patient-level split key is generated here (a non-reversible surrogate)
    so downstream modeling can still do patient-level train/test splits
    (see clean.py cohort logic) without retaining the real patient_nbr.
"""
import hashlib
import pandas as pd

from .audit_log import log_access

DIRECT_IDENTIFIERS = ["encounter_id", "patient_nbr"]
K_THRESHOLD = 5


def make_patient_split_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-way hash of patient_nbr, used ONLY to keep repeat encounters from
    the same patient in the same train/test split. Cannot be reversed to
    recover the original patient_nbr.
    """
    df = df.copy()
    df["patient_split_key"] = df["patient_nbr"].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )
    return df


def drop_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DIRECT_IDENTIFIERS if c in df.columns]
    df = df.drop(columns=present)
    log_access("anonymize", "diabetic_data", step="drop_identifiers", dropped=present)
    return df


def suppress_small_cells(summary: pd.DataFrame, count_col: str,
                          k: int = K_THRESHOLD) -> pd.DataFrame:
    """
    Apply k-anonymity suppression to an aggregated reporting table (e.g. a
    subgroup breakdown by race/age/gender). Any row with a count below k is
    replaced with a suppression marker rather than the true value, and the
    original count is never exposed downstream.
    """
    out = summary.copy()
    suppressed_mask = out[count_col] < k
    n_suppressed = int(suppressed_mask.sum())
    other_cols = out.columns.difference([count_col])
    out = out.astype({**{c: "object" for c in other_cols}, count_col: "object"})
    out.loc[suppressed_mask, other_cols] = "<suppressed>"
    out.loc[suppressed_mask, count_col] = f"<{k}"
    log_access("anonymize", "reporting_table", step="k_anonymity_suppression",
               k=k, rows_suppressed=n_suppressed, rows_total=len(summary))
    return out


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    df = make_patient_split_key(df)
    df = drop_direct_identifiers(df)
    return df


if __name__ == "__main__":
    # Demonstration: suppression applied to a small demographic cross-tab
    demo = pd.DataFrame({
        "race": ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"],
        "age_band": ["[70-80)"] * 5,
        "count": [812, 340, 3, 44, 2],
    })
    print(suppress_small_cells(demo, "count"))
