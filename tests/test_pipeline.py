"""
Unit tests for the data preprocessing pipeline.
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.clean import (
    recode_missing, drop_high_missingness_fields, impute_low_missingness_fields,
    apply_cohort_filter, _icd9_category, group_diagnosis_codes, binarize_target,
)
from src.preprocessing.anonymize import (
    drop_direct_identifiers, suppress_small_cells, make_patient_split_key,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [100, 100, 200, 300],
        "race": ["Caucasian", "?", "AfricanAmerican", "Asian"],
        "weight": ["?", "?", "?", "?"],
        "payer_code": ["?", "MC", "?", "?"],
        "max_glu_serum": ["?", "?", "None", "?"],
        "diag_1": ["250.83", "?", "410", "V10"],
        "diag_2": ["?", "?", "?", "?"],
        "diag_3": ["?", "?", "?", "?"],
        "number_inpatient": [2, 0, 3, 1],
        "number_emergency": [0, 1, 0, 0],
        "readmitted": ["<30", "NO", ">30", "<30"],
    })


def test_recode_missing_converts_question_marks(sample_df):
    out = recode_missing(sample_df)
    assert out["race"].isna().sum() == 1
    assert out["weight"].isna().sum() == 4


def test_drop_high_missingness_fields_removes_expected_columns(sample_df):
    out = drop_high_missingness_fields(sample_df)
    for col in ["weight", "payer_code", "max_glu_serum"]:
        assert col not in out.columns


def test_impute_low_missingness_fields_fills_race_and_diag(sample_df):
    df = recode_missing(sample_df)
    out = impute_low_missingness_fields(df)
    assert out["race"].isna().sum() == 0
    assert (out["race"] == "Unknown").sum() == 1
    assert out["diag_1"].isna().sum() == 0
    assert (out["diag_1"] == "Missing").sum() == 1


def test_cohort_filter_keeps_only_high_utilization_patients(sample_df):
    out = apply_cohort_filter(sample_df)
    # rows with (inpatient + emergency) >= 2: row0 (2+0=2), row2 (3+0=3)
    assert len(out) == 2
    assert set(out["encounter_id"]) == {1, 3}


def test_icd9_category_diabetes():
    assert _icd9_category("250.83") == "Diabetes"


def test_icd9_category_circulatory():
    assert _icd9_category("410") == "Circulatory"


def test_icd9_category_external_supplemental():
    assert _icd9_category("V10") == "External/Supplemental"


def test_icd9_category_missing():
    assert _icd9_category(np.nan) == "Missing"
    assert _icd9_category("Missing") == "Missing"


def test_group_diagnosis_codes_adds_category_columns(sample_df):
    df = recode_missing(sample_df)
    df = impute_low_missingness_fields(df)
    out = group_diagnosis_codes(df)
    assert "diag_1_category" in out.columns
    assert out.loc[0, "diag_1_category"] == "Diabetes"


def test_binarize_target_creates_correct_flag(sample_df):
    out = binarize_target(sample_df)
    assert out["readmitted_30d"].tolist() == [1, 0, 0, 1]


def test_drop_direct_identifiers_removes_ids(sample_df):
    out = drop_direct_identifiers(sample_df)
    assert "encounter_id" not in out.columns
    assert "patient_nbr" not in out.columns


def test_make_patient_split_key_is_deterministic_and_not_reversible(sample_df):
    out = make_patient_split_key(sample_df)
    # same patient_nbr (100) must get the same split key both times
    assert out.loc[0, "patient_split_key"] == out.loc[1, "patient_split_key"]
    # different patients must get different keys
    assert out.loc[0, "patient_split_key"] != out.loc[2, "patient_split_key"]
    # key must not just be the original id stringified
    assert out.loc[0, "patient_split_key"] != "100"


def test_suppress_small_cells_hides_low_count_rows():
    demo = pd.DataFrame({
        "race": ["A", "B"],
        "count": [812, 3],
    })
    out = suppress_small_cells(demo, "count", k=5)
    assert out.loc[0, "count"] == 812
    assert out.loc[1, "count"] == "<5"
    assert out.loc[1, "race"] == "<suppressed>"
