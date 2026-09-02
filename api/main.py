"""
FastAPI application serving the diabetic readmission risk model.
Run with: uvicorn api.main:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="Diabetic Readmission Risk API",
    description="Predicts 30-day readmission risk for high-utilization diabetic patients. "
                 "Decision support only -- see Model Card for limitations and required human oversight.",
    version="1.0.0",
)

_model = None
_scaler = None
_feature_columns = None
_numeric_cols = None
_categorical_options = None


def load_artifacts():
    global _model, _scaler, _feature_columns
    if _model is None:
        _model = joblib.load(MODEL_DIR / "xgboost.joblib")
        _scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        _feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return _model, _scaler, _feature_columns


class PatientEncounter(BaseModel):
    """
    A minimal, clinically-relevant subset of encounter fields. Fields not
    provided default to 0 / "None" / "No" (the most common category),
    since a production system would populate the full 142-feature vector
    from the EHR directly -- this schema is intentionally simplified for
    a demonstrable /predict endpoint, not the full raw pipeline input.
    """
    time_in_hospital: int = Field(..., ge=1, le=14, description="Length of stay in days")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab tests performed")
    num_procedures: int = Field(..., ge=0, description="Number of procedures performed")
    num_medications: int = Field(..., ge=0, description="Number of distinct medications administered")
    number_outpatient: int = Field(0, ge=0, description="Outpatient visits in preceding year")
    number_emergency: int = Field(0, ge=0, description="Emergency visits in preceding year")
    number_inpatient: int = Field(..., ge=0, description="Inpatient visits in preceding year (cohort-defining field)")
    number_diagnoses: int = Field(..., ge=1, description="Number of diagnoses recorded")
    admission_type_id: int = Field(1, description="Coded admission type (see IDS_mapping.csv)")
    discharge_disposition_id: int = Field(1, description="Coded discharge disposition")
    admission_source_id: int = Field(1, description="Coded admission source")
    insulin: str = Field("No", description="Insulin dosage change: No, Steady, Up, Down")
    diabetesMed: str = Field("Yes", description="Whether a diabetes medication was prescribed: Yes/No")
    A1Cresult: str = Field("Not Tested", description="HbA1c test result: Not Tested, Norm, >7, >8")


class PredictionResponse(BaseModel):
    predicted_risk: float
    risk_tier: str
    in_top_20pct_shortlist_threshold: bool
    model_version: str = "xgboost_v1.0.0"
    disclaimer: str = (
        "Decision support only. Not validated for clinical deployment. "
        "See Model Card for fairness limitations (race/age disparities flagged, unmitigated)."
    )


# Approximate threshold for "top 20%" shortlist inclusion, taken from the
# Module 4 training run (see reports/fairness_metrics_report.txt).
SHORTLIST_THRESHOLD = 0.24


def build_feature_vector(encounter: PatientEncounter, feature_columns: list, scaler) -> pd.DataFrame:
    row = {col: 0 for col in feature_columns}

    numeric_raw = {
        "admission_type_id": encounter.admission_type_id,
        "discharge_disposition_id": encounter.discharge_disposition_id,
        "admission_source_id": encounter.admission_source_id,
        "time_in_hospital": encounter.time_in_hospital,
        "num_lab_procedures": encounter.num_lab_procedures,
        "num_procedures": encounter.num_procedures,
        "num_medications": encounter.num_medications,
        "number_outpatient": encounter.number_outpatient,
        "number_emergency": encounter.number_emergency,
        "number_inpatient": encounter.number_inpatient,
        "number_diagnoses": encounter.number_diagnoses,
    }
    for col, val in numeric_raw.items():
        if col in row:
            row[col] = val

    # one-hot flags for the categorical inputs we accept
    insulin_col = f"insulin_{encounter.insulin}"
    if insulin_col in row:
        row[insulin_col] = 1
    diabetesmed_col = f"diabetesMed_{encounter.diabetesMed}"
    if diabetesmed_col in row:
        row[diabetesmed_col] = 1
    a1c_col = f"A1Cresult_{encounter.A1Cresult}"
    if a1c_col in row:
        row[a1c_col] = 1

    df = pd.DataFrame([row], columns=feature_columns)

    # Scale the same numeric columns the training pipeline scaled
    numeric_cols = [c for c in numeric_raw.keys() if c in feature_columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0)) if hasattr(scaler, "transform") else df[numeric_cols]
    return df


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def root():
    return {
        "service": "Diabetic Readmission Risk API",
        "model_card": "See docs/model-card.md in the repository",
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: PatientEncounter):
    model, scaler, feature_columns = load_artifacts()
    try:
        X = build_feature_vector(encounter, feature_columns, scaler)
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    if proba >= 0.5:
        tier = "high"
    elif proba >= SHORTLIST_THRESHOLD:
        tier = "elevated"
    else:
        tier = "low"

    return PredictionResponse(
        predicted_risk=round(proba, 4),
        risk_tier=tier,
        in_top_20pct_shortlist_threshold=proba >= SHORTLIST_THRESHOLD,
    )
