"""
Stage 4: Transformation & Integration -- pipeline entry point.

Runs the full ingest -> clean -> anonymize sequence and writes the final
analysis-ready table to data/processed/. This is the function the
orchestration DAG (dag.py) calls, and what the Great Expectations suite
validates the output of.
"""
from pathlib import Path
import pandas as pd

from .ingest import load_encounters, load_id_mappings
from .clean import run_cleaning_pipeline
from .anonymize import anonymize
from .audit_log import log_access

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_cohort.csv"


def run_pipeline() -> pd.DataFrame:
    """Execute the full pipeline and persist the output. Returns the final DataFrame."""
    encounters = load_encounters()
    id_mappings = load_id_mappings()
    cleaned = run_cleaning_pipeline(encounters, id_mappings)
    final = anonymize(cleaned)

    PROCESSED_DIR.mkdir(exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)
    log_access("transform", "cleaned_cohort.csv", rows=len(final),
               columns=len(final.columns), output_path=str(OUTPUT_FILE))
    return final


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Pipeline complete: {len(result):,} rows written to {OUTPUT_FILE}")
