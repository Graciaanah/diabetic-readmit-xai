"""
Orchestration DAG (Prefect).

Defines the ETL workflow as a Prefect flow: each pipeline stage is a task,
with explicit dependencies so the DAG shape matches the architecture
diagram in the Module 2 Project Overview Report (docs/module2-project-overview.md,
Figure 2): ingest -> clean -> anonymize -> validate -> bias_check.

Run with: python -m src.preprocessing.dag
"""
import sys
from pathlib import Path
from prefect import flow, task

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.ingest import load_encounters, load_id_mappings
from src.preprocessing.clean import run_cleaning_pipeline
from src.preprocessing.anonymize import anonymize
from src.preprocessing.transform import PROCESSED_DIR, OUTPUT_FILE
from src.preprocessing.validation.validate import run_validation
from src.fairness.bias_detection import run_bias_detection


@task(name="ingest_encounters", retries=1)
def t_ingest_encounters():
    return load_encounters()


@task(name="ingest_id_mappings", retries=1)
def t_ingest_id_mappings():
    return load_id_mappings()


@task(name="clean")
def t_clean(encounters, id_mappings):
    return run_cleaning_pipeline(encounters, id_mappings)


@task(name="anonymize")
def t_anonymize(cleaned):
    return anonymize(cleaned)


@task(name="write_processed")
def t_write_processed(final_df):
    PROCESSED_DIR.mkdir(exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    return str(OUTPUT_FILE)


@task(name="validate")
def t_validate(_dependency):
    success = run_validation()
    if not success:
        raise ValueError("Data validation failed -- see reports/validation_results.txt")
    return success


@task(name="bias_check")
def t_bias_check(_dependency):
    return run_bias_detection()


@flow(name="diabetic_readmission_pipeline")
def pipeline_flow():
    encounters = t_ingest_encounters()
    id_mappings = t_ingest_id_mappings()
    cleaned = t_clean(encounters, id_mappings)
    final = t_anonymize(cleaned)
    output_path = t_write_processed(final)
    t_validate(output_path)
    t_bias_check(output_path)
    return output_path


if __name__ == "__main__":
    result = pipeline_flow()
    print(f"DAG run complete. Output: {result}")
