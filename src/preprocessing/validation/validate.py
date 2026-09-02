"""
Data Validation Suite (Great Expectations).

Validates the pipeline's output (data/processed/cleaned_cohort.csv) against
a set of expectations derived directly from the data-quality decisions
documented in Modules 1-2. This runs AFTER the pipeline, as a gate: if the
output doesn't meet these expectations, something upstream broke silently
and the pipeline should not be trusted for modeling in Module 4.
"""
from pathlib import Path
import sys
import great_expectations as gx
import pandas as pd

PROCESSED_FILE = Path(__file__).resolve().parents[3] / "data" / "processed" / "cleaned_cohort.csv"
RESULTS_FILE = Path(__file__).resolve().parents[3] / "reports" / "validation_results.txt"


def build_suite(context: gx.data_context.AbstractDataContext) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="cleaned_cohort_suite")
    suite = context.suites.add(suite)

    # Row-count floor: must clear the course's 5,000-record minimum
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=5000))

    # Imputed fields should have no remaining nulls
    for col in ["race", "diag_1", "diag_2", "diag_3", "medical_specialty", "A1Cresult"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # Cohort filter: every remaining row must actually qualify
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="time_in_hospital", min_value=1, max_value=14))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="readmitted_30d", value_set=[0, 1]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="diag_1_category",
        value_set=["Diabetes", "Circulatory", "Respiratory", "Digestive",
                   "Genitourinary", "Injury/Poisoning", "Musculoskeletal",
                   "Neoplasms", "External/Supplemental", "Other", "Missing"]))
    return suite


SCHEMA_ABSENT_COLUMNS = ["weight", "payer_code", "max_glu_serum",
                          "encounter_id", "patient_nbr"]


def check_schema_absent_columns(df: pd.DataFrame) -> dict:
    """
    Great Expectations 1.x has no 'column must not exist' expectation, so
    this structural check (dropped high-missingness fields + dropped direct
    identifiers) runs as a plain assertion, kept alongside the GE suite's
    value-level checks rather than inside it.
    """
    return {col: (col not in df.columns) for col in SCHEMA_ABSENT_COLUMNS}


def run_validation() -> bool:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{PROCESSED_FILE} not found. Run `python -m src.preprocessing.transform` first."
        )
    df = pd.read_csv(PROCESSED_FILE)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="cleaned_cohort")
    batch_def = data_asset.add_batch_definition_whole_dataframe("full_batch")

    suite = build_suite(context)
    schema_checks = check_schema_absent_columns(df)

    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    overall_success = result.success and all(schema_checks.values())

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Validation run against {PROCESSED_FILE}\n")
        f.write(f"Rows validated: {len(df):,}\n")
        f.write(f"Overall success: {overall_success}\n\n")
        f.write("Schema checks (column correctly absent):\n")
        for col, ok in schema_checks.items():
            f.write(f"  [{'PASS' if ok else 'FAIL'}] {col} not present\n")
        f.write("\nGreat Expectations suite results:\n")
        for r in result.results:
            status = "PASS" if r.success else "FAIL"
            f.write(f"  [{status}] {r.expectation_config.type} "
                    f"{r.expectation_config.kwargs.get('column', '')}\n")

    print(f"Overall success: {overall_success}")
    print("Schema checks:")
    for col, ok in schema_checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {col} not present")
    print("GE suite results:")
    for r in result.results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {r.expectation_config.type} "
              f"{r.expectation_config.kwargs.get('column', '')}")

    return overall_success


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
