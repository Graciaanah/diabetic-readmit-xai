"""
Bias Detection Suite (representation bias).

This is distinct from Module 4's model-fairness audit (which checks a
trained model's recall/FPR parity via Fairlearn). Here, BEFORE any model
exists, we check whether the cohort-filtering step itself introduced
representation bias -- i.e., does the high-utilization cohort systematically
over- or under-represent any demographic group relative to the full raw
population, in a way that could bias what the eventual model learns.

Uses Fairlearn's MetricFrame purely as a representation-rate comparison
tool here (no model predictions involved yet).
"""
from pathlib import Path
import pandas as pd
from fairlearn.metrics import MetricFrame, count

RAW_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "diabetic_data.csv"
PROCESSED_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "cleaned_cohort.csv"
REPORT_FILE = Path(__file__).resolve().parents[2] / "reports" / "bias_detection_report.txt"

K_THRESHOLD = 5  # matches the k-anonymity suppression threshold used elsewhere


def representation_rates(raw: pd.DataFrame, cohort: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    For each subgroup in group_col, compute:
      - raw_share: % of the full raw population in this subgroup
      - cohort_share: % of the filtered high-utilization cohort in this subgroup
      - representation_ratio: cohort_share / raw_share (1.0 = proportional representation)
    """
    raw_counts = raw[group_col].value_counts()
    cohort_counts = cohort[group_col].value_counts()

    raw_share = raw_counts / raw_counts.sum()
    cohort_share = cohort_counts / cohort_counts.sum()

    out = pd.DataFrame({
        "raw_count": raw_counts,
        "raw_share": raw_share,
        "cohort_count": cohort_counts,
        "cohort_share": cohort_share,
    }).fillna(0)
    out["representation_ratio"] = (out["cohort_share"] / out["raw_share"]).round(3)

    # k-anonymity: suppress any subgroup with fewer than k cohort records
    out = out.astype({"cohort_count": "object", "cohort_share": "object",
                       "representation_ratio": "object"})
    small = out["cohort_count"].astype(float) < K_THRESHOLD
    out.loc[small, ["cohort_count", "cohort_share", "representation_ratio"]] = f"<{K_THRESHOLD}"

    return out.sort_values("raw_count", ascending=False)


def run_bias_detection() -> dict:
    raw = pd.read_csv(RAW_FILE).replace("?", pd.NA)
    cohort = pd.read_csv(PROCESSED_FILE)

    # race is imputed to the literal string "Unknown" in the cleaned cohort
    # (see clean.py); apply the same relabeling to raw's nulls here so the
    # raw-vs-cohort comparison is apples-to-apples rather than comparing a
    # true NaN against a literal "Unknown" category.
    raw["race"] = raw["race"].fillna("Unknown")

    results = {}
    for group_col in ["race", "gender", "age"]:
        results[group_col] = representation_rates(raw, cohort, group_col)

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("Bias Detection Report -- Representation Bias in Cohort Filtering\n")
        f.write("=" * 70 + "\n")
        f.write("Compares the high-utilization cohort's demographic composition to\n")
        f.write("the full raw population. representation_ratio far from 1.0 flags a\n")
        f.write("subgroup that is over- or under-represented after cohort filtering.\n")
        f.write(f"Subgroups with fewer than {K_THRESHOLD} cohort records are suppressed.\n\n")
        for group_col, table in results.items():
            f.write(f"\n--- {group_col} ---\n")
            f.write(table.to_string())
            f.write("\n")

    print(f"Bias detection report written to {REPORT_FILE}")
    for group_col, table in results.items():
        print(f"\n--- {group_col} ---")
        print(table[["raw_share", "cohort_share", "representation_ratio"]])

    return results


if __name__ == "__main__":
    run_bias_detection()
