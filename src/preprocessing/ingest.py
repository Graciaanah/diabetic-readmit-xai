"""
Stage 1: Ingestion.

Loads the raw Diabetes 130-US Hospitals CSVs from data/raw/ and returns
them as DataFrames. No transformation happens here -- this stage's only
job is getting raw bytes into memory reliably and logging that it happened
(see audit_log.py for the privacy-audit trail).
"""
from pathlib import Path
import pandas as pd

from .audit_log import log_access

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ENCOUNTERS_FILE = RAW_DIR / "diabetic_data.csv"
ID_MAPPING_FILE = RAW_DIR / "IDS_mapping.csv"


def load_encounters() -> pd.DataFrame:
    """Load the main encounter-level dataset."""
    if not ENCOUNTERS_FILE.exists():
        raise FileNotFoundError(
            f"{ENCOUNTERS_FILE} not found. Download diabetic_data.csv from "
            "https://archive.ics.uci.edu/dataset/296 and place it in data/raw/."
        )
    df = pd.read_csv(ENCOUNTERS_FILE)
    log_access("ingest", "diabetic_data.csv", rows=len(df), columns=len(df.columns))
    return df


def load_id_mappings() -> dict:
    """
    Parse the IDS_mapping.csv file into a dict of lookup tables, one per
    coded ID field (admission_type_id, discharge_disposition_id,
    admission_source_id). The source file stacks three small tables with
    blank-line separators, so we split on blank rows.
    """
    if not ID_MAPPING_FILE.exists():
        raise FileNotFoundError(f"{ID_MAPPING_FILE} not found.")

    raw_lines = ID_MAPPING_FILE.read_text().splitlines()
    tables, current, name = {}, [], None
    for line in raw_lines:
        if line.strip() in ("", ","):
            if current and name:
                tables[name] = current
            current, name = [], None
            continue
        if name is None:
            name = line.split(",")[0]
            continue
        current.append(line)
    if current and name:
        tables[name] = current

    parsed = {}
    for key, lines in tables.items():
        rows = [l.split(",", 1) for l in lines]
        parsed[key] = {int(r[0]): r[1] for r in rows if r[0].strip().isdigit()}

    log_access("ingest", "IDS_mapping.csv", tables=list(parsed.keys()))
    return parsed


if __name__ == "__main__":
    encounters = load_encounters()
    mappings = load_id_mappings()
    print(f"Loaded {len(encounters):,} encounters, {len(encounters.columns)} columns")
    print(f"Parsed ID mapping tables: {list(mappings.keys())}")
