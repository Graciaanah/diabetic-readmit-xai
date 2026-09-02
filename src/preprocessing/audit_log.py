"""
Privacy Audit Logging.

Every pipeline stage that touches data (ingestion, cleaning, anonymization,
transformation) calls log_access() so there is a durable, timestamped record
of what happened to the data and when. This satisfies the Module 3 Privacy
Audit Logging requirement and extends the Module 1/2 Data Governance
Principles (least-privilege access, no undocumented handling).

Logs are written to reports/privacy_audit.log (gitignored contents, but the
directory and mechanism are committed) so this can run against real data
without shipping a log full of dataset specifics to GitHub.
"""
import logging
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path(__file__).resolve().parents[2] / "reports"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "privacy_audit.log"

logger = logging.getLogger("privacy_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    # Also echo to console so pipeline runs are visible without tailing the file
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[audit] %(message)s"))
    logger.addHandler(console)


def log_access(stage: str, dataset: str, **details) -> None:
    """
    Record a data access or transformation event.

    stage: pipeline stage name, e.g. "ingest", "clean", "anonymize", "transform"
    dataset: which file/table was touched
    details: arbitrary key=value context (row counts, columns dropped, etc.)
             -- never pass raw field values here, only metadata about the
             operation, so the audit log itself never becomes a privacy risk.
    """
    ts = datetime.now(timezone.utc).isoformat()
    detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"stage={stage} dataset={dataset} {detail_str}".strip())


def read_audit_trail() -> list[str]:
    """Return all logged events, for inclusion in governance reporting."""
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text().splitlines()
