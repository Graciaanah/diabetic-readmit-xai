"""
API Demo Script (Module 5 Technical Requirement).

Demonstrates the Module 4 FastAPI /predict endpoint in action, using the
same three illustrative patient profiles shown in the dashboard's What-If
Analysis section, so a stakeholder can see the same numbers presented two
different ways: through the dashboard UI and through the underlying API
a real hospital IT system would actually call.

Run the API first (from the repo root):
    uvicorn api.main:app --reload --port 8000

Then run this script in a separate terminal:
    python api_demo.py
"""
import json
import time
import requests

BASE_URL = "http://localhost:8000"

PROFILES = {
    "Lower risk": {
        "time_in_hospital": 2, "num_lab_procedures": 28, "num_procedures": 1, "num_medications": 8,
        "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 4,
        "insulin": "No", "diabetesMed": "No", "A1Cresult": "Norm",
    },
    "Moderate risk": {
        "time_in_hospital": 4, "num_lab_procedures": 42, "num_procedures": 2, "num_medications": 14,
        "number_outpatient": 1, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 7,
        "insulin": "Steady", "diabetesMed": "Yes", "A1Cresult": "Not Tested",
    },
    "Higher risk": {
        "time_in_hospital": 9, "num_lab_procedures": 58, "num_procedures": 4, "num_medications": 19,
        "number_outpatient": 2, "number_emergency": 3, "number_inpatient": 5, "number_diagnoses": 9,
        "insulin": "Up", "diabetesMed": "Yes", "A1Cresult": "Not Tested",
    },
}


def check_health():
    print("=" * 60)
    print("STEP 1: Health check")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/health")
    print(f"GET {BASE_URL}/health -> {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    print()


def demo_predictions():
    print("=" * 60)
    print("STEP 2: Predictions for three illustrative patient profiles")
    print("=" * 60)
    print("(These are the same profiles shown in the dashboard's")
    print(" What-If Analysis section -- not real patient records.)\n")

    for name, payload in PROFILES.items():
        resp = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"--- {name} ---")
        print(f"POST {BASE_URL}/predict")
        print("Request body:")
        print(json.dumps(payload, indent=2))
        print(f"\nResponse ({resp.status_code}):")
        print(json.dumps(resp.json(), indent=2))
        print()
        time.sleep(0.3)


def demo_invalid_request():
    print("=" * 60)
    print("STEP 3: Invalid request handling (missing required field)")
    print("=" * 60)
    bad_payload = {"time_in_hospital": 2}  # missing several required fields
    resp = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    print(f"POST {BASE_URL}/predict")
    print("Request body (intentionally incomplete):")
    print(json.dumps(bad_payload, indent=2))
    print(f"\nResponse ({resp.status_code}) -- correctly rejected, not silently guessed:")
    print(json.dumps(resp.json(), indent=2))
    print()


if __name__ == "__main__":
    try:
        check_health()
        demo_predictions()
        demo_invalid_request()
        print("=" * 60)
        print("Demo complete. See docs/model-card.md for full model")
        print("documentation and reports/ for the underlying analysis.")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API.")
        print("Start it first, from the repo root, in a separate terminal:")
        print("    uvicorn api.main:app --reload --port 8000")
