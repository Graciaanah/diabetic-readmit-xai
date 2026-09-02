# Diabetic Readmission Risk Stratification

Capstone analytics project — MSc Data Analytics, Nexford University

## Problem

Hospital discharge planning teams have no data-driven way to prioritize post-discharge follow-up for diabetic patients with high prior-year utilization (2+ inpatient/ED visits) who are at elevated risk of 30-day readmission. This project builds a risk-stratification tool that scores that risk and converts it into a capacity-constrained, ranked follow-up shortlist.

Full problem definition, vision, and ethical foundations: [`docs/vision-document.md`](docs/vision-document.md)
Planning, architecture, and delivery plan: [`docs/module2-project-overview.md`](docs/module2-project-overview.md)

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (101,766 encounters, 130 U.S. hospitals). Raw CSVs are gitignored — download `diabetic_data.csv` and `IDS_mapping.csv` and place them in `data/raw/` before running the pipeline.

## Repository Structure

```
capstone-diabetic-readmission/
├── data/
│   ├── raw/            # diabetic_data.csv, IDS_mapping.csv (gitignored)
│   └── processed/      # cleaned_cohort.csv (gitignored)
├── notebooks/          # exploratory analysis
├── src/
│   ├── preprocessing/  # ingest.py, clean.py, anonymize.py, transform.py, dag.py, audit_log.py
│   │   └── validation/ # validate.py (Great Expectations suite)
│   ├── modeling/       # predictive risk score + prescriptive shortlist (Module 4)
│   └── fairness/       # bias_detection.py (Module 3); model fairness audit (Module 4)
├── tests/               # pytest unit tests
├── reports/             # validation_results.txt, bias_detection_report.txt, privacy_audit.log
├── dashboard/           # Streamlit dashboard app (Module 5)
├── docs/                # Vision Document, Project Overview Report, RAID log, data dictionary, governance framework, diagrams
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# Full orchestrated pipeline (ingest -> clean -> anonymize -> validate -> bias check)
python -m src.preprocessing.dag

# Individual stages
python -m src.preprocessing.transform          # ingest + clean + anonymize + write
python -m src.preprocessing.validation.validate # Great Expectations validation
python -m src.fairness.bias_detection           # representation bias check

# Unit tests
pytest tests/ -v

# Containerized run
docker build -t diabetic-readmission-pipeline .
docker run -v $(pwd)/data:/app/data diabetic-readmission-pipeline
```

## Project Status

| Module | Deliverable | Status |
|---|---|---|
| Module 1 | Vision Document with ethical foundations | ✅ Complete |
| Module 2 | Planning, architecture, RAID log, sprint board | ✅ Complete |
| Module 3 | Data pipeline: ingestion, cleaning, validation, bias detection, governance | ✅ Complete |
| Module 4 | Predictive modeling, MLflow tracking, SHAP/LIME/counterfactuals, fairness audit, bias mitigation, FastAPI | ✅ Complete |
| Module 5 | Stakeholder presentation, Q&A backup deck, live interactive dashboard | ✅ Complete |
| Final Project | Full integration and deployment | ⏳ Upcoming |

## Key Documents

- [Vision Document](docs/Vision_Document.docx) — vision, goals, ethical charter, stakeholders
- [Module 2 Project Overview Report](docs/Module2_Project_Overview_Report.docx) — root cause analysis, architecture, sprint plan, privacy/fairness/compliance
- [Technical RAID Log](docs/Technical_RAID_Log.xlsx) — risks, assumptions, issues, dependencies
- [Data Dictionary](docs/data-dictionary.md) — schema of the cleaned cohort output
- [Data Governance Framework](docs/data-governance-framework.md) — access, usage, retention, k-anonymity policy
- Architecture and root-cause diagrams: [`docs/assets/`](docs/assets/)

## Pipeline Results (real run against the dataset)

- Raw dataset: 101,766 encounters → cohort filter (2+ prior inpatient/ED visits, deceased/hospice excluded) → **17,983 encounters** in the final cohort
- Great Expectations validation: **10/10 checks passed**, 5/5 schema checks passed
- Unit tests: **13/13 passed**
- Bias detection flagged two representation shifts worth noting in Module 4: African-American patients (ratio ~1.16) and the 20–30 age band (ratio ~1.59) are over-represented in the high-utilization cohort relative to the raw population — see `reports/bias_detection_report.txt`

## Module 5: Stakeholder Presentation & Dashboard

**Live Dashboard:** https://diabetic-readmit-xai-jlfoaphnrgsetvppdwxhsf.streamlit.app/

- **Live interactive dashboard** (`dashboard/app.py`): 6 sections translating Module 4's results, explainability, and fairness findings into plain language for non-technical stakeholders (discharge planning, administration, ethics/compliance), including a live What-If Analysis tool
- **14-slide stakeholder presentation**: [`docs/Module5_Stakeholder_Presentation.pptx`](docs/Module5_Stakeholder_Presentation.pptx) — no recommendation or executive summary, per this module's scope
- **Q&A Backup Deck** (technical appendix): [`docs/Module5_QA_Backup_Deck.pptx`](docs/Module5_QA_Backup_Deck.pptx) — full metrics, fairness methodology, mitigation detail
- **API demo script**: `api/api_demo.py` — run against a live `uvicorn api.main:app` server to see the `/predict` endpoint in action

### Running the Dashboard Locally

```bash
streamlit run dashboard/app.py
```

### Deploying to Streamlit Cloud

1. Push this repo to GitHub (already done — see below)
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click "New app", select this repo, set the main file path to `dashboard/app.py`
4. If the default root `requirements.txt` is slow to install, use "Advanced settings" to point to `dashboard/requirements.txt` instead — a lean file with only what the dashboard actually needs

### Running the API Demo

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
python api/api_demo.py
```

## Module 4: Predictive Modeling Results

- 3 models trained (Logistic Regression, Random Forest, XGBoost) with MLflow experiment tracking; **XGBoost selected** (ROC-AUC 0.596, recall@20% 31.2%, registered in MLflow Model Registry as `diabetic_readmission_xgboost`, alias `champion`)
- **Recall@20% (31.2%) falls short of the Module 1 target (75%)** — reported as a genuine finding, not adjusted
- SHAP, LIME, and DiceML counterfactual explanations all working — see `notebooks/shap_analysis.ipynb` and `reports/`
- Fairness audit (Fairlearn): race and age both flagged for disparate impact on the deployed top-20% shortlist decision; two bias mitigation attempts tested and **both failed** to produce a deployable result (documented honestly in `reports/bias_mitigation_report.txt`)
- FastAPI `/predict` endpoint tested live and working — see `api/main.py`
- Full results, figures, and interpretation: [`docs/Module4_Model_Validation_Report.docx`](docs/Module4_Model_Validation_Report.docx) and [`docs/model-card.md`](docs/model-card.md)

### Running the Model Pipeline

```bash
python -m src.modeling.train              # train all 3 models, MLflow tracking
python -m src.modeling.validation_plots    # ROC curves, confusion matrices
python -m src.modeling.shap_analysis       # SHAP global + local explanations
python -m src.modeling.lime_analysis       # LIME local explanations
python -m src.modeling.counterfactual      # DiceML counterfactuals
python -m src.fairness.fairness_metrics    # Fairlearn fairness audit
python -m src.fairness.bias_mitigation     # ThresholdOptimizer mitigation attempts
python -m src.modeling.sensitivity         # Sensitivity analysis
python -m src.modeling.register_model      # Register best model in MLflow registry

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Run the API
uvicorn api.main:app --reload --port 8000
# Then visit http://localhost:8000/docs
```

## Ethical Principles

This project is governed by three principles from the Ethical AI Charter (see Vision Document, Section 3):
1. **Explainability Over Black-Box Accuracy**
2. **Equity of Access Over Aggregate Performance**
3. **Human-in-the-Loop, Not Human-Replaced**

Fairness is audited per demographic subgroup (race, age band, gender) at every model iteration using Fairlearn (Demographic Parity, Equalized Odds).
