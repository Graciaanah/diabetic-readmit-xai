"""
Diabetic Readmission Risk — Stakeholder Dashboard (Module 5)

A plain-language, interactive dashboard translating the Module 4 model's
results, explainability, and fairness findings for non-technical hospital
stakeholders (discharge planning leadership, quality improvement, ethics
review). Deployed standalone on Streamlit Community Cloud -- loads the
trained model directly rather than depending on the FastAPI service.

No recommendation or executive summary is included, per the Module 5
assignment brief: this dashboard's job is clear, honest communication of
findings, not a call to action.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------------------
# Page config & design tokens ("Teal Trust" palette, carried from
# Module 3's presentation deck for visual continuity across the capstone)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetic Readmission Risk — Stakeholder Dashboard",
    page_icon="\U0001FA7A",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL = "#028090"
SEAFOAM = "#00A896"
MINT = "#02C39A"
DARK = "#01353D"
INK = "#0B2027"
MUTED = "#5B7379"
AMBER = "#C4622D"
BG = "#F5F9F9"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ color: {INK}; font-family: Georgia, serif; }}
    .stat-card {{
        background: {DARK}; border-radius: 10px; padding: 20px 24px;
        color: white; margin-bottom: 10px;
    }}
    .stat-number {{ font-size: 2.2rem; font-weight: 700; color: {MINT}; margin: 0; }}
    .stat-label {{ font-size: 0.85rem; color: #BFE3E0; margin: 0; }}
    .plain-box {{
        background: white; border-left: 4px solid {SEAFOAM}; border-radius: 6px;
        padding: 16px 20px; margin: 10px 0;
    }}
    .limitation-box {{
        background: #FCEEE5; border-left: 4px solid {AMBER}; border-radius: 6px;
        padding: 16px 20px; margin: 10px 0; color: #7A3A18;
    }}
    .transparency-box {{
        background: {DARK}; color: white; border-radius: 10px; padding: 24px 28px;
    }}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_DIR / "xgboost.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, scaler, feature_columns


@st.cache_data
def load_sample_profiles():
    return pd.read_csv(Path(__file__).resolve().parent / "sample_profiles.csv")


NUMERIC_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id",
                 "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                 "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]

model, scaler, feature_columns = load_model_artifacts()
sample_profiles = load_sample_profiles()

# ---------------------------------------------------------------------
# Fixed reporting numbers (from Module 4's actual, verified results --
# reports/mlflow_runs_summary.csv, fairness_metrics_summary.csv)
# ---------------------------------------------------------------------
RESULTS = {
    "recall_at_20": 0.312, "lift_at_20": 1.56, "roc_auc": 0.596,
    "target_recall_at_20": 0.75, "revised_target_recall_at_20": 0.35,
    "revised_target_lift": 1.5,
}

TOP_DRIVERS = [
    ("Number of recent hospital stays", "The strongest signal by far. Patients with more inpatient visits in the past year are flagged as higher risk."),
    ("Where the patient goes after discharge", "Some discharge destinations (e.g., a skilled nursing facility vs. home) carry different readmission risk."),
    ("Length of the current hospital stay", "Longer stays are associated with somewhat higher risk."),
    ("Number of medications prescribed", "Patients on more medications during their stay tend to show elevated risk."),
    ("Recent emergency room visits", "Emergency visits in the past year add to the risk signal."),
]

FAIRNESS_SUMMARY = pd.DataFrame([
    {"Attribute": "Gender", "Finding": "No meaningful disparity found", "Status": "OK"},
    {"Attribute": "Race", "Finding": "Disparity found — flagged for review", "Status": "Flagged"},
    {"Attribute": "Age Group", "Finding": "Disparity found — flagged for review", "Status": "Flagged"},
])

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("\U0001FA7A Readmission Risk Tool")
st.sidebar.caption("Stakeholder Dashboard — Module 5")
section = st.sidebar.radio("Go to section", [
    "1. Business Problem",
    "2. Model & Key Results",
    "3. What Drives Predictions",
    "4. Try It — What-If Analysis",
    "5. Fairness & Ethics",
    "6. Limitations & Transparency",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard summarizes findings from a capstone analytics project. "
    "It is a decision-support illustration, not a validated clinical tool."
)

# ---------------------------------------------------------------------
# SECTION 1 — Business Problem
# ---------------------------------------------------------------------
if section == "1. Business Problem":
    st.title("Helping Discharge Planning Teams Prioritize Follow-Up Care")
    st.markdown(
        "<div class='plain-box'>Hospital discharge planning teams have no reliable way to tell, at the moment "
        "a patient leaves the hospital, which diabetic patients are most likely to be readmitted within 30 days. "
        "This project explores whether a data-driven tool can help prioritize limited follow-up capacity toward "
        "the patients who need it most.</div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Who this is for")
        st.markdown("""
        - **Discharge planning nurses & case managers** — deciding who gets a follow-up call
        - **Hospital administrators** — tracking readmission-related costs and penalties
        - **Ethics & compliance reviewers** — overseeing fair, responsible use of the tool
        """)
    with col2:
        st.subheader("The decision this tool supports")
        st.markdown("""
        With only enough staff time to follow up with a fraction of discharged patients each week,
        **who should get a call first?** Today, that decision is made by clinical judgment alone,
        under time pressure, with no consistent signal to work from.
        """)

    st.subheader("Focus population")
    st.markdown(
        "This analysis focuses on diabetic patients with a **high prior-year hospital utilization** "
        "(two or more inpatient or emergency visits in the year before their current stay) — "
        "the group most likely to benefit from prioritized follow-up."
    )

# ---------------------------------------------------------------------
# SECTION 2 — Model & Key Results
# ---------------------------------------------------------------------
elif section == "2. Model & Key Results":
    st.title("What the Model Does, in Plain Language")
    st.markdown(
        "<div class='plain-box'><b>Plain-language explanation:</b> the model looks at a patient's hospital "
        "record — how many times they've been admitted recently, how long their current stay is, how many "
        "medications they're on, and similar details — and produces a single risk score between 0 and 100%. "
        "Patients with the highest scores are placed on a prioritized follow-up list, sized to match the "
        "discharge team's actual weekly capacity.</div>", unsafe_allow_html=True
    )

    st.subheader("Key Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        pct_better = (RESULTS['lift_at_20'] - 1) * 100
        st.markdown(f"""<div class='stat-card'><p class='stat-number'>{pct_better:.0f}%</p>
        <p class='stat-label'>Better than random selection at finding truly high-risk patients within the same-size follow-up list</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='stat-card'><p class='stat-number'>{RESULTS['recall_at_20']*100:.0f}%</p>
        <p class='stat-label'>Of all patients who were actually readmitted, this share were caught in the top 20% follow-up list</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='stat-card'><p class='stat-number'>17,983</p>
        <p class='stat-label'>Patient records used to build and test this model</p></div>""", unsafe_allow_html=True)

    st.markdown("")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Random selection", "This model's shortlist"],
        y=[1.0, RESULTS["lift_at_20"]],
        marker_color=[MUTED, MINT],
        text=["1.0x (baseline)", f"{RESULTS['lift_at_20']}x"],
        textposition="outside",
    ))
    fig.update_layout(
        title="How much better is the model's shortlist than picking patients at random?",
        yaxis_title="Effectiveness (times better than random)",
        plot_bgcolor="white", height=380, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='limitation-box'><b>Honest context:</b> the model was originally targeted to catch 75% "
        f"of true readmissions within its follow-up shortlist. It currently reaches {RESULTS['recall_at_20']*100:.0f}%. "
        "This gap is discussed in full in Section 6 (Limitations & Transparency) — it is not hidden here.</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# SECTION 3 — What Drives Predictions
# ---------------------------------------------------------------------
elif section == "3. What Drives Predictions":
    st.title("What Drives a High-Risk Prediction?")
    st.markdown(
        "<div class='plain-box'>Technical explainability tools (SHAP and LIME) were used to determine which "
        "patient details most influence the model's risk score. Translated into plain language, "
        "the top drivers are:</div>", unsafe_allow_html=True
    )

    for i, (title, desc) in enumerate(TOP_DRIVERS, 1):
        st.markdown(f"**{i}. {title}**")
        st.caption(desc)

    st.subheader("Example: Two Patients, Different Risk Levels")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown("**Patient A — Lower risk (7% predicted)**")
        st.markdown(
            "- No hospital stays in the past year\n"
            "- Short current stay (2 days)\n"
            "- Fewer medications (8)\n\n"
            "*In plain terms: this patient's hospital history shows no recent pattern of frequent admissions, "
            "which is the strongest factor pushing their risk score down.*"
        )
    with ex2:
        st.markdown("**Patient C — Higher risk (38% predicted)**")
        st.markdown(
            "- 5 hospital stays in the past year\n"
            "- Longer current stay (9 days)\n"
            "- More medications (19), including a recent insulin dose increase\n\n"
            "*In plain terms: a pattern of frequent recent admissions, combined with a longer, more complex "
            "current stay, is what pushes this patient's risk score up.*"
        )
    st.caption("Full technical detail (SHAP/LIME values, counterfactual analysis) available in the Q&A Backup Deck.")

# ---------------------------------------------------------------------
# SECTION 4 — What-If Analysis
# ---------------------------------------------------------------------
elif section == "4. Try It — What-If Analysis":
    st.title("Try It Yourself: What-If Analysis")
    st.markdown(
        "<div class='plain-box'>Start from one of three illustrative patient profiles below, then adjust the "
        "sliders to see how the predicted risk changes. <b>These are illustrative example profiles, not real "
        "patient records</b> — built to demonstrate how the model responds to different inputs.</div>",
        unsafe_allow_html=True
    )

    profile_name = st.selectbox("Start from a profile:", sample_profiles["profile_name"].tolist(), index=1)
    base_row = sample_profiles[sample_profiles["profile_name"] == profile_name].iloc[0]

    # Reconstruct approximate original (unscaled) values for the sliders by
    # inverse-transforming the numeric columns back from the scaler
    base_scaled = base_row[NUMERIC_COLS].values.astype(float).reshape(1, -1)
    base_unscaled = scaler.inverse_transform(base_scaled)[0]
    base_values = dict(zip(NUMERIC_COLS, base_unscaled))

    st.subheader("Adjust patient details")
    c1, c2 = st.columns(2)
    with c1:
        number_inpatient = st.slider("Inpatient visits in the past year", 0, 10, int(round(base_values["number_inpatient"])))
        number_emergency = st.slider("Emergency visits in the past year", 0, 10, int(round(base_values["number_emergency"])))
        time_in_hospital = st.slider("Length of current stay (days)", 1, 14, int(round(base_values["time_in_hospital"])))
    with c2:
        num_medications = st.slider("Number of medications", 1, 30, int(round(base_values["num_medications"])))
        num_lab_procedures = st.slider("Number of lab tests performed", 1, 90, int(round(base_values["num_lab_procedures"])))
        num_procedures = st.slider("Number of procedures performed", 0, 6, int(round(base_values["num_procedures"])))

    # Build the modified row
    row = base_row.copy()
    overrides = {
        "number_inpatient": number_inpatient, "number_emergency": number_emergency,
        "time_in_hospital": time_in_hospital, "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
    }
    unscaled = base_unscaled.copy()
    for k, v in overrides.items():
        unscaled[NUMERIC_COLS.index(k)] = v
    scaled = scaler.transform(unscaled.reshape(1, -1))[0]
    row_features = row.drop("profile_name").copy()
    for i, col in enumerate(NUMERIC_COLS):
        row_features[col] = scaled[i]

    X_row = pd.DataFrame([row_features.values], columns=feature_columns).astype(float)
    new_risk = model.predict_proba(X_row)[0][1]
    base_row_features = base_row.drop("profile_name")
    X_base = pd.DataFrame([base_row_features.values], columns=feature_columns).astype(float)
    original_risk = model.predict_proba(X_base)[0][1]

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Original profile risk", f"{original_risk*100:.1f}%")
    r2.metric("Adjusted risk", f"{new_risk*100:.1f}%", delta=f"{(new_risk-original_risk)*100:+.1f} pts")
    shortlist_cutoff = 0.24
    r3.metric("In top-20% follow-up shortlist?", "Yes" if new_risk >= shortlist_cutoff else "No")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=new_risk*100,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": MINT},
               "steps": [{"range": [0, 24], "color": "#E4F5F3"}, {"range": [24, 100], "color": "#FCEEE5"}],
               "threshold": {"line": {"color": AMBER, "width": 3}, "value": shortlist_cutoff*100}},
        title={"text": "Predicted 30-Day Readmission Risk"},
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dashed line marks the approximate cutoff for the top-20% follow-up shortlist.")

# ---------------------------------------------------------------------
# SECTION 5 — Fairness & Ethics
# ---------------------------------------------------------------------
elif section == "5. Fairness & Ethics":
    st.title("Fairness Summary")
    st.markdown(
        "<div class='plain-box'>A fairness audit checked whether the model's follow-up shortlist treats patients "
        "differently based on race, gender, or age, in ways it shouldn't. This builds directly on the Ethical AI "
        "Charter and Privacy Plan established earlier in this project.</div>", unsafe_allow_html=True
    )

    st.subheader("Ethical Compliance Dashboard")
    for _, row in FAIRNESS_SUMMARY.iterrows():
        color = MINT if row["Status"] == "OK" else AMBER
        icon = "\u2705" if row["Status"] == "OK" else "\u26A0\uFE0F"
        st.markdown(
            f"<div style='background:white; border-left:4px solid {color}; border-radius:6px; "
            f"padding:14px 18px; margin:8px 0;'><b>{icon} {row['Attribute']}</b> — {row['Finding']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("")
    fairness_df = pd.DataFrame({
        "Group": ["Caucasian", "African-American", "Hispanic", "Younger patients (20-30)", "Older patients (80-90)"],
        "Selected for follow-up": [18.4, 24.6, 25.0, 38.8, 14.3],
    })
    fig = px.bar(fairness_df, x="Group", y="Selected for follow-up", color="Selected for follow-up",
                 color_continuous_scale=[SEAFOAM, AMBER], title="Share of Each Group Selected for Follow-Up (%)")
    fig.update_layout(plot_bgcolor="white", height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What Was Found & How It Was Addressed")
    st.markdown("""
    - A gap was found: some race and age groups are selected for follow-up at noticeably different rates
      than others, even though they may not have proportionally different actual readmission rates.
    - Two technical corrections were tested to close this gap automatically. **Both were tried honestly and
      both failed** — one badly hurt the tool's ability to catch true high-risk patients; the other reduced
      the tool to flagging almost no one at all, which would be "fair" but useless.
    - **This gap is not yet resolved.** It is documented here transparently rather than hidden, and is flagged
      as a required fix before any real deployment.
    """)

# ---------------------------------------------------------------------
# SECTION 6 — Limitations & Transparency
# ---------------------------------------------------------------------
elif section == "6. Limitations & Transparency":
    st.title("Model Limitations")

    st.markdown("""
    <div class='limitation-box'>
    <b>What this model can do:</b> rank discharged diabetic patients by estimated 30-day readmission risk,
    meaningfully better than random selection, using information already in the hospital record.
    </div>
    <div class='limitation-box'>
    <b>What this model cannot do:</b>
    <ul>
    <li>Reliably catch the majority of true readmissions — it currently reaches about 31% of them within its
    shortlist, short of the original 75% goal</li>
    <li>Guarantee equal treatment across race and age groups — a fairness gap was found and remains unresolved</li>
    <li>Replace clinical judgment — every prediction is a decision-support signal only, never an automated action</li>
    <li>Reflect current clinical practice — it was trained on hospital data from 1999–2008</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.title("Transparency Statement")
    st.markdown(f"""
    <div class='transparency-box'>
    This tool is a decision-support illustration built for an academic capstone project. It is <b>not</b> validated
    for real clinical deployment. Every prediction requires human clinical review before any action is taken —
    the model does not, and should not, make decisions on its own. Known limitations, including the fairness gap
    described above and the gap between target and actual performance, are disclosed here rather than omitted,
    consistent with the project's Explainability Over Black-Box Accuracy and Human-in-the-Loop principles.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Business Implications")
    st.markdown("""
    - A working, quantified prioritization signal (56% better than random selection) now exists where none did before —
      replacing ad hoc, inconsistent clinical judgment under time pressure.
    - The fairness gap found here has real implications: if deployed as-is, the tool would direct follow-up
      resources unevenly across race and age groups, which carries both an ethical and a regulatory dimension.
    - The gap between target and actual recall suggests the currently available hospital data (administrative
      and coded fields) may not be sufficient on its own to reach the original goal.
    """)

    st.subheader("Next Steps")
    st.markdown("""
    - Resolve the fairness gap using a different technical approach than the two already tested
    - Explore whether richer data sources could close the recall gap
    - Continue human-in-the-loop review as a permanent design requirement, not a temporary safeguard
    """)
    st.caption("Per this assignment's scope, no formal recommendation is made here — see the Final Project deliverable.")
