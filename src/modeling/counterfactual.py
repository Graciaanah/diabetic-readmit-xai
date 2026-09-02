"""
Explainable AI -- Counterfactual Explanations (Module 4 Sec. 6).
Answers "what would need to change for a different outcome?" for a
high-risk patient, using DiceML against the best model (XGBoost).
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import dice_ml
from dice_ml import Dice

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_NAME = "xgboost"
TARGET = "readmitted_30d"


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()
    return model, X_test, y_test


def build_dice_explainer(model, X_test, y_test):
    df_for_dice = X_test.copy()
    df_for_dice[TARGET] = y_test.values

    # Only allow a clinically/operationally ACTIONABLE subset of features to
    # vary in the counterfactual search -- varying e.g. num_lab_procedures
    # or medication flags reflects care decisions a clinician could act on;
    # varying prior-utilization COUNTS (number_inpatient) or diagnosis
    # categories would not be a meaningful "what could change" story.
    actionable_features = [c for c in X_test.columns if any(
        c.startswith(p) for p in ["num_medications", "num_lab_procedures", "num_procedures",
                                    "time_in_hospital", "insulin_", "metformin_", "change_", "diabetesMed_"]
    )]

    data = dice_ml.Data(
        dataframe=df_for_dice,
        continuous_features=[c for c in ["num_medications", "num_lab_procedures", "num_procedures", "time_in_hospital"] if c in X_test.columns],
        outcome_name=TARGET,
    )

    # DiceML's random-search candidate generator round-trips through pandas
    # in a way that can leave columns as dtype=object; XGBoost 2.x+ enforces
    # strict numeric/category dtypes on predict, so wrap the model to coerce
    # dtypes back to float64 on every predict_proba call.
    class DtypeSafeModel:
        def __init__(self, inner_model, columns):
            self.inner_model = inner_model
            self.columns = columns

        def predict_proba(self, X):
            X = pd.DataFrame(X, columns=self.columns).astype("float64")
            return self.inner_model.predict_proba(X)

        def predict(self, X):
            X = pd.DataFrame(X, columns=self.columns).astype("float64")
            return self.inner_model.predict(X)

    safe_model = DtypeSafeModel(model, X_test.columns.tolist())
    dice_model = dice_ml.Model(model=safe_model, backend="sklearn")
    explainer = Dice(data, dice_model, method="random")
    return explainer, actionable_features


def generate_counterfactuals(explainer, X_test, model, actionable_features, row_idx, total_cfs=3):
    query_instance = X_test.iloc[[row_idx]]
    original_pred = model.predict_proba(query_instance)[0][1]

    cf = explainer.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_class="opposite",
        features_to_vary=actionable_features,
    )
    return cf, original_pred


if __name__ == "__main__":
    model, X_test, y_test = load_artifacts()
    explainer, actionable_features = build_dice_explainer(model, X_test, y_test)

    y_proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(y_proba))

    print(f"Actionable features allowed to vary: {len(actionable_features)}")
    print(f"Query patient (row {high_risk_idx}) original risk: {y_proba[high_risk_idx]:.4f}\n")

    try:
        cf, original_pred = generate_counterfactuals(explainer, X_test, model, actionable_features, high_risk_idx)
        cf_df = cf.cf_examples_list[0].final_cfs_df
        out_path = REPORTS_DIR / "counterfactual_explanations.csv"
        cf_df.to_csv(out_path, index=False)

        # Human-readable diff: only the columns that actually changed
        original_row = X_test.iloc[high_risk_idx]
        diff_lines = []
        for i, (_, cf_row) in enumerate(cf_df.drop(columns=[TARGET]).iterrows()):
            changed = {
                col: (round(float(original_row[col]), 3), round(float(cf_row[col]), 3))
                for col in original_row.index
                if not np.isclose(float(original_row[col]), float(cf_row[col]), atol=1e-6)
            }
            new_risk = model.predict_proba(pd.DataFrame([cf_row.values], columns=X_test.columns))[0][1]
            diff_lines.append(f"Counterfactual {i+1} (new predicted risk: {new_risk:.4f}, was {original_pred:.4f}):")
            for col, (old, new) in changed.items():
                diff_lines.append(f"    {col}: {old} -> {new}")
            diff_lines.append("")

        diff_text = "\n".join(diff_lines)
        print("Counterfactuals (changed features only):\n")
        print(diff_text)

        diff_path = REPORTS_DIR / "counterfactual_explanations_readable.txt"
        diff_path.write_text(
            f"Counterfactual explanations for high-risk patient (row {high_risk_idx})\n"
            f"Original predicted risk: {original_pred:.4f}\n"
            f"Actionable features allowed to vary: {actionable_features}\n\n" + diff_text
        )
        print(f"\nSaved: {out_path}")
        print(f"Saved: {diff_path}")
    except Exception as e:
        print(f"DiceML generation issue: {e}")
        raise
