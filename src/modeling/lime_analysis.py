"""
Explainable AI -- Local (LIME) (Module 4 Sec. 6).
Complements SHAP's local waterfall plots with LIME explanations for the
same two reference patients (highest- and lowest-risk in the test set),
so the report can compare what each method highlights.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_NAME = "xgboost"


def load_artifacts():
    model = joblib.load(MODEL_DIR / f"{BEST_MODEL_NAME}.joblib")
    X_train_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    return model, X_test, X_train_cols


def build_explainer(X_test):
    return LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=["Not readmitted", "Readmitted <30d"],
        mode="classification",
        discretize_continuous=True,
    )


def explain_instance(explainer, model, X_test, row_idx, tag, num_features=10):
    exp = explainer.explain_instance(
        X_test.iloc[row_idx].values,
        model.predict_proba,
        num_features=num_features,
    )
    out_txt = REPORTS_DIR / f"lime_local_{tag}.txt"
    with open(out_txt, "w") as f:
        f.write(f"LIME explanation for {tag} (row {row_idx})\n")
        f.write(f"Predicted probability of readmission: {model.predict_proba(X_test.iloc[[row_idx]])[0][1]:.4f}\n\n")
        for feature, weight in exp.as_list():
            f.write(f"  {feature}: {weight:+.4f}\n")

    fig = exp.as_pyplot_figure()
    out_png = REPORTS_DIR / f"lime_local_{tag}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_txt, out_png, exp


if __name__ == "__main__":
    model, X_test, feature_cols = load_artifacts()
    explainer = build_explainer(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(y_proba))
    low_risk_idx = int(np.argmin(y_proba))

    for idx, tag in [(high_risk_idx, "high_risk"), (low_risk_idx, "low_risk")]:
        txt_path, png_path, exp = explain_instance(explainer, model, X_test, idx, tag)
        print(f"{tag}: risk={y_proba[idx]:.4f}")
        for feature, weight in exp.as_list():
            print(f"  {feature}: {weight:+.4f}")
        print(f"Saved: {txt_path}, {png_path}\n")
