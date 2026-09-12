"""
Model Development (Module 4 Sec. 4): trains 3 algorithms with MLflow
experiment tracking -- Logistic Regression, Random Forest, and XGBoost --
using grid search with cross-validation on the training set, then final
evaluation on the untouched, patient-grouped held-out test set.

Run with: python -m src.modeling.train
"""
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
import joblib

from .features import load_modeling_frame, build_feature_matrix
from .split import grouped_train_test_split, verify_no_patient_overlap, run_baseline, RANDOM_STATE

ROOT = Path(__file__).resolve().parents[2]
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("diabetic_readmission_risk")


def recall_at_k(y_true, y_proba, k_pct=0.20) -> float:
    """
    Recall@k%: of all actual positives, what fraction are captured within
    the top k% of patients ranked by predicted risk? This is the project's
    actual success metric (Module 1 Vision Document, Goal 1: recall@20%
    >= 75%), distinct from recall at a fixed 0.5 probability threshold --
    which matters here because the deployed tool is a capacity-constrained
    RANKED SHORTLIST, not a binary yes/no classifier.
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(-y_proba)
    top_k_idx = order[:k]
    y_true_arr = np.asarray(y_true)
    total_positives = y_true_arr.sum()
    if total_positives == 0:
        return 0.0
    captured = y_true_arr[top_k_idx].sum()
    return float(captured / total_positives)


def lift_at_k(y_true, y_proba, k_pct=0.20) -> float:
    """Lift over random selection at the same k% (Module 1 Goal 2):
    precision within the top k% divided by the overall base rate."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(-y_proba)
    top_k_idx = order[:k]
    y_true_arr = np.asarray(y_true)
    base_rate = y_true_arr.mean()
    if base_rate == 0:
        return 0.0
    precision_at_k = y_true_arr[top_k_idx].mean()
    return float(precision_at_k / base_rate)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "recall_at_20pct": recall_at_k(y_true, y_proba, 0.20),
        "lift_at_20pct": lift_at_k(y_true, y_proba, 0.20),
    }


def train_logistic_regression(X_train, y_train, cv):
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }
    base = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_random_forest(X_train, y_train, cv):
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [6, 10, None],
        "class_weight": [None, "balanced"],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_xgboost(X_train, y_train, cv):
    # scale_pos_weight ~ handling class imbalance (positive rate ~20%)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "scale_pos_weight": [1, neg / pos],
    }
    base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_all():
    df = load_modeling_frame()
    X, y, protected, split_key, scaler = build_feature_matrix(df)
    X_train, X_test, y_train, y_test, train_idx, test_idx = grouped_train_test_split(X, y, split_key)
    assert verify_no_patient_overlap(split_key, train_idx, test_idx), "Patient leakage between train/test!"

    protected_test = protected.iloc[test_idx].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # --- Baseline (logged too, for comparison in MLflow UI) ---
    with mlflow.start_run(run_name="baseline_majority_class"):
        baseline_metrics = run_baseline(y_train, y_test)
        mlflow.log_metrics({k: v for k, v in baseline_metrics.items() if k != "model"})
        mlflow.log_param("strategy", "most_frequent")
        results["baseline"] = baseline_metrics

    trainers = {
        "logistic_regression": train_logistic_regression,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
    }

    fitted_models = {}
    for name, trainer_fn in trainers.items():
        with mlflow.start_run(run_name=name):
            model, best_params = trainer_fn(X_train, y_train, cv)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            if name == "xgboost":
                mlflow.xgboost.log_model(model, name=name)
            else:
                mlflow.sklearn.log_model(model, name=name)

            cm = confusion_matrix(y_test, y_pred)
            cm_path = MODEL_DIR / f"{name}_confusion_matrix.txt"
            cm_path.write_text(f"Confusion matrix for {name}:\n{cm}\n")
            mlflow.log_artifact(str(cm_path))

            results[name] = {"model": name, **metrics, "best_params": best_params}
            fitted_models[name] = model
            joblib.dump(model, MODEL_DIR / f"{name}.joblib")
            print(f"{name}: {metrics}")

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.joblib")
    X_test.to_csv(MODEL_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODEL_DIR / "y_test.csv", index=False)
    protected_test.to_csv(MODEL_DIR / "protected_test.csv", index=False)

    return results, fitted_models, X_test, y_test, protected_test


if __name__ == "__main__":
    results, fitted_models, X_test, y_test, protected_test = run_all()
    print("\n=== Summary ===")
    for name, m in results.items():
        print(name, {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in m.items() if k not in ("model", "best_params")})
