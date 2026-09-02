"""
Model Registry (Module 4 Technical Requirements).
Registers the best-performing model (XGBoost, highest ROC-AUC and
recall@20%) into MLflow's Model Registry, versioned and staged.
"""
import mlflow

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetic_readmission_risk"
REGISTERED_MODEL_NAME = "diabetic_readmission_xgboost"
BEST_RUN_NAME = "xgboost"


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    best_run = next(r for r in runs if r.data.tags.get("mlflow.runName") == BEST_RUN_NAME)
    model_uri = f"runs:/{best_run.info.run_id}/{BEST_RUN_NAME}"

    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered model: {result.name}, version: {result.version}")

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", result.version)
    print(f"Alias 'champion' set on version {result.version}")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME, version=result.version,
        description=(
            f"XGBoost readmission risk model. ROC-AUC={best_run.data.metrics.get('roc_auc'):.4f}, "
            f"recall@20%={best_run.data.metrics.get('recall_at_20pct'):.4f}. "
            f"Fairness caveat: race/age disparate impact flagged, unmitigated -- see Model Card."
        ),
    )
    return result


if __name__ == "__main__":
    register_best_model()
