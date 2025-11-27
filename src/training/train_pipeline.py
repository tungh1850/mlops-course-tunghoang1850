"""
DVC Pipeline Training Script
Integrates with MLflow for experiment tracking
"""

import json
import os

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def train_pipeline():
    """Full training pipeline with experiment tracking"""
    try:
        import mlflow
        import mlflow.sklearn

        mlflow_available = True
    except ImportError:
        mlflow_available = False
        print("⚠ MLflow not available, proceeding without experiment tracking")

    params = load_params()

    # Set MLflow experiment if available
    if mlflow_available:
        mlflow.set_experiment("credit_risk_v2")
        mlflow.start_run()

    # Load data
    print("📥 Loading data...")
    df = pd.read_csv("data/facts_dataset.csv")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Basic preprocessing
    print("🔧 Preprocessing...")
    # Assuming last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model with parameters from params.yaml
    print("🚀 Training model...")
    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["train"]["random_state"],
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("📊 Evaluating...")
    y_pred = model.predict(X_test)

    # Handle case where only binary output (no probabilities)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        roc_auc = None

    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
    }
    if roc_auc is not None:
        metrics["roc_auc"] = float(roc_auc)

    report = classification_report(y_test, y_pred, output_dict=True)

    # Log to MLflow if available
    if mlflow_available:
        mlflow.log_params(
            {
                "n_estimators": params["train"]["n_estimators"],
                "max_depth": params["train"]["max_depth"],
                "test_size": params["train"]["test_size"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "classification_report.json")

    # Save model
    os.makedirs("scripts/credit_score_model/artifacts", exist_ok=True)
    model_path = "scripts/credit_score_model/artifacts/credit_risk_model.joblib"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

    # Save metrics
    metrics_file = "scripts/credit_score_model/artifacts/metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_file}")

    # MLflow model logging
    if mlflow_available:
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    train_pipeline()
