"""
ML Training Airflow DAG
Orchestrates: Data Pull → DVC Pipeline → Model Training → Artifact Push → Model Registry
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Configuration
DVC_REPO_PATH = "/home/airflow/workspace"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

# Default arguments for all tasks
default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 1, 1),
}

# DAG Definition
with DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="Daily ML training pipeline with DVC + MLflow orchestration",
    schedule_interval="0 2 * * *",  # Daily at 2 AM UTC
    catchup=False,
    tags=["mlops", "credit-risk", "training"],
) as dag:

    def configure_dvc_credentials():
        """Configure DVC to use MinIO S3-compatible storage"""
        print("🔐 Configuring DVC S3 credentials...")
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["AWS_ENDPOINT_URL"] = "http://minio:9000"
        os.environ["DVC_S3_USE_SSL"] = "false"
        print("✓ DVC S3 credentials configured for MinIO")

    def promote_model_to_production():
        """Promote latest model to production stage in MLflow"""
        print("🚀 Promoting model to production stage...")
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()

            # Get latest experiment
            experiment = client.get_experiment_by_name(
                "credit_score_model_experiment_v2"
            )
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if runs:
                    latest_run = runs[0]
                    print(f"✓ Latest run: {latest_run.info.run_id}")
                    print(f"  Metrics: {latest_run.data.metrics}")

                    # Try to transition model to Production
                    try:
                        model_name = "credit_risk_FPD10_plus"
                        versions = client.get_latest_versions(
                            model_name, stages=["None"]
                        )
                        if versions:
                            latest_version = versions[0].version
                            client.transition_model_version_stage(
                                name=model_name,
                                version=latest_version,
                                stage="Production",
                            )
                            print(
                                f"✓ Model version {latest_version} promoted to Production"
                            )
                    except Exception as e:
                        print(f"⚠ Model transition warning: {e}")
        except Exception as e:
            print(f"⚠ Model promotion warning: {e}")

    # Task 1: Configure credentials
    task_configure_creds = PythonOperator(
        task_id="configure_dvc_credentials",
        python_callable=configure_dvc_credentials,
        doc="""Configure DVC S3 credentials for MinIO""",
    )

    # Task 2: Pull latest data from S3/MinIO
    task_dvc_pull = BashOperator(
        task_id="dvc_pull_data",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo "📥 Pulling data from MinIO..." && \
        dvc pull && \
        echo "✓ Data pulled successfully"
        """,
        doc="""Pull data artifacts from DVC remote storage""",
    )

    # Task 3: Run DVC pipeline (preprocessing stages)
    task_dvc_repro = BashOperator(
        task_id="dvc_repro_pipeline",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo "🔄 Running DVC pipeline..." && \
        dvc repro && \
        echo "✓ DVC pipeline completed"
        """,
        doc="""Execute DVC reproduction of pipeline stages""",
    )

    # Task 4: Train model with MLflow logging
    task_train_model = BashOperator(
        task_id="train_model_mlflow",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI} && \
        echo "🚀 Training model with refactored training script..." && \
        python src/training/training_credit_risk_refactor.py && \
        echo "✓ Model training completed"
        """,
        doc="""Train ML model with Logistic Regression and log experiments to MLflow""",
    )

    # Task 5: Push artifacts back to S3/MinIO
    task_dvc_push = BashOperator(
        task_id="dvc_push_artifacts",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo "📤 Pushing artifacts to MinIO..." && \
        dvc push && \
        echo "✓ Artifacts pushed successfully"
        """,
        doc="""Push model and data artifacts to DVC remote storage""",
    )

    # Task 6: Promote model to production registry
    task_promote_model = PythonOperator(
        task_id="promote_model_production",
        python_callable=promote_model_to_production,
        doc="""Promote trained model to production stage in MLflow Registry""",
    )

    # Define task dependencies (linear flow)
    (
        task_configure_creds
        >> task_dvc_pull
        >> task_dvc_repro
        >> task_train_model
        >> task_dvc_push
        >> task_promote_model
    )
