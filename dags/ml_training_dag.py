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

# Environment variables for all tasks (inherited from docker-compose)
TASK_ENV = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    "AWS_ENDPOINT_URL_S3": os.getenv("AWS_ENDPOINT_URL_S3", "http://minio:9000"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_S3_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL_S3", "http://minio:9000"),
}

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
        os.environ["AWS_ENDPOINT_URL_S3"] = "http://minio:9000"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
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
    # Task 2.1: Clean stale DVC lock
    clean_dvc_lock = BashOperator(
        task_id="clean_dvc_lock",
        bash_command="""
            set -e
            echo "Cleaning stale DVC lock files..."
            rm -f /home/airflow/airflow/dags/your-project/.dvc/tmp/lock
            rm -rf /home/airflow/airflow/dags/your-project/.dvc/tmp/run_cache
            echo "Lock files cleaned"
        """,
        dag=dag,
    )
    # Task 2: Pull latest data from S3/MinIO
    task_dvc_pull = BashOperator(
        task_id="dvc_pull_data",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo " Pulling data from MinIO..." && \
        /home/airflow/.local/bin/dvc pull || echo " files missing!!" && \
        echo "Pull attempted!"
        """,
        env=TASK_ENV,
        doc="""Pull data artifacts from DVC remote storage""",
    )

    # Task 3: Run DVC pipeline (includes training with MLflow)
    task_dvc_repro = BashOperator(
        task_id="dvc_repro_pipeline",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo "Running DVC pipeline with MLflow tracking" && \
        /home/airflow/.local/bin/dvc repro --force && echo "DVC pipeline completed!"
        """,
        env=TASK_ENV,
        doc="""Execute DVC reproduction of pipeline stages (includes model training)""",
    )

    def check_mlflow_experiments():
        """Check what's in MLflow"""
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiments = client.search_experiments()
        print("=" * 60)
        print("MLFLOW EXPERIMENTS")
        print("=" * 60)
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
            print(f"    Runs: {len(runs)}")
            for run in runs:
                print(f"      • {run.info.run_id[:8]}... Status: {run.info.status}")
        print("=" * 60)

    task_check_mlflow = PythonOperator(
        task_id="check_mlflow_experiments",
        python_callable=check_mlflow_experiments,
        doc="""Check MLflow experiments and runs""",
    )

    # Task 4: Push artifacts back to S3/MinIO
    task_dvc_push = BashOperator(
        task_id="dvc_push_artifacts",
        bash_command=f"""
        cd {DVC_REPO_PATH} && \
        echo "📤Pushing artifacts to MinIO" && \
        /home/airflow/.local/bin/dvc push && \
        echo "✓ Artifacts pushed successfully"
        """,
        env=TASK_ENV,
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
        >> clean_dvc_lock
        >> task_dvc_pull
        >> task_dvc_repro
        >> task_check_mlflow
        >> task_dvc_push
        >> task_promote_model
    )
