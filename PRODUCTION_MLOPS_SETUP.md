# Production ML Pipeline Setup - Complete Action Plan

## Overview
This guide transforms your current credit risk prediction API into a production-grade MLOps pipeline using:
- **DVC**: Data versioning and reproducible pipelines
- **Airflow**: Workflow orchestration and scheduling
- **MLflow**: Experiment tracking and model registry
- **MinIO**: S3-compatible storage (local dev) → AWS S3 (prod)
- **Prometheus/Grafana**: Model monitoring and drift detection

---

## PHASE 1: DVC Data Versioning (Days 1-2)

### Step 1.1: Install DVC with S3 Support

```bash
# In your venv
pip install dvc[s3]
```

### Step 1.2: Initialize DVC

```bash
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850

# Initialize DVC repo
dvc init

# This creates:
# .dvc/
# .dvc/config          (DVC configuration)
# .dvcignore           (Like .gitignore for DVC)
# .gitignore           (Updated to ignore .dvc cache)

# Commit to Git
git add .dvc .dvcignore
git commit -m "feat: initialize DVC for data versioning"
```

### Step 1.3: Configure DVC Remote Storage

**Development Setup (MinIO locally):**

```bash
# Create a MinIO bucket named 'dvc-storage' via console (http://localhost:9001)
# Then configure DVC:

dvc remote add -d local_minio s3://dvc-storage

# Configure endpoint (local MinIO)
dvc remote modify local_minio endpointurl http://localhost:9000

# Set credentials (stored locally, NOT in Git)
dvc config --local s3.access_key_id minioadmin
dvc config --local s3.secret_access_key minioadmin
dvc config --local s3.ssl_verify false
```

**Production Setup (AWS S3):**

```bash
# Switch to S3 when deploying
dvc remote remove local_minio
dvc remote add -d production_s3 s3://your-mlops-dvc-bucket

# AWS credentials from environment or AWS CLI config
dvc remote modify production_s3 region us-east-1
```

### Step 1.4: Version Your Data and Model

```bash
# Add your dataset
dvc add data/facts_dataset.csv

# This creates: data/facts_dataset.csv.dvc
# Git ignores: data/facts_dataset.csv

# Add trained model artifact
dvc add scripts/credit_score_model/artifacts/credit_risk_model.joblib

# Commit pointers to Git
git add data/facts_dataset.csv.dvc scripts/credit_score_model/artifacts/credit_risk_model.joblib.dvc .gitignore
git commit -m "track: dataset and model artifacts with DVC"

# Push to MinIO (requires running docker-compose for MinIO)
dvc push
```

---

## PHASE 2: DVC Pipeline Definition (Days 3-4)

### Step 2.1: Create Training Script

Create `src/training/train_pipeline.py`:

```python
# src/training/train_pipeline.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
import yaml

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_pipeline():
    """Full training pipeline with MLflow logging"""
    params = load_params()

    # MLflow setup
    mlflow.set_experiment("credit_risk_v2")
    mlflow.start_run()

    # Load data
    print("Loading data...")
    df = pd.read_csv("data/facts_dataset.csv")

    # Preprocess
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with parameters from params.yaml
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=params['train']['n_estimators'],
        max_depth=params['train']['max_depth'],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'accuracy': float((y_pred == y_test).mean())
    }

    report = classification_report(y_test, y_pred, output_dict=True)

    # Log to MLflow
    mlflow.log_params({
        'n_estimators': params['train']['n_estimators'],
        'max_depth': params['train']['max_depth']
    })
    mlflow.log_metrics(metrics)
    mlflow.log_dict(report, "classification_report.json")

    # Save model
    os.makedirs("scripts/credit_score_model/artifacts", exist_ok=True)
    model_path = "scripts/credit_score_model/artifacts/credit_risk_model.joblib"
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, "model")

    # Save metrics
    metrics_file = "scripts/credit_score_model/artifacts/metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

    mlflow.end_run()
    print(f"Model trained and logged. ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    train_pipeline()
```

### Step 2.2: Create params.yaml

Create `params.yaml` in root:

```yaml
# params.yaml
train:
  n_estimators: 100
  max_depth: 10
  test_size: 0.2
  random_state: 42

preprocess:
  missing_value_strategy: drop
  scaling: standard

model:
  type: random_forest
  version: v2
```

### Step 2.3: Define DVC Pipeline (dvc.yaml)

Create `dvc.yaml` in root:

```yaml
# dvc.yaml
stages:
  prepare_data:
    cmd: python src/training/prepare_data.py
    deps:
      - src/training/prepare_data.py
      - data/facts_dataset.csv
    params:
      - preprocess.missing_value_strategy
      - preprocess.scaling
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/training/train_pipeline.py
    deps:
      - src/training/train_pipeline.py
      - data/processed/train.csv
      - params.yaml
    params:
      - train.n_estimators
      - train.max_depth
    metrics:
      - scripts/credit_score_model/artifacts/metrics.json:
          cache: false
    outs:
      - scripts/credit_score_model/artifacts/credit_risk_model.joblib

plots:
  - scripts/credit_score_model/artifacts/classification_report.json:
      template: linear
      x: recall
      y: precision
```

### Step 2.4: Run DVC Pipeline

```bash
# Install required packages first
pip install pyyaml

# Run pipeline
dvc repro

# This will:
# 1. Check dependency hashes
# 2. Run any changed stages in order
# 3. Update dvc.lock with outputs

# Commit results
git add dvc.yaml dvc.lock params.yaml
git commit -m "feat: define DVC training pipeline with parameters"

# Push data artifacts
dvc push
```

---

## PHASE 3: Airflow Orchestration (Days 5-7)

### Step 3.1: Create Full Stack Docker Compose

Create `docker-compose.full-stack.yml`:

```yaml
version: '3.7'

services:
  # ============ MinIO (S3-Compatible Storage) ============
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - mlops

  # ============ Airflow Infrastructure ============
  airflow-postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow_db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    networks:
      - mlops

  airflow-redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5
    networks:
      - mlops

  airflow-init:
    image: apache/airflow:2.7.1
    command: bash -c "airflow db init && airflow users create --username airflow --password airflow --firstname Admin --lastname User --role Admin --email admin@example.com"
    depends_on:
      - airflow-postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://airflow-redis:6379/1
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    networks:
      - mlops

  airflow-webserver:
    image: apache/airflow:2.7.1
    command: webserver
    ports:
      - "8080:8080"
    depends_on:
      - airflow-postgres
      - airflow-redis
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - .:/home/airflow/workspace
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://airflow-redis:6379/1
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - mlops

  airflow-scheduler:
    image: apache/airflow:2.7.1
    command: scheduler
    depends_on:
      - airflow-postgres
      - airflow-redis
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - .:/home/airflow/workspace
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://airflow-redis:6379/1
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    networks:
      - mlops

  airflow-worker:
    image: apache/airflow:2.7.1
    command: celery worker
    depends_on:
      - airflow-postgres
      - airflow-redis
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - .:/home/airflow/workspace
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@airflow-postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://airflow-redis:6379/1
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    networks:
      - mlops

  # ============ MLflow Server ============
  mlflow-postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - mlflow_db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 5s
      retries: 5
    networks:
      - mlops

  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5000:5000"
    depends_on:
      - mlflow-postgres
      - minio
    command: mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlflow:mlflow@mlflow-postgres:5432/mlflow
      --artifacts-destination s3://mlflow-artifacts
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - mlops

volumes:
  minio_data:
  airflow_db:
  mlflow_db:

networks:
  mlops:
    driver: bridge
```

### Step 3.2: Create Airflow DAG Directory

```bash
# Create directories
mkdir -p dags logs plugins

# Create __init__.py
touch dags/__init__.py logs/.gitkeep plugins/__init__.py
```

### Step 3.3: Create ML Training DAG

Create `dags/ml_training_dag.py`:

```python
"""
DAG for ML Pipeline: Data Preparation -> Training -> Model Registry
Runs daily at 2 AM
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import mlflow

# Configuration
DVC_REPO_PATH = "/home/airflow/workspace"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

# Default arguments
default_args = {
    'owner': 'ml-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
}

# DAG definition
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Daily ML training pipeline with DVC + MLflow',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['mlops', 'credit-risk'],
)

def configure_dvc_s3():
    """Configure DVC to use MinIO (S3-compatible)"""
    s3_hook = S3Hook(aws_conn_id='minio_connection')
    creds = s3_hook.get_credentials()

    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["DVC_S3_USE_SSL"] = "false"
    print("✓ DVC S3 credentials configured")

def promote_model_to_production():
    """
    Move best model to production registry in MLflow
    In real scenario: compare metrics, decide if promote
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get latest run from credit_risk_v2 experiment
    experiment = client.get_experiment_by_name("credit_risk_v2")
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if runs:
            latest_run = runs[0]
            # In production: check metrics, decide
            # For now: register model
            print(f"✓ Latest model run: {latest_run.run_id}")

# Task 1: Configure DVC
configure_task = PythonOperator(
    task_id='configure_dvc_s3',
    python_callable=configure_dvc_s3,
    dag=dag,
)

# Task 2: Pull latest data
pull_data_task = BashOperator(
    task_id='dvc_pull_data',
    bash_command=f'cd {DVC_REPO_PATH} && dvc pull',
    dag=dag,
)

# Task 3: Run DVC pipeline
run_pipeline_task = BashOperator(
    task_id='dvc_repro_pipeline',
    bash_command=f'cd {DVC_REPO_PATH} && dvc repro',
    dag=dag,
)

# Task 4: Train model with MLflow
train_model_task = BashOperator(
    task_id='train_model',
    bash_command=f'''
    cd {DVC_REPO_PATH} && \
    export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI} && \
    python src/training/train_pipeline.py
    ''',
    dag=dag,
)

# Task 5: Push new artifacts
push_artifacts_task = BashOperator(
    task_id='dvc_push_artifacts',
    bash_command=f'cd {DVC_REPO_PATH} && dvc push',
    dag=dag,
)

# Task 6: Promote model
promote_task = PythonOperator(
    task_id='promote_model_to_production',
    python_callable=promote_model_to_production,
    dag=dag,
)

# Define task dependencies
configure_task >> pull_data_task >> run_pipeline_task >> train_model_task >> push_artifacts_task >> promote_task
```

### Step 3.4: Start Airflow Stack

```bash
# Copy .env configuration
echo "AIRFLOW_UID=$(id -u)" > .env

# Initialize Airflow DB
docker-compose -f docker-compose.full-stack.yml up airflow-init

# Start all services
docker-compose -f docker-compose.full-stack.yml up -d

# Check status
docker-compose -f docker-compose.full-stack.yml ps
```

### Step 3.5: Configure MinIO Buckets

```bash
# Access MinIO console: http://localhost:9001
# Login: minioadmin / minioadmin
# Create two buckets:
# 1. dvc-storage       (for DVC data)
# 2. mlflow-artifacts  (for MLflow models)
```

### Step 3.6: Configure Airflow Connection

In Airflow UI (http://localhost:8080):
1. Go to **Admin** → **Connections**
2. Click **+ Add New**
3. Fill in:
   - **Conn Id**: `minio_connection`
   - **Conn Type**: `S3`
   - **Login**: `minioadmin`
   - **Password**: `minioadmin`
   - **Extra**: `{"host": "http://minio:9000"}`
4. Save

---

## PHASE 4: Integrate FastAPI with Model Registry (Days 8-9)

### Step 4.1: Update FastAPI to Load from MLflow

Modify `src/predictapi.py`:

```python
import os
import mlflow
from fastapi import FastAPI
from src.router.predict import router

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"  # or "Staging"

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from MLflow registry
def load_production_model():
    """Load latest production model from MLflow"""
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        print(f"✓ Loaded {MODEL_NAME} ({MODEL_STAGE}) from MLflow")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

# Create FastAPI app
app = FastAPI(title="Credit Risk Prediction API")
app.include_router(router)

# Load model on startup
@app.on_event("startup")
async def startup():
    app.state.model = load_production_model()

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running."}

@app.get("/health")
def health():
    """Health check endpoint"""
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Step 4.2: Update Prediction Router

Modify `src/router/predict.py`:

```python
from fastapi import APIRouter, FastAPI, HTTPException
from src.schemas.request import PredictionRequest
from src.schemas.response import PredictionResponse
import mlflow

router = APIRouter(prefix="/api", tags=["predictions"])

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using loaded model"""
    try:
        from fastapi import Request
        from starlette.requests import Request as StarletteRequest

        # Get app from request context (passed via middleware)
        # For now, we'll access it via global (best practice: use dependency injection)

        if not hasattr(app_instance, 'state') or app_instance.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model = app_instance.state.model

        # Prepare input
        input_data = request.dict()
        prediction = model.predict([input_data])[0]
        probability = float(model.predict_proba([input_data])[0][1])

        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            model_version=mlflow.tracking.MlflowClient().get_latest_versions("credit-risk-model", stages=["Production"])[0].version
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## PHASE 5: Monitoring & Model Drift Detection (Days 10-11)

### Step 5.1: Add Prometheus Metrics

Create `src/monitoring/metrics.py`:

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'prediction_class']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=(0.1, 0.5, 1.0, 2.0)
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

prediction_distribution = Histogram(
    'prediction_probability',
    'Distribution of prediction probabilities',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

def record_prediction(model_version, prediction, probability):
    """Record metrics for a prediction"""
    predictions_total.labels(
        model_version=model_version,
        prediction_class=int(prediction)
    ).inc()
    prediction_distribution.observe(probability)
```

### Step 5.2: Integrate with FastAPI

Update `src/predictapi.py`:

```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring.metrics import record_prediction

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# Wrap prediction endpoint with metrics
@app.post("/api/predict")
async def predict_with_metrics(request):
    # ... prediction code ...
    record_prediction(model_version, prediction, probability)
    return response
```

### Step 5.3: Docker Compose with Prometheus + Grafana

Add to `docker-compose.full-stack.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - mlops

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    networks:
      - mlops

volumes:
  prometheus_data:
  grafana_data:
```

---

## QUICK START (All-in-One)

```bash
# 1. Navigate to project
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850

# 2. Install dependencies
pip install dvc[s3] apache-airflow mlflow prometheus-client pyyaml

# 3. Initialize DVC
dvc init
dvc remote add -d local_minio s3://dvc-storage
dvc config --local s3.access_key_id minioadmin
dvc config --local s3.secret_access_key minioadmin
dvc config --local s3.ssl_verify false
git add .dvc .dvcignore && git commit -m "init: DVC"

# 4. Create dvc.yaml and params.yaml (copy from above)
# 5. Create dags/ directory and DAG file (copy from above)

# 6. Start Airflow + MinIO + MLflow
docker-compose -f docker-compose.full-stack.yml up -d airflow-init
docker-compose -f docker-compose.full-stack.yml up -d

# 7. Create MinIO buckets (via console: http://localhost:9001)

# 8. Configure Airflow Connection (http://localhost:8080)

# 9. Trigger DAG manually to test

# 10. Check results:
#     - Airflow: http://localhost:8080
#     - MLflow: http://localhost:5000
#     - MinIO: http://localhost:9001
```

---

## RECOMMENDATION SUMMARY

| Aspect | Recommendation | Reason |
|--------|---|---|
| **Architecture** | **OPTION 1: Integrated Single Repo** | Small team, easier to manage, single CI/CD |
| **Storage** | **MinIO (dev) → AWS S3 (prod)** | Easy migration, cost-effective |
| **Orchestrator** | **Airflow** | Industry standard, robust scheduling |
| **Experiment Tracking** | **MLflow** | Lightweight, integrates with sklearn |
| **Monitoring** | **Prometheus + Grafana** | Native Kubernetes support for future |
| **Timeline** | **2 weeks** | Phased approach, 1 week per 2-3 phases |
