# Testing DVC and Airflow Pipeline Guide

This guide shows how to test your complete MLOps pipeline with DVC data versioning and Airflow orchestration.

## 🎯 What This Test Demonstrates

1. **DVC Data Versioning**: Track dataset changes across versions
2. **Model Training Pipeline**: Train with your refactored Logistic Regression code
3. **MLflow Experiment Tracking**: Log metrics, artifacts, and models
4. **Airflow Orchestration**: Automate the entire workflow
5. **Model Registry**: Register and promote models to production

---

## 📋 Prerequisites

Ensure all services are running:
```bash
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850
docker-compose -f docker-compose.full-stack.yml ps
```

You should see:
- ✅ mlflow-server (http://localhost:5000)
- ✅ airflow-webserver (http://localhost:8080)
- ✅ minio (http://localhost:9001)
- ✅ airflow-scheduler, airflow-worker
- ✅ PostgreSQL instances

---

## 🧪 Testing Workflow

### **Step 1: Initialize DVC Remote Storage**

First, configure DVC to use MinIO as remote storage:

```bash
# Configure DVC remote
dvc remote add -d myremote s3://dvc-storage/data
dvc remote modify myremote endpointurl http://localhost:9000
dvc remote modify myremote access_key_id minioadmin
dvc remote modify myremote secret_access_key minioadmin

# Commit the configuration
git add .dvc/config
git commit -m "chore: configure DVC remote storage"
```

### **Step 2: Create Initial Data Version (v1.0 - 100% data)**

Track the full dataset with DVC:

```bash
# Add the data file to DVC tracking
dvc add data/facts_dataset.csv

# Commit to Git (only the .dvc file, not the data)
git add data/facts_dataset.csv.dvc data/.gitignore
git commit -m "data: add facts_dataset v1.0 (100% data)"

# Push data to MinIO
dvc push

# Verify data is in MinIO
echo "✓ Data pushed to MinIO. Check at http://localhost:9001"
```

### **Step 3: Run Initial Training Pipeline**

Train the model with the full dataset:

```bash
# Option A: Run locally (fastest for testing)
python src/training/training_credit_risk_refactor.py

# Option B: Run via DVC pipeline
dvc repro

# Option C: Trigger via Airflow (see Step 6)
```

**Expected Output:**
- Model trained and saved to `scripts/credit_score_model/artifacts/credit_risk_model.joblib`
- Experiment logged to MLflow: http://localhost:5000
- Artifacts: confusion_matrix.png, precision_recall_curve.png, feature_importance.json

### **Step 4: Create New Data Version (v2.0 - 30% sample)**

Simulate receiving new data by creating a 30% sample:

```bash
# Create 30% data sample
python scripts/create_data_version.py \
    --source data/facts_dataset.csv \
    --output data/facts_dataset.csv \
    --fraction 0.3 \
    --seed 42

# Check the new size
wc -l data/facts_dataset.csv
```

### **Step 5: Version and Train with New Data**

```bash
# DVC will detect the change
dvc status

# Add the new version to DVC
dvc add data/facts_dataset.csv

# Commit to Git with version tag
git add data/facts_dataset.csv.dvc
git commit -m "data: update to v2.0 (30% sample for testing)"
git tag -a data-v2.0 -m "Data version 2.0: 30% sample"

# Push new version to MinIO
dvc push

# Run training with new data
dvc repro

# Check MLflow for new experiment run
echo "✓ Check MLflow: http://localhost:5000"
```

### **Step 6: Test Airflow Orchestration**

Now test the full automated pipeline via Airflow:

#### **6.1: Access Airflow UI**
```bash
# Open browser to http://localhost:8080
# Login: airflow / airflow
```

#### **6.2: Prepare Airflow Environment**

The DAG expects code at `/home/airflow/workspace`. We need to mount your code:

**Option A: Add volume mount to docker-compose** (Recommended)

Edit `docker-compose.full-stack.yml` and add to all airflow services:

```yaml
volumes:
  - ./dags:/opt/airflow/dags
  - ./logs:/opt/airflow/logs
  - ./plugins:/opt/airflow/plugins
  - .:/home/airflow/workspace  # Add this line
```

Then restart Airflow:
```bash
docker-compose -f docker-compose.full-stack.yml restart airflow-webserver airflow-scheduler airflow-worker
```

**Option B: Copy files into container** (For testing)

```bash
# Copy project files to Airflow worker
docker cp . mlops-course-tunghoang1850-airflow-worker-1:/home/airflow/workspace/

# Install dependencies in Airflow containers
docker-compose -f docker-compose.full-stack.yml exec airflow-worker \
    pip install pandas scikit-learn joblib pydantic seaborn dvc mlflow boto3
```

#### **6.3: Enable and Trigger DAG**

1. Go to Airflow UI: http://localhost:8080
2. Find DAG: `ml_training_pipeline`
3. Toggle it to **ON** (unpause)
4. Click **Trigger DAG** (play button)
5. Click on the DAG run to see task execution

**Monitor Tasks:**
- ✅ `configure_dvc_credentials` - Setup S3 credentials
- ✅ `dvc_pull_data` - Pull data from MinIO
- ✅ `dvc_repro_pipeline` - Run DVC pipeline
- ✅ `train_model_mlflow` - Train model with MLflow logging
- ✅ `dvc_push_artifacts` - Push artifacts back to MinIO
- ✅ `promote_model_production` - Promote to production in MLflow

#### **6.4: Check Task Logs**

Click any task → **Log** to see detailed execution:

```
[2025-11-27 10:45:00] 🚀 Training model with refactored training script...
[2025-11-27 10:45:05] INFO - Training the model
[2025-11-27 10:45:15] INFO - Model registered under name 'credit_risk_FPD10_plus'
[2025-11-27 10:45:16] ✓ Model training completed
```

### **Step 7: Create Additional Data Versions**

Test multiple versions:

```bash
# Version 3: 50% sample
python scripts/create_data_version.py --fraction 0.5 --seed 100
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc
git commit -m "data: v3.0 (50% sample)"
git tag -a data-v3.0 -m "Data version 3.0: 50% sample"
dvc push

# Version 4: 70% sample
python scripts/create_data_version.py --fraction 0.7 --seed 200
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc
git commit -m "data: v4.0 (70% sample)"
git tag -a data-v4.0 -m "Data version 4.0: 70% sample"
dvc push

# Trigger Airflow DAG for each version to see how metrics change
```

### **Step 8: Switch Between Data Versions**

DVC allows you to switch to any previous version:

```bash
# Switch to v2.0 (30% data)
git checkout data-v2.0
dvc checkout

# Train with this version
dvc repro

# Switch back to latest
git checkout main
dvc checkout

# Compare experiments in MLflow
```

---

## 📊 Verification Checklist

### **DVC Verification**
```bash
# Check DVC status
dvc status

# List all versions
git tag -l "data-v*"

# Check what's in MinIO
dvc remote list
```

### **MLflow Verification**
1. Open http://localhost:5000
2. Check experiment: `credit_score_model_experiment_v2`
3. Compare runs from different data versions
4. View metrics: precision, recall, f1_score, gini_test, optimized_profit
5. Check artifacts: confusion_matrix.png, feature_importance.json
6. Go to **Models** tab → `credit_risk_FPD10_plus` → See versions

### **MinIO Verification**
1. Open http://localhost:9001
2. Login: minioadmin / minioadmin
3. Check buckets:
   - `dvc-storage`: Contains versioned data files
   - `mlflow-artifacts`: Contains model artifacts

### **Airflow Verification**
1. Open http://localhost:8080
2. Check **DAGs** tab: `ml_training_pipeline` should be visible
3. Check **Browse** → **DAG Runs**: See execution history
4. Check **Browse** → **Task Instances**: See individual task status

---

## 🔄 Complete Test Scenario

Here's a full end-to-end test scenario:

```bash
# 1. Start fresh
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850

# 2. Backup original data
cp data/facts_dataset.csv data/facts_dataset_backup.csv

# 3. Initialize DVC (if not done)
dvc init --no-scm
dvc remote add -d myremote s3://dvc-storage/data
dvc remote modify myremote endpointurl http://localhost:9000
dvc remote modify myremote access_key_id minioadmin
dvc remote modify myremote secret_access_key minioadmin

# 4. Version 1: Full dataset (100%)
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc .dvc/config data/.gitignore
git commit -m "data: v1.0 - full dataset"
git tag -a data-v1.0 -m "Full dataset"
dvc push
dvc repro
# Check MLflow: Experiment 1

# 5. Version 2: 30% sample
python scripts/create_data_version.py --fraction 0.3 --seed 42
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc
git commit -m "data: v2.0 - 30% sample"
git tag -a data-v2.0 -m "30% sample for testing"
dvc push
dvc repro
# Check MLflow: Experiment 2 (compare metrics with Exp 1)

# 6. Trigger Airflow DAG
# Go to http://localhost:8080 and trigger ml_training_pipeline
# Watch all tasks execute automatically

# 7. Restore original data
cp data/facts_dataset_backup.csv data/facts_dataset.csv
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc
git commit -m "data: restore full dataset"
dvc push
```

---

## 🎓 What You Learned

✅ **DVC**: Version control for data and models
✅ **MLflow**: Experiment tracking and model registry
✅ **Airflow**: Workflow orchestration and automation
✅ **MinIO**: S3-compatible object storage
✅ **Docker**: Containerized services orchestration
✅ **Production MLOps**: Complete reproducible ML pipeline

---

## 🐛 Troubleshooting

### Issue: DVC push fails
```bash
# Check MinIO connection
curl http://localhost:9000/minio/health/live

# Re-configure remote
dvc remote modify myremote endpointurl http://localhost:9000
```

### Issue: Airflow DAG not visible
```bash
# Check DAG file syntax
docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler \
    python /opt/airflow/dags/ml_training_dag.py

# Check logs
docker-compose -f docker-compose.full-stack.yml logs airflow-scheduler --tail 50
```

### Issue: MLflow connection failed
```bash
# Check MLflow is accessible
curl http://localhost:5000/health

# Check environment variable in training script
grep MLFLOW_TRACKING_URI src/training/training_credit_risk_refactor.py
```

### Issue: Import errors in Airflow
```bash
# Install missing packages
docker-compose -f docker-compose.full-stack.yml exec airflow-worker \
    pip install pandas scikit-learn joblib pydantic seaborn
```

---

## 📚 Next Steps

1. **Automate**: Schedule Airflow DAG to run daily/weekly
2. **Monitor**: Add Prometheus/Grafana for metrics monitoring
3. **Deploy**: Integrate FastAPI with MLflow model registry
4. **Scale**: Add more Airflow workers for parallel training
5. **Test**: Add data validation with Great Expectations
6. **Alert**: Configure Airflow email alerts on failure

---

## 🔗 Quick Links

- **Airflow UI**: http://localhost:8080 (airflow/airflow)
- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **DVC Docs**: https://dvc.org/doc
- **Airflow Docs**: https://airflow.apache.org/docs/
- **MLflow Docs**: https://mlflow.org/docs/latest/
