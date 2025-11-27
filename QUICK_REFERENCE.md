# Quick Reference: Testing DVC + Airflow Pipeline

## 🚀 Quick Start (5 minutes)

### 1. Run the Interactive Test Script
```bash
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850
./test_pipeline.sh
```

Choose option 9 for complete end-to-end test, or test individual scenarios.

---

## 📝 Manual Testing Steps

### Test 1: Create 30% Data Version and Train

```bash
# Create new data version
python scripts/create_data_version.py --fraction 0.3 --seed 42

# Track with DVC
dvc add data/facts_dataset.csv
git add data/facts_dataset.csv.dvc
git commit -m "data: 30% sample for testing"

# Push to MinIO
dvc push

# Train model
dvc repro

# Check results in MLflow
open http://localhost:5000
```

### Test 2: Compare Multiple Data Versions

```bash
# Version 1: 30% sample
python scripts/create_data_version.py --fraction 0.3 --seed 42
dvc add data/facts_dataset.csv && git add . && git commit -m "v1: 30%"
git tag data-v1.0
dvc push && dvc repro

# Version 2: 50% sample
python scripts/create_data_version.py --fraction 0.5 --seed 100
dvc add data/facts_dataset.csv && git add . && git commit -m "v2: 50%"
git tag data-v2.0
dvc push && dvc repro

# Version 3: 70% sample
python scripts/create_data_version.py --fraction 0.7 --seed 200
dvc add data/facts_dataset.csv && git add . && git commit -m "v3: 70%"
git tag data-v3.0
dvc push && dvc repro

# Compare in MLflow: http://localhost:5000
# You'll see how metrics change with different data sizes
```

### Test 3: Switch Between Versions

```bash
# Switch to version 1 (30% data)
git checkout data-v1.0
dvc checkout
dvc repro

# Switch to version 2 (50% data)
git checkout data-v2.0
dvc checkout
dvc repro

# Back to latest
git checkout main
dvc checkout
```

### Test 4: Trigger Airflow Pipeline

```bash
# Install dependencies in Airflow workers first
docker-compose -f docker-compose.full-stack.yml exec airflow-worker \
    pip install pandas scikit-learn joblib pydantic seaborn matplotlib dvc mlflow boto3

# Trigger DAG via CLI
docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler \
    airflow dags trigger ml_training_pipeline

# Or via UI: http://localhost:8080 (airflow/airflow)
# 1. Toggle DAG ON
# 2. Click trigger button
# 3. Monitor execution
```

---

## 🔍 Key Commands

### DVC Commands
```bash
dvc status                      # Check for changes
dvc add data/facts_dataset.csv  # Track file with DVC
dvc push                        # Upload to MinIO
dvc pull                        # Download from MinIO
dvc repro                       # Run pipeline
dvc dag                         # Show pipeline graph
```

### Git Commands for Versioning
```bash
git tag -l "data-v*"           # List all data versions
git checkout data-v2.0         # Switch to specific version
git log --oneline              # See commit history
```

### Docker Commands
```bash
# Check services
docker-compose -f docker-compose.full-stack.yml ps

# View logs
docker-compose -f docker-compose.full-stack.yml logs mlflow-server
docker-compose -f docker-compose.full-stack.yml logs airflow-scheduler

# Restart services
docker-compose -f docker-compose.full-stack.yml restart

# Stop all
docker-compose -f docker-compose.full-stack.yml down
```

### Airflow Commands
```bash
# List DAGs
docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler \
    airflow dags list

# Trigger DAG
docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler \
    airflow dags trigger ml_training_pipeline

# Check DAG status
docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler \
    airflow dags list-runs -d ml_training_pipeline
```

---

## 🎯 What to Check After Each Test

### 1. MLflow UI (http://localhost:5000)
- ✅ New experiment run appears
- ✅ Metrics logged: precision, recall, f1_score, gini_test, optimized_profit
- ✅ Artifacts: confusion_matrix.png, precision_recall_curve.png
- ✅ Model registered: credit_risk_FPD10_plus

### 2. MinIO Console (http://localhost:9001)
Login: minioadmin/minioadmin
- ✅ Bucket `dvc-storage`: Contains versioned data files
- ✅ Bucket `mlflow-artifacts`: Contains model artifacts

### 3. Git Repository
```bash
git log --oneline              # Check commits
git tag -l                     # Check tags
```

### 4. Airflow UI (http://localhost:8080)
Login: airflow/airflow
- ✅ DAG `ml_training_pipeline` visible
- ✅ All tasks green (successful)
- ✅ Check task logs for execution details

---

## 🔧 Configuration Files

### Your Training Code
- **Script**: `src/training/training_credit_risk_refactor.py`
- **Experiment**: `credit_score_model_experiment_v2`
- **Model**: `credit_risk_FPD10_plus`
- **MLflow URI**: Set in script as `http://localhost:5050` (you might want to change to 5000)

### DVC Pipeline
- **File**: `dvc.yaml`
- **Stage**: `train`
- **Command**: `python src/training/training_credit_risk_refactor.py`
- **Dependencies**: training script, score_function.py, dataset
- **Outputs**: model.joblib, plots

### Airflow DAG
- **File**: `dags/ml_training_dag.py`
- **DAG ID**: `ml_training_pipeline`
- **Schedule**: Daily at 2 AM UTC
- **Tasks**: 6 tasks (credentials → pull → repro → train → push → promote)

---

## ⚠️ Important Note

Your training script has MLflow tracking URI set to port **5050**:
```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
```

But MLflow is running on port **5000**. To fix:

```bash
# Option 1: Set environment variable
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/training/training_credit_risk_refactor.py

# Option 2: Update the script
# Change line 46 in training_credit_risk_refactor.py from 5050 to 5000
```

---

## 📊 Expected Results

After running with different data sizes, you should see in MLflow:

| Data Size | Train Score | Test Score | Precision | Recall | Gini Test |
|-----------|-------------|------------|-----------|--------|-----------|
| 30%       | ~0.85       | ~0.83      | ~0.75     | ~0.70  | ~0.55     |
| 50%       | ~0.87       | ~0.85      | ~0.78     | ~0.73  | ~0.60     |
| 70%       | ~0.88       | ~0.86      | ~0.80     | ~0.75  | ~0.63     |
| 100%      | ~0.89       | ~0.87      | ~0.82     | ~0.77  | ~0.65     |

(Numbers are approximate and depend on your data)

---

## 🐛 Troubleshooting

### Issue: "No such file or directory: data/facts_dataset.csv"
```bash
# Check if file exists
ls -lh data/facts_dataset.csv

# If not, pull from DVC
dvc pull
```

### Issue: "Module not found: score_function"
```bash
# The script imports from score_function.py
# Make sure you're running from project root
cd /home/hoangtung/mlops/mlops_project/mlops-course-tunghoang1850
python src/training/training_credit_risk_refactor.py
```

### Issue: Airflow DAG not executing
```bash
# Check if workspace is mounted
docker-compose -f docker-compose.full-stack.yml exec airflow-worker \
    ls -la /home/airflow/workspace

# Install dependencies
docker-compose -f docker-compose.full-stack.yml exec airflow-worker \
    pip install pandas scikit-learn joblib pydantic seaborn matplotlib
```

### Issue: MLflow connection failed
Check the tracking URI in your training script matches the running service (5000 not 5050).

---

## 📚 Resources

- **Full Guide**: `TESTING_DVC_AIRFLOW_PIPELINE.md`
- **Test Script**: `test_pipeline.sh`
- **DVC Docs**: https://dvc.org/doc
- **Airflow Docs**: https://airflow.apache.org/docs/
- **MLflow Docs**: https://mlflow.org/docs/latest/
