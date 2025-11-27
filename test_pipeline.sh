#!/bin/bash
# Quick Test Script for DVC + Airflow Pipeline
# This script automates the testing workflow

set -e  # Exit on error

echo "🧪 DVC + Airflow Pipeline Testing Script"
echo "========================================="
echo ""

# Load environment variables for local DVC operations
if [ -f .env.local ]; then
    echo "📦 Loading environment variables from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
    echo "✓ Environment configured (endpoint: $AWS_ENDPOINT_URL_S3)"
else
    echo "⚠️  .env.local not found. Using defaults..."
    export AWS_ENDPOINT_URL_S3=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    export AWS_DEFAULT_REGION=us-east-1
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/tungh1850/learning/mls_ops/tnex/"
DATA_FILE="data/facts_dataset.csv"
BACKUP_FILE="data/facts_dataset_backup.csv"

cd "$PROJECT_DIR"

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if services are running
print_step "Checking Docker services..."
if ! docker-compose -f docker-compose.full-stack.yml ps | grep -q "Up"; then
    echo "⚠ Services not running. Starting services..."
    docker-compose -f docker-compose.full-stack.yml up -d
    sleep 20
fi
print_success "Services are running"

# Backup original data if not exists
if [ ! -f "$BACKUP_FILE" ]; then
    print_step "Backing up original dataset..."
    cp "$DATA_FILE" "$BACKUP_FILE"
    print_success "Backup created: $BACKUP_FILE"
fi

# Activate virtual environment if exists
if [ -d ".venv_ml" ]; then
    source .venv_ml/bin/activate
    echo "✓ Virtual environment activated"
fi

# Menu
echo ""
echo "Choose a testing scenario:"
echo "1) Initialize DVC and create version 1.0 (100% data)"
echo "2) Create version 2.0 (30% sample) and train"
echo "3) Create version 3.0 (50% sample) and train"
echo "4) Create version 4.0 (70% sample) and train"
echo "5) Restore original dataset (100%)"
echo "6) Install Python dependencies in Airflow workers"
echo "7) Trigger Airflow DAG"
echo "8) View experiment results in MLflow"
echo "9) Run complete end-to-end test"
echo "0) Exit"
echo ""
read -p "Enter your choice: " choice

case $choice in
    1)
        print_step "Initializing DVC and creating version 1.0..."

        # Configure DVC remote if not configured
        if ! dvc remote list | grep -q "myremote"; then
            print_step "Configuring DVC remote..."
            dvc remote add -d myremote s3://dvc-storage/data
            dvc remote modify myremote endpointurl http://localhost:9000
            dvc remote modify myremote access_key_id minioadmin
            dvc remote modify myremote secret_access_key minioadmin
            git add .dvc/config
            git commit -m "chore: configure DVC remote storage" || true
        fi

        # Restore full dataset
        cp "$BACKUP_FILE" "$DATA_FILE"

        # Track with DVC
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc data/.gitignore
        git commit -m "data: v1.0 - full dataset (100%)" || true
        git tag -a data-v1.0 -m "Full dataset" -f

        # Push to remote
        dvc push

        print_success "Version 1.0 created and pushed to MinIO"
        print_warning "Run option 6 to install dependencies, then option 7 to train via Airflow"
        ;;

    2)
        print_step "Creating version 2.0 (30% sample)..."
        python scripts/create_data_version.py --fraction 0.3 --seed 42
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: v2.0 - 30% sample for testing" || true
        git tag -a data-v2.0 -m "30% sample" -f
        dvc push

        print_success "Version 2.0 created (30% sample)"
        print_step "Training model..."
        dvc repro
        print_success "Training complete! Check MLflow: http://localhost:5000"
        ;;

    3)
        print_step "Creating version 3.0 (50% sample)..."
        python scripts/create_data_version.py --fraction 0.5 --seed 100
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: v3.0 - 50% sample" || true
        git tag -a data-v3.0 -m "50% sample" -f
        dvc push

        print_success "Version 3.0 created (50% sample)"
        print_step "Training model..."
        dvc repro
        print_success "Training complete! Check MLflow: http://localhost:5000"
        ;;

    4)
        print_step "Creating version 4.0 (70% sample)..."
        python scripts/create_data_version.py --fraction 0.7 --seed 200
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: v4.0 - 70% sample" || true
        git tag -a data-v4.0 -m "70% sample" -f
        dvc push

        print_success "Version 4.0 created (70% sample)"
        print_step "Training model..."
        dvc repro
        print_success "Training complete! Check MLflow: http://localhost:5000"
        ;;

    5)
        print_step "Restoring original dataset..."
        cp "$BACKUP_FILE" "$DATA_FILE"
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: restore full dataset" || true
        dvc push
        print_success "Original dataset restored"
        ;;

    6)
        print_step "Installing Python dependencies in Airflow workers..."
        docker-compose -f docker-compose.full-stack.yml exec -T airflow-worker \
            pip install pandas scikit-learn joblib pydantic seaborn matplotlib dvc mlflow boto3

        docker-compose -f docker-compose.full-stack.yml exec -T airflow-scheduler \
            pip install pandas scikit-learn joblib pydantic seaborn matplotlib dvc mlflow boto3

        print_success "Dependencies installed in Airflow containers"
        ;;

    7)
        print_step "Triggering Airflow DAG..."
        print_warning "Please go to http://localhost:8080 (login: airflow/airflow)"
        print_warning "1. Find 'ml_training_pipeline' DAG"
        print_warning "2. Toggle it ON (unpause)"
        print_warning "3. Click the Play button to trigger"
        print_warning ""
        print_warning "Or use Airflow CLI:"
        echo "docker-compose -f docker-compose.full-stack.yml exec airflow-scheduler airflow dags trigger ml_training_pipeline"
        ;;

    8)
        print_step "Opening MLflow UI..."
        print_success "MLflow UI: http://localhost:5000"
        print_success "Check experiment: credit_score_model_experiment_v2"
        print_success "Compare runs from different data versions"
        xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || echo "Please open http://localhost:5000 in your browser"
        ;;

    9)
        print_step "Running complete end-to-end test..."

        # Step 1: Initialize
        print_step "1/5: Initializing DVC..."
        if ! dvc remote list | grep -q "myremote"; then
            dvc remote add -d myremote s3://dvc-storage/data
            dvc remote modify myremote endpointurl http://localhost:9000
            dvc remote modify myremote access_key_id minioadmin
            dvc remote modify myremote secret_access_key minioadmin
            git add .dvc/config
            git commit -m "chore: configure DVC remote" || true
        fi

        # Step 2: Version 1
        print_step "2/5: Creating version 1.0 (100% data)..."
        cp "$BACKUP_FILE" "$DATA_FILE"
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc data/.gitignore
        git commit -m "data: v1.0 - full dataset" || true
        git tag -a data-v1.0 -m "Full dataset" -f
        dvc push
        dvc repro

        # Step 3: Version 2
        print_step "3/5: Creating version 2.0 (30% sample)..."
        python scripts/create_data_version.py --fraction 0.3 --seed 42
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: v2.0 - 30% sample" || true
        git tag -a data-v2.0 -m "30% sample" -f
        dvc push
        dvc repro

        # Step 4: Version 3
        print_step "4/5: Creating version 3.0 (50% sample)..."
        python scripts/create_data_version.py --fraction 0.5 --seed 100
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: v3.0 - 50% sample" || true
        git tag -a data-v3.0 -m "50% sample" -f
        dvc push
        dvc repro

        # Step 5: Restore
        print_step "5/5: Restoring original dataset..."
        cp "$BACKUP_FILE" "$DATA_FILE"
        dvc add "$DATA_FILE"
        git add data/facts_dataset.csv.dvc
        git commit -m "data: restore full dataset" || true
        dvc push

        print_success "Complete end-to-end test finished!"
        print_success "Check results:"
        echo "  - MLflow: http://localhost:5000"
        echo "  - MinIO: http://localhost:9001 (minioadmin/minioadmin)"
        echo "  - Git tags: git tag -l 'data-v*'"
        ;;

    0)
        print_step "Exiting..."
        exit 0
        ;;

    *)
        print_warning "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Done! 🎉"
