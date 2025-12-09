# TNEX Credit Risk ML Pipeline Architecture

## Overall System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        RawData[Raw Data CSV]
        MinIO[(MinIO S3 Storage)]
    end

    subgraph "ML Pipeline"
        DataPrep[Data Preparation]
        FeatureEng[Feature Engineering]
        ModelTrain[Model Training]
        ModelEval[Model Evaluation]
    end

    subgraph "Model Registry"
        MLflow[(MLflow Server)]
        PostgresML[(PostgreSQL - MLflow)]
    end

    subgraph "Deployment"
        ModelReg[Model Registration]
        ModelAlias[Set Alias: production/staging]
        API[FastAPI Prediction Service]
    end

    subgraph "Infrastructure"
        Docker[Docker Compose]
        Network[Docker Network: mlops]
    end

    RawData -->|Read| DataPrep
    DataPrep -->|Transform| FeatureEng
    FeatureEng -->|Train| ModelTrain
    ModelTrain -->|Validate| ModelEval
    ModelEval -->|Log Metrics & Artifacts| MLflow
    MLflow -->|Store Artifacts| MinIO
    MLflow -->|Store Metadata| PostgresML
    ModelEval -->|Register Model| ModelReg
    ModelReg -->|Set Alias| ModelAlias
    ModelAlias -->|Load Model| API
    API -->|Fetch Artifacts| MinIO
    API -->|Query Model Info| MLflow
    Docker -->|Orchestrate| Network
    Network -->|Connect| MLflow
    Network -->|Connect| API
    Network -->|Connect| MinIO

    style MLflow fill:#ff6b6b
    style API fill:#4ecdc4
    style MinIO fill:#ffe66d
    style ModelTrain fill:#95e1d3
```

## Detailed Training Pipeline Flow

```mermaid
sequenceDiagram
    participant Data as Raw Data
    participant Prep as Preprocessing
    participant FE as Feature Engineering
    participant Train as Model Training
    participant Eval as Evaluation
    participant MLflow as MLflow Server
    participant MinIO as MinIO Storage

    Data->>Prep: Load CSV data
    Prep->>Prep: Handle missing values
    Prep->>Prep: Remove duplicates
    Prep->>FE: Clean data

    FE->>FE: Encode categorical features
    FE->>FE: Scale numerical features
    FE->>FE: Create feature pipeline
    FE->>Train: Processed features

    Train->>Train: Split train/test
    Train->>Train: Fit model (XGBoost/LightGBM)
    Train->>Train: Hyperparameter tuning
    Train->>Eval: Trained model

    Eval->>Eval: Calculate metrics (AUC, Precision, Recall)
    Eval->>Eval: Generate plots
    Eval->>MLflow: Log metrics
    Eval->>MLflow: Log parameters
    Eval->>MinIO: Store model artifacts
    Eval->>MLflow: Register model version
    MLflow->>MLflow: Set alias (production/staging)
```

## Model Serving Architecture

```mermaid
graph LR
    subgraph "Client Layer"
        Client[Client Application]
        Swagger[Swagger UI]
    end

    subgraph "API Layer"
        FastAPI[FastAPI Server :8000]
        Health[Health Check Endpoint]
        Predict[Prediction Endpoint]
    end

    subgraph "Model Layer"
        ModelLoad[Model Loader]
        Cache[Model Cache]
        Pipeline[sklearn Pipeline]
    end

    subgraph "Storage & Registry"
        MLflow[(MLflow Server :5000)]
        MinIO[(MinIO :9000)]
        PostgresML[(PostgreSQL :5434)]
    end

    Client -->|POST /credit_risk/predict| FastAPI
    Swagger -->|Test API| FastAPI
    FastAPI -->|Route| Predict
    FastAPI -->|Route| Health

    Predict -->|Get Model| ModelLoad
    ModelLoad -->|Check Cache| Cache
    Cache -->|Miss| MLflow
    MLflow -->|Fetch Metadata| PostgresML
    MLflow -->|Download Artifacts| MinIO
    MinIO -->|Return Model| Cache
    Cache -->|Load Pipeline| Pipeline
    Pipeline -->|Predict| Predict
    Predict -->|Response| Client

    Health -->|Check Connection| MLflow
    Health -->|Check Connection| MinIO

    style FastAPI fill:#4ecdc4
    style MLflow fill:#ff6b6b
    style MinIO fill:#ffe66d
    style Cache fill:#95e1d3
```

## Docker Network Architecture

```mermaid
graph TB
    subgraph "Docker Network: mlops"
        subgraph "Storage Services"
            MinIO[MinIO Container<br/>Port: 9000]
            MinIOConsole[MinIO Console<br/>Port: 9001]
            PostgresML[MLflow PostgreSQL<br/>Port: 5434]
        end

        subgraph "ML Services"
            MLflowServer[MLflow Server<br/>Port: 5000]
            OurApp[Prediction API<br/>Port: 8000]
        end

        MLflowServer -->|Read/Write| PostgresML
        MLflowServer -->|Store Artifacts| MinIO
        OurApp -->|Query Models| MLflowServer
        OurApp -->|Download Artifacts| MinIO
    end

    subgraph "Host Machine"
        Browser[Browser]
        APIClient[API Client]
    end

    Browser -->|localhost:5000| MLflowServer
    Browser -->|localhost:9001| MinIOConsole
    Browser -->|localhost:8000| OurApp
    APIClient -->|localhost:8000| OurApp

    style MinIO fill:#ffe66d
    style MLflowServer fill:#ff6b6b
    style OurApp fill:#4ecdc4
    style PostgresML fill:#a8e6cf
```

## Model Lifecycle Management

```mermaid
stateDiagram-v2
    [*] --> Training: Start Training
    Training --> Logged: Log to MLflow
    Logged --> Registered: Register Model
    Registered --> Staging: Set Alias: staging
    Staging --> Validation: Validate Performance
    Validation --> Production: Promote if metrics good
    Validation --> Archived: Archive if metrics bad
    Production --> InProduction: Serving Predictions
    InProduction --> Staging: New Model Available
    Archived --> [*]
    Production --> Archived: Deprecated

    note right of Training
        - Data preprocessing
        - Feature engineering
        - Model training
        - Hyperparameter tuning
    end note

    note right of Validation
        - A/B testing
        - Canary deployment
        - Metric comparison
    end note

    note right of InProduction
        - API serves this version
        - Monitor performance
        - Log predictions
    end note
```

## Key Components

### 1. **MLflow Server**
- Tracks experiments, metrics, and parameters
- Manages model registry and versions
- Uses PostgreSQL for metadata
- Uses MinIO for artifact storage

### 2. **MinIO (S3-compatible storage)**
- Stores model artifacts (`.joblib` files)
- Stores plots and reports
- Bucket: `mlflow-artifacts`

### 3. **FastAPI Prediction Service**
- Loads models using MLflow aliases
- Validates input schemas
- Returns risk predictions
- Health checks for dependencies

### 4. **Model Pipeline**
- Preprocessing: Handle missing values, duplicates
- Feature Engineering: Encoding, scaling
- Training: XGBoost/LightGBM with hyperparameter tuning
- Evaluation: AUC, Precision, Recall, F1

### 5. **Docker Network**
- All services on `mlops` network
- Service discovery via container names
- Environment variables for configuration

## Configuration Flow

```mermaid
graph LR
    ENV[.env File] -->|Load| Docker[Docker Compose]
    Docker -->|Inject| MinIO[MinIO Service]
    Docker -->|Inject| MLflow[MLflow Service]
    Docker -->|Inject| API[API Service]
    Docker -->|Inject| Postgres[PostgreSQL]

    ENV -->|AWS_ACCESS_KEY_ID| MinIO
    ENV -->|AWS_SECRET_ACCESS_KEY| MinIO
    ENV -->|MLFLOW_BACKEND_STORE_URI| MLflow
    ENV -->|MLFLOW_S3_ENDPOINT_URL| MLflow
    ENV -->|OUR_MLFLOW_HOST| API

    style ENV fill:#ffeaa7
```

## Deployment Workflow

1. **Training**: Run pipeline → Log to MLflow → Register model
2. **Validation**: Test model with staging alias
3. **Promotion**: Update alias to `production`
4. **Serving**: API loads model by alias → Serves predictions
5. **Monitoring**: Track metrics, logs, errors
6. **Rollback**: Change alias back if issues detected

---

**Legend:**
- 🔴 Red: MLflow components
- 🔵 Blue: API/Service components
- 🟡 Yellow: Storage components
- 🟢 Green: Data/Pipeline components
