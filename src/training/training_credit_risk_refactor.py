import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, Field, field_validator
from score_function import extract_cic_features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sklearn.set_config(transform_output="pandas")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("credit_score_model_training")


# MODEL CONFIG
class ModelConfig(BaseModel):
    experiment_name: str = "credit_score_model_experiment_v4"
    run_name: str = "credit_risk_model_v4"

    # model Parameters
    random_state: int = 42
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    penalty: str = "l2"
    class_weight: str = "balanced"
    solver: str = "liblinear"

    # data Parameters
    feature_cols: List[str] = Field(
        default_factory=lambda: [
            "Age",
            "occupation",
            "gender",
            "operating_system",
            "phone_provider",
            "ENQ_3M",
            "has_group2_debt_12m",
            "NUM_CC_NON_BANK",
            "NUM_NEW_LOAN_12M",
            "MID_TERM_COUNT_NON_BANK",
            "NUM_NEW_LOAN_6M",
            "OUTS_BAL_LOAN_M1",
            "LONG_TERM_AMOUNT",
            "ENQ_9M",
            "NUM_CC_BANK",
            "NUM_NEW_LOAN_3M",
        ]
    )
    target: str = "FPD10+"

    @field_validator("solver")
    @classmethod
    def check_solver_penalty(cls, val: str, values):
        valid_solvers = ["liblinear", "lbfgs"]
        if val not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}")
        return val


def get_path_config() -> dict[str, Path]:
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "facts_dataset.csv"
    ARTIFACT_DIR = PROJECT_ROOT / "scripts" / "credit_score_model" / "artifacts"
    MODEL_PATH = ARTIFACT_DIR / "credit_risk_model.joblib"
    return {
        "data": DATA_PATH,
        "artifacts": ARTIFACT_DIR,
        "model": MODEL_PATH,
    }


def load_and_clean_data(config: ModelConfig, data_path: Path) -> Any:
    logger.info(f"loading data from {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8")
    df["gender"] = df["gender"].replace("Nam", "MALE")
    df["Age"] = df["Age"].replace(124, 24)
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])

    logger.info("extracting CIC features")
    cic_features = df["CIC_DATA"].apply(extract_cic_features)
    features_df = pd.json_normalize(cic_features)
    df_final = pd.concat([df, features_df], axis=1)

    features = [c for c in config.feature_cols if c in df_final.columns]
    numeric_features = [
        col
        for col in features
        if col in df_final.select_dtypes(include=[np.number]).columns
    ]
    categorical_features = [col for col in features if col not in numeric_features]

    # cast numeric features to float
    df_final[numeric_features] = df_final[numeric_features].astype(float)

    return df_final, numeric_features, categorical_features


def building_pipeline(
    config: ModelConfig, numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    # Preprocessing for numerical data
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=True,
    )

    # Create the full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "logreg",
                LogisticRegression(
                    penalty=config.penalty,
                    class_weight=config.class_weight,
                    solver=config.solver,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    return pipeline


def evaluate_and_log_model(
    config: ModelConfig,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    paths: Dict[str, Path],
) -> None:
    logger.info("Evaluating model performance")

    # =============== PREDICTING ===============
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]

    # =============== CORE METRICS CALCULATION ===============
    metrics = {
        "train_score": pipeline.score(X_train, y_train),
        "test_score": pipeline.score(X_test, y_test),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # ================ THRESHOLD OPTIMIZATION ==================
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)

    print(metrics, precision_curve, recall_curve, thresholds, y_proba_train)
