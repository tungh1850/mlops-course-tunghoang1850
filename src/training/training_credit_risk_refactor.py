import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from pydantic import BaseModel, Field, field_validator
from score_function import (
    calculate_profit,
    calculate_psi_dataframe,
    costs,
    extract_cic_features,
    gini_from_auc,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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
    y: pd.Series,
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
    f1_curve = (
        2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    )
    best_f1_score = np.argmax(f1_curve)

    metrics.update(
        {
            "best_threshold": thresholds[best_f1_score],
            "best_f1": f1_curve[best_f1_score],
            "best_precision": precision_curve[best_f1_score],
            "best_recall": recall_curve[best_f1_score],
        }
    )

    # ================== PROFIT & GINI & PSI ================
    final_pred = (y_proba >= metrics["best_threshold"]).astype(int)
    default_profit = calculate_profit(y_test, y_pred, costs)
    optimized_profit = calculate_profit(y_test, final_pred, costs)

    metrics.update(
        {
            "gini_train": gini_from_auc(y_train, y_proba_train),
            "gini_test": gini_from_auc(y_test, y_proba),
            "default_profit": default_profit,
            "optimized_profit": optimized_profit,
            "optimized_different": optimized_profit - default_profit,
        }
    )

    # =============== PSI CALCULATION =========================
    preprocessor_step = pipeline.named_steps["preprocessor"]
    psi_df = calculate_psi_dataframe(
        preprocessor_step.transform(X_train), preprocessor_step.transform(X_test)
    )

    metrics.update({"total_psi": psi_df["psi"].sum(), "mean_psi": psi_df["psi"].mean()})

    mlflow.log_params(
        {
            "penalty": config.penalty,
            "class_weight": config.class_weight,
            "solver": config.solver,
            "random_state": config.random_state,
        }
    )
    mlflow.log_metrics(metrics=metrics)
    mlflow.log_table(data=psi_df, artifact_file="psi.json")

    # ============= ARTIFACTS: FEATURE IMPORTANCE ===================
    importance_map = pd.DataFrame(
        {
            "feature": preprocessor_step.get_feature_names_out(),
            "coefficient": pipeline.named_steps["logreg"].coef_[0],
        }
    ).sort_values(by="coefficient", ascending=False)

    mlflow.log_table(data=importance_map, artifact_file="feature_importance.json")
    logger.info(f"Top 5 Features:\n{importance_map.head(5)}")

    # ============== ARTIFACTS: PLOT ===================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = paths["artifacts"] / "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # ============== ARTIFACTS: PR CURVE ===================
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="Logistic Regression")

    plt.axhline(
        y=y.mean(), color="r", linestyle="--", label=f"random baseline {y.mean():.2f}"
    )
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    pr_curve_path = paths["artifacts"] / "precision_recall_curve.png"
    plt.savefig(pr_curve_path)
    mlflow.log_artifact(pr_curve_path)
    plt.close()

    # ============== ARTIFACTS: CLASSIFICATION REPORT ===================
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    mlflow.log_table(data=class_report_df, artifact_file="classification_report.json")
    logger.info(f"Classification Report:\n{class_report_df}")


def train_model() -> None:
    config = ModelConfig()
    paths = get_path_config()
    paths["artifacts"].mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(config.experiment_name)

    df, numeric_features, categorical_features = load_and_clean_data(
        config, paths["data"]
    )

    X = df[numeric_features + categorical_features]
    y = df[config.target]

    # Split the data
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    logger.info("building the pipeline")
    pipeline = building_pipeline(config, numeric_features, categorical_features)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_param("sklearn_config.transform_output", "pandas")
        logger.info("Training the model")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating and logging the model")
        evaluate_and_log_model(
            config, pipeline, y, X_train, y_train, X_test, y_test, paths
        )

        # SERIALIZE THE MODEL
        int_cols = X_train.select_dtypes(
            include=["int", "int64", "int32"]
        ).columns.tolist()
        logging.info(f"Integer Features in X_train: {int_cols}")

        # 2. Check the Target (This is likely the 'culprit')
        logging.info(f"Target Type: {y_train.dtype}")

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_train.iloc[:5],
        )
        joblib.dump(pipeline, paths["model"])
        mlflow.log_artifact(str(paths["model"]), "artifacts")
        mlflow.log_artifact(str(paths["data"]), "data")
        logger.info(f"Model saved at {paths['model']}")

        logger.info("Training run completed.")
        logger.info(f"Logged data and model in run {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_model()
