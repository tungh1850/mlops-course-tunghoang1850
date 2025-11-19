# LOAD LIBRARIES
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from mlflow.models import infer_signature
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
    average_precision_score,
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

EXPERIMENT_NAME = "credit_score_model_experiment_v3"
logger = logging.getLogger("credit_score_model_training")
sklearn.set_config(transform_output="pandas")


def train() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "facts_dataset.csv"
    ARTIFACT_DIR = PROJECT_ROOT / "scripts" / "credit_score_model" / "artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "credit_score_model.joblib"

    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")

    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    logger.info("Preprocessing data...")
    df["gender"] = df["gender"].replace("Nam", "MALE")
    df["Age"] = df["Age"].replace(124, 24)
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])

    feature_series = df["CIC_DATA"].apply(extract_cic_features)
    features_df = pd.json_normalize(feature_series)
    df_final = pd.concat([df, features_df], axis=1)

    # Create Preprocessing Pipelines
    logger.info("Preparing features and target...")
    feature_cols = [
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
    TARGET = "FPD10+"
    NUM_FEATURES = [
        col
        for col in feature_cols
        if col in df_final.select_dtypes(include=[np.number]).columns
    ]
    df_final[NUM_FEATURES] = df_final[NUM_FEATURES].astype(float)
    CAT_FEATURES = [col for col in feature_cols if col not in NUM_FEATURES]
    X = df_final[NUM_FEATURES + CAT_FEATURES]
    y = df_final[TARGET]

    logger.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Building pipeline...")
    num_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_transformer, NUM_FEATURES), ("cat", cat_transformer, CAT_FEATURES)]
    )

    _penalty = "l2"
    _class_weight = "balanced"
    _solver = "liblinear"
    _random_state = 42
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            # ('smote', SMOTE(sampling_strategy=1,random_state=42)),
            (
                "logreg",
                LogisticRegression(
                    penalty=_penalty,
                    class_weight=_class_weight,
                    solver=_solver,
                    random_state=_random_state,
                ),
            ),
        ]
    )
    logger.info("pipeline built successfully.")

    with mlflow.start_run(run_name="credit_score_model_run_v1"):
        logger.info("Training model...")
        mlflow.log_param("penalty", _penalty)
        mlflow.log_param("class_weight", _class_weight)
        mlflow.log_param("solver", _solver)
        mlflow.log_param("random_state", _random_state)
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating model...")
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        _precision = precision_score(y_test, y_pred)
        _recall = recall_score(y_test, y_pred)
        _f1 = f1_score(y_test, y_pred)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_pr = average_precision_score(y_test, y_proba)
        logger.info(
            f"Train Score: {train_score:.4f} | Test Score: {test_score:.4f} | Precision: {_precision:.4f} | Recall: {_recall:.4f} | F1: {_f1:.4f} | AUC-PR: {auc_pr:.4f}"
        )

        metrics_report = classification_report(y_test, y_pred)
        with open(ARTIFACT_DIR / "classification_report.txt", "w") as f:
            f.write(metrics_report)
        logger.info(f"Classification Report:\n{metrics_report}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(ARTIFACT_DIR / "confusion_matrix.png")

        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, name="Logistic Regression"
        )
        # 2. Add the "Random Baseline" for context
        # We use 0.091 because that is the specific positive rate
        plt.axhline(
            y=0.091, color="red", linestyle="--", label="Random Baseline (0.09)"
        )
        # 3. Format and Show
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")
        plt.savefig(ARTIFACT_DIR / "precision_recall_curve.png")
        plt.close()

        # 1. Calculate the curve points
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        # 2. Calculate F1-Score for every single threshold point
        # Note: We add a small epsilon (1e-10) to avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # 3. Find the index of the highest F1 score
        best_idx = np.argmax(f1_scores)

        # 4. Get the optimal values
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        logger.info(
            f"Best Threshold: {best_threshold:.4f} | Best F1: {best_f1:.4f} | Precision: {best_precision:.4f} | Recall: {best_recall:.4f}"
        )

        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        transfomer = pipeline.named_steps["preprocessor"]
        transfomer.set_output(transform="pandas")
        # compute for train and test (uses existing y_train, y_prob_train, y_test, y_proba / y_prob_test)
        _gini_train = gini_from_auc(y_train, y_prob_train)
        _gini_test = gini_from_auc(y_test, y_proba)
        # Compute PSI between training and test preprocessed feature sets

        _psi_df = calculate_psi_dataframe(
            transfomer.transform(X_train), transfomer.transform(X_test), bins=10
        )
        # Summary metrics
        _total_psi = _psi_df["psi"].sum()
        _mean_psi = _psi_df["psi"].mean()
        logger.info(f"Gini Train: {_gini_train:.4f} | Gini Test: {_gini_test:.4f}")
        logger.info(f"Total PSI: {_total_psi:.4f} | Mean PSI: {_mean_psi:.4f}")

        # Calculate profit metrics
        final_predictions = (y_proba >= best_threshold).astype(int)
        default_profit = calculate_profit(y_test, y_pred, costs)
        optimized_profit = calculate_profit(y_test, final_predictions, costs)
        diff = optimized_profit - default_profit

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefficients = pipeline.named_steps["logreg"].coef_[0]
        importance_map = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefficients}
        )
        logger.info(
            f"Feature Importance:\n{importance_map.sort_values(by='coefficient', ascending=False)}"
        )

        mlflow.log_metric("gini_train", _gini_train)
        mlflow.log_metric("total_psi", _total_psi)
        mlflow.log_metric("mean_psi", _mean_psi)
        mlflow.log_metric("gini_test", _gini_test)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("best_precision", best_precision)
        mlflow.log_metric("best_recall", best_recall)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("precision", _precision)
        mlflow.log_metric("recall", _recall)
        mlflow.log_metric("f1_score", _f1)
        mlflow.log_metric("auc_pr", auc_pr)
        mlflow.log_table(data=_psi_df, artifact_file="psi_dataframe.json")
        mlflow.log_table(data=importance_map, artifact_file="feature_importance.json")
        mlflow.log_artifact(ARTIFACT_DIR / "classification_report.txt")
        mlflow.log_artifact(ARTIFACT_DIR / "precision_recall_curve.png")
        mlflow.log_artifact(ARTIFACT_DIR / "classification_report.txt")
        mlflow.log_artifact(ARTIFACT_DIR / "confusion_matrix.png")

        mlflow.log_metric("default_profit", default_profit)
        mlflow.log_metric("optimized_profit", optimized_profit)
        mlflow.log_metric("profit_difference", diff)
        mlflow.log_artifact(MODEL_PATH, "artifacts")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, y_train),
            input_example=X_train.head(),
        )
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
