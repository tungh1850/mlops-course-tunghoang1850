# LOAD LIBRARIES
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger("credit_score_model_training")


# CODING SOME PRE-DEFINED FUNCTIONS
def gini_from_auc(y_true, y_score):
    """Calculate Gini coefficient from ROC AUC."""
    _auc = roc_auc_score(y_true, y_score)
    return 2 * _auc - 1


def _psi_from_props(train_percentage, test_percentage, eps=1e-6):
    # Compare input percentages to calculate PSI, replace zeros to avoid division by zero / log(0)
    _predicted = np.clip(train_percentage, eps, 1.0)
    _actual = np.clip(test_percentage, eps, 1.0)
    return np.sum((_predicted - _actual) * np.log(_predicted / _actual))


def compute_psi_series(_expected, _actual, bins=10):
    """
    Compute PSI for one feature (expected = reference/train, actual = new/test).
    Handles numeric (quantile bins) and categorical features.
    """
    _expected = pd.Series(_expected).dropna()
    _actual = pd.Series(_actual).dropna()
    if _expected.empty or _actual.empty:
        return np.nan

    _unique_vals = np.union1d(_expected.unique(), _actual.unique())
    _is_categorical = len(_unique_vals) <= bins

    try:
        # By default, use raw series for binning (categorical)
        _exp_binned, _act_binned = _expected, _actual
        _levels = list(_unique_vals)

        if not _is_categorical:
            # Numeric: try to create quantile bins
            _quantiles = np.linspace(0, 1, bins + 1)
            breaks = np.unique(np.quantile(_expected, _quantiles))

            if len(breaks) > 1:
                # If successful, use binned series and new levels
                _exp_binned = pd.cut(
                    _expected, bins=breaks, include_lowest=True, duplicates="drop"
                )
                _act_binned = pd.cut(
                    _actual, bins=breaks, include_lowest=True, duplicates="drop"
                )
                _levels = _exp_binned.cat.categories
            # If len(breaks) <= 1, it fails and automatically uses the categorical logic defined above

        _exp_counts = (
            _exp_binned.value_counts(normalize=True)
            .reindex(_levels, fill_value=0)
            .values
        )
        _act_counts = (
            _act_binned.value_counts(normalize=True)
            .reindex(_levels, fill_value=0)
            .values
        )

        return _psi_from_props(_exp_counts, _act_counts)

    except Exception:
        return np.nan


def calculate_psi_dataframe(train_df, test_df, bins=10, sort=True):
    """
    Calculate PSI for all columns present in both train_df and test_df.
    Returns a DataFrame with 'feature' and 'psi' columns.
    """
    _common_cols = [c for c in train_df.columns if c in test_df.columns]
    _results = []
    for col in _common_cols:
        _psi_val = compute_psi_series(train_df[col], test_df[col], bins=bins)
        _results.append((col, _psi_val))
    _psi_df = pd.DataFrame(_results, columns=["feature", "psi"])
    if sort:
        _psi_df = _psi_df.sort_values("psi", ascending=False).reset_index(drop=True)
    return _psi_df


def calculate_woe_iv(data, target_col, max_categories=20, n_bins=10):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for all features in a DataFrame
    against a specified binary target column.

    Assumes:
    - Target column is binary (0 = 'Good', 1 = 'Bad').
    - 'Good' is the non-event (e.g., loan paid).
    - 'Bad' is the event (e.g., loan default, FPD10+).

    Args:
        data (pd.DataFrame): The input DataFrame with features and target.
        target_col (str): The name of the binary target column.
        max_categories (int): Threshold for treating a numeric feature as categorical.
                               If nunique() <= max_categories, it's treated as categorical.
        n_bins (int): Number of bins to create for continuous features (using quantiles).

    Returns:
        tuple:
            - iv_summary_df (pd.DataFrame): DataFrame with 'Variable' and 'IV' columns,
                                            sorted by IV descending.
            - woe_tables (dict): A dictionary where keys are variable names and
                                 values are DataFrames containing the detailed WoE
                                 calculations for that variable.
    """

    # Separate features from target
    _features = [col for col in data.columns if col != target_col]
    # Calculate total goods and bads
    _total_bads = data[target_col].sum()
    _total_goods = len(data) - _total_bads
    # Add a small epsilon to prevent division by zero (ln(0))
    _eps = 0.5

    _iv_list = []
    _woe_tables = {}

    print(f"Total Goods (Target=0): {_total_goods}")
    print(f"Total Bads (Target=1): {_total_bads}")
    print("-" * 30)

    for feature in _features:
        try:
            # --- 1. Binning & Grouping ---
            # Create a clean version of the feature, filling NaNs
            _feature_data = data[feature].fillna("Missing")
            # Determine if continuous or categorical
            if (
                pd.api.types.is_numeric_dtype(data[feature])
                and data[feature].nunique() > max_categories
            ):
                # --- Continuous Feature ---
                # Bin using quantiles. We use data[feature] (not feature_data) (avoid trying to bin the 'Missing' string)
                _binned_feature = pd.qcut(
                    data[feature], q=n_bins, duplicates="drop"
                ).astype(str)
                # Combine binned data with the 'Missing' values (.cat.add_categories is for pd.Categorical, but .astype(str) handles it)
                _binned_feature = _binned_feature.fillna("Missing")

            else:
                # --- Categorical Feature ---
                # convert to string to ensure consistency
                _binned_feature = _feature_data.astype(str)
            # Name the binned series for the crosstab
            _binned_feature.name = "Category"
            # --- 2. crosstab Calculation ---
            # Create the contingency table
            _crosstab = pd.crosstab(_binned_feature, data[target_col])
            # Ensure both 'Good' (0) and 'Bad' (1) columns exist
            if 0 not in _crosstab.columns:
                _crosstab[0] = 0
            if 1 not in _crosstab.columns:
                _crosstab[1] = 0
            # Rename columns
            _crosstab = _crosstab.rename(columns={0: "Goods", 1: "Bads"})
            # Add epsilon smoothing
            _crosstab["Goods"] = _crosstab["Goods"] + _eps
            _crosstab["Bads"] = _crosstab["Bads"] + _eps

            # --- 3. WoE and IV Calculation ---
            # Calculate percentages
            _crosstab["%_Goods"] = _crosstab["Goods"] / (_total_goods + _eps)
            _crosstab["%_Bads"] = _crosstab["Bads"] / (_total_bads + _eps)
            # Calculate WoE
            _crosstab["WoE"] = np.log(_crosstab["%_Goods"] / _crosstab["%_Bads"])
            # Calculate IV for each bin
            _crosstab["IV_Bin"] = (
                _crosstab["%_Goods"] - _crosstab["%_Bads"]
            ) * _crosstab["WoE"]
            # Calculate totals for the category
            _crosstab["Total_Bin"] = _crosstab["Goods"] + _crosstab["Bads"]
            _crosstab["%_Population"] = _crosstab["Total_Bin"] / (
                _total_goods + _total_bads
            )

            # --- 4. Store Results ---
            # Sum IV for the entire variable
            total_iv = _crosstab["IV_Bin"].sum()
            # Get WoE values as a dictionary (bin: woe)
            woe_dict = _crosstab["WoE"].to_dict()
            # Append to the summary list
            _iv_list.append(
                {"Variable": feature, "IV": total_iv, "WoE_Values": woe_dict}
            )
            # Store the detailed WoE table
            _woe_tables[feature] = _crosstab.reset_index()
            # print(f"Successfully processed: {feature}")

        except Exception as e:
            print(f"--- FAILED to process {feature}: {e} ---")

    # --- 5. Final Output ---
    _iv_summary_df = (
        pd.DataFrame(_iv_list)
        .sort_values(by="IV", ascending=False)
        .reset_index(drop=True)
    )
    return _iv_summary_df, _woe_tables


# Helper function to safely get nested values
def safe_get(data_dict, keys, default=None):
    """
    Safely traverses a nested dictionary.
    Example: safe_get(data, ['QHTDHT', 'QHTD', 'DONG'])
    """
    _temp = data_dict
    for key in keys:
        if isinstance(_temp, dict):
            _temp = _temp.get(key, default)
        else:
            return default
    return _temp


def extract_cic_features(raw_json_string):
    """
    Takes one raw JSON string from the CIC_DATA column,
    cleans it, parses it, and extracts a flat dictionary of features.
    """

    # --- 1. Initialize all features with default values ---
    _features = {
        "cic_score": None,
        "cic_rank_percentile": None,
        "cic_rank_grade": None,
        "has_bad_debt_36m": None,
        "has_group2_debt_12m": None,
        "worst_current_debt_group": None,
        "has_collateral": None,
    }

    # --- 2. Clean and Parse ---
    try:
        data = json.loads(raw_json_string)
        # Get the main cic_content blob
        _cic_content = safe_get(data, ["NOIDUNG"], {})
        if not _cic_content:
            return _features  # Return defaults if no cic_content
    # If JSON is invalid, return the default feature set
    except Exception:
        return _features

    # --- 3. Extract Features (Tier 1) ---
    # convert "" or "N/A" to Nan
    _credit_score = safe_get(_cic_content, ["DIEMTD"], {})
    _features["cic_score"] = pd.to_numeric(
        safe_get(_credit_score, ["DIEM"]), errors="coerce"
    )
    _features["cic_rank_percentile"] = pd.to_numeric(
        safe_get(_credit_score, ["XEPHANGKH"]), errors="coerce"
    )
    _features["cic_rank_grade"] = pd.to_numeric(
        safe_get(_credit_score, ["HANG"]), errors="coerce"
    )

    _credit_history = safe_get(_cic_content, ["LSQHTD"], {})
    _baddebt_36m = safe_get(_credit_history, ["NOXAU_36THANG", "DONG"], [])
    _features["has_bad_debt_36m"] = (
        1 if isinstance(_baddebt_36m, list) and len(_baddebt_36m) > 0 else 0
    )

    _group2_12m = safe_get(_credit_history, ["NHOM2_12THANG"])  # This might be None
    _features["has_group2_debt_12m"] = (
        0 if _group2_12m is None else 1
    )  # Or more complex logic if it's a list

    # Get the list of currently active loans
    _active_loans = safe_get(_cic_content, ["QHTDHT", "QHTD", "DONG"], [])
    if isinstance(_active_loans, list) and len(_active_loans) > 0:
        # OBJECTIVE: find the WORST (highest number) debt group among all active loans. (Group 5 > Group 1)
        _worst_group = 0
        for loan in _active_loans:
            # Check inside CTLOAIVAY for debt groups
            _debt_detail = safe_get(loan, ["CTLOAIVAY", "DONG"], [])
            if isinstance(_debt_detail, list) and len(_debt_detail) > 0:
                _group_str = safe_get(_debt_detail[0], ["NHOMNO"])  # Get first group
                _group_num = pd.to_numeric(_group_str, errors="coerce")
                if pd.notna(_group_num) and _group_num > _worst_group:
                    _worst_group = int(_group_num)

        if _worst_group > 0:
            _features["worst_current_debt_group"] = _worst_group

    # --- 5. Extract Features (Tier 3) ---
    _collateral_assets_desc = safe_get(_cic_content, ["TSDB", "MOTA_TSDB"])
    if _collateral_assets_desc:
        _features["has_collateral"] = 1 if _collateral_assets_desc != "Không có" else 0

    return _features


costs = {
    "TP_gain": 100,  # Value of catching a bad guy
    "FP_cost": -5,  # Cost of bothering a good guy (Manual review cost)
    "FN_cost": -50,  # Cost of missing a bad guy (Theft amount)
    "TN_gain": 0,  # Value of letting a good guy through (usually 0 for risk models)
}


def calculate_profit(y_true, y_pred, costs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_profit = (
        (tp * costs["TP_gain"])
        + (fp * costs["FP_cost"])
        + (fn * costs["FN_cost"])
        + (tn * costs["TN_gain"])
    )
    return total_profit


def train() -> None:
    mlflow.set_experiment("credit_score_model_experiment")
    PROJECT_ROOT = Path("..")
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

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            # ('smote', SMOTE(sampling_strategy=1,random_state=42)),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    logger.info("pipeline built successfully.")

    with mlflow.start_run(run_name="credit_score_model_run_v1"):
        logger.info("Training model...")
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
        display = PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, name="Logistic Regression"
        )

        # 2. Add the "Random Baseline" for context
        # We use 0.091 because that is your specific positive rate
        plt.axhline(
            y=0.091, color="red", linestyle="--", label="Random Baseline (0.09)"
        )

        # 3. Format and Show
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")

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

        final_predictions = (y_proba >= best_threshold).astype(int)

        default_profit = calculate_profit(y_test, y_pred, costs)
        optimized_profit = calculate_profit(y_test, final_predictions, costs)
        diff = optimized_profit - default_profit

        print(
            MODEL_PATH,
            display,
            best_f1,
            best_threshold,
            best_precision,
            best_recall,
            default_profit,
            optimized_profit,
            diff,
        )
