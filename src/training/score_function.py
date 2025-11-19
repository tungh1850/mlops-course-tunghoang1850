# LOAD LIBRARIES
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


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
