import mlflow.sklearn
import pandas as pd
import sklearn

sklearn.set_config(transform_output="pandas")
model_name = "credit_risk_FPD10+"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

"""
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
"""

mock_input = {
    "Age": 30,
    "occupation": "Engineer",
    "gender": "Male",
    "operating_system": "iOS",
    "phone_provider": "viettel",
    "ENQ_3M": 0,
    "has_group2_debt_12m": 0,
    "NUM_CC_NON_BANK": 1,
    "NUM_NEW_LOAN_12M": 0,
    "MID_TERM_COUNT_NON_BANK": 0,
    "NUM_NEW_LOAN_6M": 0,
    "OUTS_BAL_LOAN_M1": 0,
    "LONG_TERM_AMOUNT": 0,
    "ENQ_9M": 1,
    "NUM_CC_BANK": 2,
    "NUM_NEW_LOAN_3M": 0,
}
prediction = model.predict_proba(pd.DataFrame([mock_input]))[:, 1][0]
print(f"Predicted probability of FPD10+: {prediction:.4f}")
