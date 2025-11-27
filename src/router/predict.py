import os

import mlflow.sklearn
import pandas as pd
import sklearn
from fastapi import APIRouter
from mlflow.tracking import MlflowClient

from src.schemas.request import PredictionRequest
from src.schemas.response import PredictionResponse

MLFLOW_TRACKING_URI = os.getenv("OUR_MLFLOW_HOST", "http://localhost:5050")
print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
sklearn.set_config(transform_output="pandas")

# Model configuration
model_name = "credit_risk_FPD10_plus"
version = "2"
model_uri = f"models:/{model_name}/{version}"

router = APIRouter(prefix="/credit_risk")

# Global variables for caching model and client
_model = None
_client = None
_model_run_id = None


# Load model and get metadata
def get_model():
    global _model
    global _client
    global _model_run_id
    if _model is None:
        print(f"Loading model from: {model_uri}")
        _model = mlflow.sklearn.load_model(model_uri)
        _client = MlflowClient()
        model_version_details = _client.get_model_version(
            name=model_name, version=version
        )
        _model_run_id = model_version_details.run_id
        print(f"Loaded model: {model_name} v{version} (run_id: {_model_run_id})")
    return _model, _model_run_id


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model, model_run_id = get_model()
        input_data = {
            "Age": request.Age,
            "occupation": request.occupation.value,
            "gender": request.gender.value,
            "operating_system": request.operating_system.value,
            "phone_provider": request.phone_provider.value,
            "ENQ_3M": request.ENQ_3M,
            "has_group2_debt_12m": request.has_group2_debt_12m,
            "NUM_CC_NON_BANK": request.NUM_CC_NON_BANK,
            "NUM_NEW_LOAN_12M": request.NUM_NEW_LOAN_12M,
            "MID_TERM_COUNT_NON_BANK": request.MID_TERM_COUNT_NON_BANK,
            "NUM_NEW_LOAN_6M": request.NUM_NEW_LOAN_6M,
            "OUTS_BAL_LOAN_M1": request.OUTS_BAL_LOAN_M1,
            "LONG_TERM_AMOUNT": request.LONG_TERM_AMOUNT,
            "ENQ_9M": request.ENQ_9M,
            "NUM_CC_BANK": request.NUM_CC_BANK,
            "NUM_NEW_LOAN_3M": request.NUM_NEW_LOAN_3M,
        }
        df = pd.DataFrame([input_data])
        print(f"Input data: {input_data}")
        probability = model.predict_proba(df)[:, 1][0]
        probability = float(probability)
        print(f"Prediction: {probability}")
        return PredictionResponse(
            FPD10_plus_probability=probability,
            model_name=model_name,
            model_version=version,
            model_run_id=model_run_id,
        )
        # some change
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
