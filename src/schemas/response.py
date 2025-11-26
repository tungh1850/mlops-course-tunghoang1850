from typing import Optional

from pydantic import BaseModel


# Response schema for prediction results
class PredictionResponse(BaseModel):
    FPD10_plus_probability: float
    model_name: str
    model_version: str
    model_run_id: Optional[str] = None
