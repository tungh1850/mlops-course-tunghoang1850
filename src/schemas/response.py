from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    FPD10_plus_probability: float
    model_name: str
    model_version: str
    model_run_id: Optional[str] = None
