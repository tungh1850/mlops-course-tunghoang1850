from pydantic import BaseModel


class PredictionResponse(BaseModel):
    FPD10_plus_probability: float
