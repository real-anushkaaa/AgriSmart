
from pydantic import BaseModel, Field


class TomatoDiseasePredictionResponse(BaseModel):
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)
