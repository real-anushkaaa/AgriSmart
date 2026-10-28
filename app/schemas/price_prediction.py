
from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class PricePredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    crop: str = Field(min_length=1, max_length=100)
    state: str = Field(min_length=1, max_length=100)
    market: str = Field(min_length=1, max_length=150)
    date: date = Field(description="Date in YYYY-MM-DD format")


class PricePredictionResponse(BaseModel):
    crop: str
    state: str
    market: str
    date: date
    price_guidance: str
