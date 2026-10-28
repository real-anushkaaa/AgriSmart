
from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    crop: str = Field(min_length=1, max_length=100)
    state: str = Field(min_length=1, max_length=100)
    market: str = Field(min_length=1, max_length=150)
    date: str = Field(
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date in YYYY-MM-DD format",
    )


class PricePredictionResponse(BaseModel):
    crop: str
    state: str
    market: str
    date: str
    price_guidance: str
