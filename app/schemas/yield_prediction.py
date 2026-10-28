
from pydantic import BaseModel, ConfigDict, Field


class YieldPredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    crop: str = Field(min_length=1, max_length=100)
    crop_year: int = Field(ge=1990, le=2100)
    season: str = Field(min_length=1, max_length=100)
    state: str = Field(min_length=1, max_length=100)
    area: float = Field(gt=0, allow_inf_nan=False)
    annual_rainfall: float = Field(ge=0, allow_inf_nan=False)
    fertilizer: float = Field(ge=0, allow_inf_nan=False)
    pesticide: float = Field(ge=0, allow_inf_nan=False)


class YieldPredictionResponse(BaseModel):
    crop: str
    state: str
    crop_year: int
    predicted_yield: float
    model_type: str = "RandomForestRegressor"
