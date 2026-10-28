
from pydantic import BaseModel, Field


class CropCalendarRequest(BaseModel):
    crop: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=150)
    season: str = Field(min_length=1, max_length=100)
    start_month: int = Field(ge=1, le=12)


class CropCalendarResponse(BaseModel):
    crop: str
    location: str
    season: str
    calendar: str
