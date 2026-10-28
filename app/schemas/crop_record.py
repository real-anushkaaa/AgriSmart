
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CropRecordCreate(BaseModel):
    crop_name: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=150)
    area_hectares: float = Field(gt=0)


class CropRecordResponse(BaseModel):
    id: int
    crop_name: str
    location: str
    area_hectares: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
