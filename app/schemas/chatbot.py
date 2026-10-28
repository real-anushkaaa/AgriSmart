
from pydantic import BaseModel, ConfigDict, Field


class ChatbotRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Agriculture-related question from the user",
    )


class ChatbotResponse(BaseModel):
    question: str
    answer: str
