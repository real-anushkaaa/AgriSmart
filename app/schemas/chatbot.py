
from pydantic import BaseModel, Field


class ChatbotRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Agriculture-related question from the user",
    )


class ChatbotResponse(BaseModel):
    question: str
    answer: str
