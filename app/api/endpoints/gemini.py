from typing import NoReturn

from fastapi import APIRouter, HTTPException, status

from app.schemas.chatbot import ChatbotRequest, ChatbotResponse
from app.schemas.crop_calendar import CropCalendarRequest, CropCalendarResponse
from app.schemas.price_prediction import (
    PricePredictionRequest,
    PricePredictionResponse,
)
from app.services.chatbot_service import get_agriculture_answer
from app.services.crop_calendar_service import generate_crop_calendar
from app.services.errors import GeminiConfigurationError, GeminiServiceError
from app.services.price_prediction_service import predict_crop_price


router = APIRouter()


def _raise_gemini_http_error(
    exc: GeminiConfigurationError | GeminiServiceError,
    service_name: str,
) -> NoReturn:
    if isinstance(exc, GeminiConfigurationError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API is not configured.",
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"{service_name} service is temporarily unavailable.",
    ) from exc


@router.post(
    "/prices/predict",
    response_model=PricePredictionResponse,
    tags=["Crop Price Guidance"],
)
def predict_price(payload: PricePredictionRequest):
    try:
        guidance = predict_crop_price(
            crop=payload.crop,
            state=payload.state,
            market=payload.market,
            date=payload.date,
        )
    except (GeminiConfigurationError, GeminiServiceError) as exc:
        _raise_gemini_http_error(exc, "Crop price guidance")
    return PricePredictionResponse(
        crop=payload.crop,
        state=payload.state,
        market=payload.market,
        date=payload.date,
        price_guidance=guidance,
    )


@router.post(
    "/crop-calendar/generate",
    response_model=CropCalendarResponse,
    tags=["Crop Calendar"],
)
def create_crop_calendar(payload: CropCalendarRequest):
    try:
        calendar_text = generate_crop_calendar(
            crop=payload.crop,
            location=payload.location,
            season=payload.season,
            start_month=payload.start_month,
        )
    except (GeminiConfigurationError, GeminiServiceError) as exc:
        _raise_gemini_http_error(exc, "Crop calendar")
    return CropCalendarResponse(
        crop=payload.crop,
        location=payload.location,
        season=payload.season,
        calendar=calendar_text,
    )


@router.post(
    "/chatbot/ask",
    response_model=ChatbotResponse,
    tags=["AI Agriculture Chatbot"],
)
def ask_agriculture_chatbot(payload: ChatbotRequest):
    try:
        answer = get_agriculture_answer(payload.question)
    except (GeminiConfigurationError, GeminiServiceError) as exc:
        _raise_gemini_http_error(exc, "Agriculture chatbot")
    return ChatbotResponse(question=payload.question, answer=answer)
