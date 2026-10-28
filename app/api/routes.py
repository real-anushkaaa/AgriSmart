
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models import CropRecord

from app.schemas.crop_record import CropRecordCreate, CropRecordResponse
from app.schemas.yield_prediction import (
    YieldPredictionRequest,
    YieldPredictionResponse,
)
from app.schemas.price_prediction import (
    PricePredictionRequest,
    PricePredictionResponse,
)
from app.schemas.tomato_disease import TomatoDiseasePredictionResponse
from app.schemas.soybean_disease import SoybeanDiseasePredictionResponse
from app.schemas.crop_calendar import CropCalendarRequest, CropCalendarResponse
from app.schemas.chatbot import ChatbotRequest, ChatbotResponse

from app.ml.yield_predictor import predict_crop_yield
from app.ml.tomato_disease_predictor import predict_tomato_disease
from app.ml.soybean_disease_predictor import predict_soybean_disease

from app.services.price_prediction_service import predict_crop_price
from app.services.crop_calendar_service import generate_crop_calendar
from app.services.chatbot_service import get_agriculture_answer


router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Welcome to AgriSmart",
        "status": "running",
        "app_name": settings.app_name,
    }


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.get("/info")
def app_info():
    return {
        "name": settings.app_name,
        "environment": settings.app_env,
        "version": "1.0.0",
        "features": [
            "crop_yield_prediction",
            "crop_price_prediction",
            "tomato_disease_detection",
            "soybean_disease_detection",
            "crop_calendar",
            "agriculture_chatbot",
        ],
    }


@router.get("/api/v1/db/health")
def database_health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"database": "connected"}
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable",
        ) from exc


@router.post(
    "/api/v1/crops",
    response_model=CropRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_crop_record(
    payload: CropRecordCreate,
    db: Session = Depends(get_db),
):
    record = CropRecord(**payload.model_dump())

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save crop record",
        ) from exc


@router.get(
    "/api/v1/crops",
    response_model=list[CropRecordResponse],
)
def list_crop_records(db: Session = Depends(get_db)):
    return (
        db.query(CropRecord)
        .order_by(CropRecord.id.desc())
        .all()
    )


@router.post(
    "/api/v1/yield/predict",
    response_model=YieldPredictionResponse,
    tags=["Crop Yield Prediction"],
)
def predict_yield(payload: YieldPredictionRequest):
    try:
        predicted_yield = predict_crop_yield(
            crop=payload.crop,
            crop_year=payload.crop_year,
            season=payload.season,
            state=payload.state,
            area=payload.area,
            annual_rainfall=payload.annual_rainfall,
            fertilizer=payload.fertilizer,
            pesticide=payload.pesticide,
        )
        return YieldPredictionResponse(
            crop=payload.crop,
            state=payload.state,
            crop_year=payload.crop_year,
            predicted_yield=round(predicted_yield, 4),
            model_type="RandomForestRegressor",
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crop yield prediction model is unavailable",
        ) from exc


@router.post(
    "/api/v1/prices/predict",
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
        return PricePredictionResponse(
            crop=payload.crop,
            state=payload.state,
            market=payload.market,
            date=payload.date,
            price_guidance=guidance,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Crop price guidance service is temporarily unavailable.",
        ) from exc


@router.post(
    "/api/v1/diseases/tomato/predict",
    response_model=TomatoDiseasePredictionResponse,
    tags=["Tomato Disease Detection"],
)
async def predict_tomato_disease_endpoint(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    try:
        result = predict_tomato_disease(image_bytes)
        return TomatoDiseasePredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tomato disease model is unavailable.",
        ) from exc


@router.post(
    "/api/v1/diseases/soybean/predict",
    response_model=SoybeanDiseasePredictionResponse,
    tags=["Soybean Disease Detection"],
)
async def predict_soybean_disease_endpoint(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    try:
        result = predict_soybean_disease(image_bytes)
        return SoybeanDiseasePredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Soybean disease model is unavailable.",
        ) from exc


@router.post(
    "/api/v1/crop-calendar/generate",
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
        return CropCalendarResponse(
            crop=payload.crop,
            location=payload.location,
            season=payload.season,
            calendar=calendar_text,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Crop calendar service is temporarily unavailable.",
        ) from exc


@router.post(
    "/api/v1/chatbot/ask",
    response_model=ChatbotResponse,
    tags=["AI Agriculture Chatbot"],
)
def ask_agriculture_chatbot(payload: ChatbotRequest):
    try:
        answer = get_agriculture_answer(payload.question)
        return ChatbotResponse(
            question=payload.question,
            answer=answer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Agriculture chatbot service is temporarily unavailable.",
        ) from exc
