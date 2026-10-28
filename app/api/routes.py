
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models import CropRecord
from app.schemas.crop_record import (
    CropRecordCreate,
    CropRecordResponse,
)

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
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable",
        )


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
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save crop record",
        )


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
