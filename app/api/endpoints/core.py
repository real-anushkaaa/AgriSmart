from fastapi import APIRouter

from app.core.config import settings


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
