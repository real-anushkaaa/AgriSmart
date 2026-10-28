import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.ml.errors import InvalidImageError, ModelUnavailableError
from app.ml.soybean_disease_predictor import predict_soybean_disease
from app.ml.tomato_disease_predictor import predict_tomato_disease
from app.ml.yield_predictor import predict_crop_yield
from app.schemas.soybean_disease import SoybeanDiseasePredictionResponse
from app.schemas.tomato_disease import TomatoDiseasePredictionResponse
from app.schemas.yield_prediction import (
    YieldPredictionRequest,
    YieldPredictionResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/yield/predict",
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
    except ModelUnavailableError as exc:
        logger.exception("Crop yield model is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crop yield prediction model is unavailable",
        ) from exc

    return YieldPredictionResponse(
        crop=payload.crop,
        state=payload.state,
        crop_year=payload.crop_year,
        predicted_yield=round(predicted_yield, 4),
        model_type="RandomForestRegressor",
    )


async def _read_image(file: UploadFile) -> bytes:
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Please upload an image file.",
            )

        image_bytes = await file.read(settings.max_image_upload_bytes + 1)
        if len(image_bytes) > settings.max_image_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image exceeds the upload size limit.",
            )
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )
        return image_bytes
    finally:
        await file.close()


async def _predict_disease(file: UploadFile, predictor, response_model):
    image_bytes = await _read_image(file)
    try:
        result = await run_in_threadpool(predictor, image_bytes)
        return response_model(**result)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelUnavailableError as exc:
        logger.exception("Disease model is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Disease prediction model is unavailable.",
        ) from exc


@router.post(
    "/diseases/tomato/predict",
    response_model=TomatoDiseasePredictionResponse,
    tags=["Tomato Disease Detection"],
)
async def predict_tomato_disease_endpoint(file: UploadFile = File(...)):
    return await _predict_disease(
        file, predict_tomato_disease, TomatoDiseasePredictionResponse
    )


@router.post(
    "/diseases/soybean/predict",
    response_model=SoybeanDiseasePredictionResponse,
    tags=["Soybean Disease Detection"],
)
async def predict_soybean_disease_endpoint(file: UploadFile = File(...)):
    return await _predict_disease(
        file, predict_soybean_disease, SoybeanDiseasePredictionResponse
    )
