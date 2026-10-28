from fastapi import APIRouter, Depends

from app.api.endpoints import core, crops, database, gemini, predictions
from app.api.security import require_api_key
from app.core.config import settings


router = APIRouter()
router.include_router(core.router)

versioned_router = APIRouter(
    prefix="/" + settings.api_v1_prefix.strip("/"),
    dependencies=[Depends(require_api_key)],
)
versioned_router.include_router(database.router)
versioned_router.include_router(crops.router)
versioned_router.include_router(predictions.router)
versioned_router.include_router(gemini.router)
router.include_router(versioned_router)
