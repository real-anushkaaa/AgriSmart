import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models import CropRecord
from app.schemas.crop_record import CropRecordCreate, CropRecordResponse


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/crops",
    response_model=CropRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_crop_record(payload: CropRecordCreate, db: Session = Depends(get_db)):
    record = CropRecord(**payload.model_dump())
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to save crop record")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save crop record",
        ) from exc


@router.get("/crops", response_model=list[CropRecordResponse])
def list_crop_records(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    try:
        return (
            db.query(CropRecord)
            .order_by(CropRecord.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to list crop records")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crop records are temporarily unavailable",
        ) from exc
