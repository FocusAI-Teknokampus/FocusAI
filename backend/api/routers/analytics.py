from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.services.history_service import HistoryService

router = APIRouter()


@router.get("/analytics/focus-history/{user_id}")
def get_focus_history(
    user_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    service = HistoryService(db)
    return service.get_focus_history(user_id)


@router.get("/analytics/focus-trend/{user_id}")
def get_focus_trend(
    user_id: str,
    days: int = 7,
    db: Session = Depends(get_db),
) -> dict:
    service = HistoryService(db)
    return service.get_focus_trend(user_id=user_id, days=days)
