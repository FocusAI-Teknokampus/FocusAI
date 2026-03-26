from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.services.history_service import HistoryService

router = APIRouter()


@router.get("/history/sessions/{user_id}")
def get_user_sessions(
    user_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    service = HistoryService(db)
    return service.get_user_sessions(user_id)


@router.get("/history/session/{session_id}/messages")
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    service = HistoryService(db)
    return service.get_session_messages(session_id)
