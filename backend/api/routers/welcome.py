from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.services.continuity_service import ContinuityService

router = APIRouter()


@router.get("/welcome/{user_id}")
def get_welcome(
    user_id: str,
    topic: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    service = ContinuityService(db)
    return service.get_welcome(user_id, topic=topic)
