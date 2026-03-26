from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.schemas import FeedbackRequest, FeedbackResponse
from backend.services.feedback_service import FeedbackService

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    service = FeedbackService(db)
    return service.submit_feedback(request)
