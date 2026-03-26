from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.agents.session_agent import SessionAgent
from backend.core.database import get_db
from backend.core.schemas import CameraFrameRequest, CameraStatusResponse
from backend.services.camera_runtime_service import camera_runtime_service
from backend.services.session_service import SessionService

router = APIRouter()
_session_agent = SessionAgent()


@router.post("/frame", response_model=CameraStatusResponse)
def process_camera_frame(
    request: CameraFrameRequest,
    db: Session = Depends(get_db),
) -> CameraStatusResponse:
    context = _session_agent.load_context(request.session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{request.session_id}' bulunamadi.",
        )

    if context.user_id != request.user_id:
        raise HTTPException(
            status_code=400,
            detail="Kamera frame'i farkli bir kullaniciya ait oturuma gonderilemez.",
        )

    status = camera_runtime_service.process_frame(
        session_id=request.session_id,
        image_base64=request.image_base64,
    )
    SessionService(db).mark_camera_used(request.session_id)
    return status


@router.get("/{session_id}/status", response_model=CameraStatusResponse)
def get_camera_status(
    session_id: str,
    _: Session = Depends(get_db),
) -> CameraStatusResponse:
    context = _session_agent.load_context(session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{session_id}' bulunamadi.",
        )
    return camera_runtime_service.get_status(session_id)
