# backend/api/routers/chat.py

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.agents.graph import mentor_graph
from backend.agents.session_agent import SessionAgent
from backend.core.database import get_db
from backend.core.schemas import ChatMessage, ChatResponse
from backend.services.session_service import SessionService
from backend.services.behavior_service import BehaviorService

logger = logging.getLogger(__name__)
router = APIRouter()

_session_agent = SessionAgent()


@router.post("", response_model=ChatResponse)
def chat(
    message: ChatMessage,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Kullanıcı mesajını işler ve mentor yanıtı döner.

    Akış:
        1. Session RAM'de var mı kontrol edilir
        2. Graph çalıştırılır
        3. User message RAM + DB'ye yazılır
        4. Behavior/focus event'leri kaydedilir
        5. Intervention varsa kaydedilir
        6. Assistant message RAM + DB'ye yazılır
    """
    context = _session_agent.load_context(message.session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"session_id '{message.session_id}' bulunamadı. "
                "Önce /session/start ile oturum açın."
            ),
        )

    initial_state = {
        "message": message,
        "session_context": None,
        "user_profile": None,
        "baseline_profile": None,
        "feature_vector": None,
        "state_estimate": None,
        "response_policy": None,
        "should_intervene": False,
        "intervention": None,
        "rag_context": None,
        "rag_source": None,
        "llm_response": None,
        "final_response": None,
        "error": None,
    }

    try:
        result = mentor_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph çalıştırılırken hata oluştu: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Yanıt üretilirken bir sorun oluştu. Lütfen tekrar deneyin.",
        )

    final: ChatResponse | None = result.get("final_response")
    if final is None:
        raise HTTPException(
            status_code=500,
            detail="Graph tamamlandı ancak yanıt üretilemedi.",
        )

    state_estimate = result.get("state_estimate")
    feature_vector = result.get("feature_vector")
    intervention = result.get("intervention")

    new_state = state_estimate.state if state_estimate else None
    detected_topic = feature_vector.topic if feature_vector and feature_vector.topic else None
    llm_confidence = state_estimate.confidence if state_estimate else None

    # 1) Kullanıcı mesajını kaydet
    _session_agent.update_context(
        session_id=message.session_id,
        role="user",
        content=message.content,
        db=db,
        new_state=new_state,
        topic=detected_topic,
        message_type="question",
        llm_confidence=llm_confidence,
    )

    # 2) Davranış + focus event'lerini kaydet
    behavior_service = BehaviorService(db)
    behavior_service.persist_analysis(
        session_id=message.session_id,
        user_id=message.user_id,
        feature_vector=feature_vector,
        state_estimate=state_estimate,
    )

    # 3) Müdahale varsa kaydet
    if intervention is not None and intervention.message:
        session_service = SessionService(db)
        session_service.save_intervention(
            session_id=message.session_id,
            user_id=message.user_id,
            intervention_type=intervention.intervention_type.value,
            message=intervention.message,
            triggered_by=intervention.triggered_by.value if intervention.triggered_by else None,
            reason=intervention.decision_reason or "Graph uncertainty/mentor flow sonucunda uretildi.",
            confidence=intervention.confidence,
            was_successful=None,
        )

    # 4) Asistan cevabını kaydet
    _session_agent.update_context(
        session_id=message.session_id,
        role="assistant",
        content=final.content,
        db=db,
        new_state=new_state,
        topic=detected_topic,
        message_type="answer",
        llm_confidence=llm_confidence,
    )

    return final
