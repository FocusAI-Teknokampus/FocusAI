# backend/api/routers/chat.py
#
# Chat endpoint'i — sistemin kalbi.
# Her kullanıcı mesajını alır, LangGraph üzerinden işler, yanıt döner.
#
# POST /chat  → mentor_graph.invoke() → ChatResponse
#
# Akış:
#   1. Mesajı al ve session'ın var olduğunu doğrula
#   2. Graph'ı invoke et (session → feature → state → uncertainty → mentor → rag → response)
#   3. Session context'ini güncelle (mesaj geçmişine ekle)
#   4. ChatResponse döner

import logging
from fastapi import APIRouter, HTTPException

from backend.agents.graph import mentor_graph
from backend.agents.session_agent import SessionAgent
from backend.core.schemas import ChatMessage, ChatResponse, UserState

logger = logging.getLogger(__name__)
router = APIRouter()

_session_agent = SessionAgent()


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
def chat(message: ChatMessage) -> ChatResponse:
    """
    Kullanıcıdan gelen mesajı işler, LLM yanıtı döner.

    İstek gövdesi (ChatMessage):
        session_id    : /session/start'tan alınan kimlik
        user_id       : kullanıcı kimliği
        content       : kullanıcının mesajı
        channel       : TEXT | IMAGE (varsayılan: TEXT)
        image_base64  : fotoğraf varsa base64 (opsiyonel)

    Döner (ChatResponse):
        content              : LLM'in ürettiği yanıt
        mentor_intervention  : varsa proaktif müdahale mesajı
        current_state        : kullanıcının tespit edilen durumu
        rag_source           : RAG kullandıysa nottan alınan bölüm
    """
    # ── 1. Session'ın var olduğunu doğrula ───────────────────────────
    context = _session_agent.load_context(message.session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{message.session_id}' bulunamadı. "
                   "Önce /session/start ile oturum açın.",
        )

    # ── 2. Graph'ı çalıştır ───────────────────────────────────────────
    # Initial state: sadece gelen mesaj dolu, gerisini node'lar dolduracak.
    initial_state = {
        "message":          message,
        "session_context":  None,
        "user_profile":     None,
        "feature_vector":   None,
        "state_estimate":   None,
        "should_intervene": False,
        "intervention":     None,
        "rag_context":      None,
        "llm_response":     None,
        "final_response":   None,
        "error":            None,
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

    # ── 3. Yanıtı al ─────────────────────────────────────────────────
    final: ChatResponse | None = result.get("final_response")

    if final is None:
        # Graph tamamlandı ama response üretilemedi — savunmacı fallback.
        raise HTTPException(
            status_code=500,
            detail="Graph tamamlandı ancak yanıt üretilemedi.",
        )

    # ── 4. Session context'ini güncelle ──────────────────────────────
    # Kullanıcı mesajını ve asistan yanıtını geçmişe ekle.

    state_estimate = result.get("state_estimate")
    feature_vector = result.get("feature_vector")

    new_state = state_estimate.state if state_estimate else None
    detected_topic = feature_vector.topic if feature_vector and feature_vector.topic else None

    _session_agent.update_context(
        session_id=message.session_id,
        role="user",
        content=message.content,
        new_state=new_state,
        topic=detected_topic,
    )

    _session_agent.update_context(
        session_id=message.session_id,
        role="assistant",
        content=final.content,
        topic=detected_topic,
    )

    return final