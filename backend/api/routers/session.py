# backend/api/routers/session.py
#
# Oturum yaşam döngüsü endpoint'leri.
#
# POST /session/start  → yeni oturum aç, session_id döner
# POST /session/end    → oturumu kapat, özet döner


from fastapi import APIRouter, HTTPException

from backend.agents.session_agent import SessionAgent
from backend.core.schemas import (
    SessionEndRequest,
    SessionEndResponse,
    SessionStartRequest,
    SessionStartResponse,
)

router = APIRouter()
_session_agent = SessionAgent()  # Uygulama boyunca tek instance


# ── POST /session/start ────────────────────────────────────────────────────────

@router.post("/start", response_model=SessionStartResponse)
def start_session(request: SessionStartRequest) -> SessionStartResponse:
    """
    Yeni bir öğrenme oturumu başlatır.

    İstek gövdesi:
        user_id        : kullanıcı kimliği
        topic          : "Bugün ne çalışıyorum?" (opsiyonel)
        camera_enabled : kamera açık mı?

    Döner:
        session_id : oturum boyunca her istekte gönderilecek kimlik
    """
    return _session_agent.start_session(request)


# ── POST /session/end ──────────────────────────────────────────────────────────

@router.post("/end")
def end_session(request: SessionEndRequest) -> dict:
    """
    Aktif oturumu kapatır.
    Önemli olayları memory'ye yazar (şimdilik log, Hafta 2'de Mem0).

    Döner:
        status                 : "ended" veya "not_found"
        memory_entries_written : kaç kayıt yazıldı
        topics_covered         : bu oturumda işlenen konular
    """
    result = _session_agent.end_session(request)

    if result.get("status") == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{request.session_id}' bulunamadı. "
                   "Oturum hiç başlatılmamış veya zaten kapatılmış olabilir.",
        )

    return result