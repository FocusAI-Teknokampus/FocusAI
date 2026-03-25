# backend/api/routers/session.py
#
# Oturum yaşam döngüsü endpoint'leri.
#
# POST /session/start  → yeni oturum aç, session_id döner
# POST /session/end    → oturumu kapat, özet döner

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.agents.session_agent import SessionAgent
from backend.core.database import get_db
from backend.core.schemas import (
    SessionEndRequest,
    SessionStartRequest,
    SessionStartResponse,
)

router = APIRouter()
_session_agent = SessionAgent()  # Uygulama boyunca tek instance


# ── POST /session/start ────────────────────────────────────────────────────────

@router.post("/start", response_model=SessionStartResponse)
def start_session(
    request: SessionStartRequest,
    db: Session = Depends(get_db),
) -> SessionStartResponse:
    """
    Yeni bir öğrenme oturumu başlatır.

    İstek gövdesi:
        user_id        : kullanıcı kimliği
        topic          : "Bugün ne çalışıyorum?" (opsiyonel)
        camera_enabled : kamera açık mı?

    Döner:
        session_id : oturum boyunca her istekte gönderilecek kimlik

    Veritabanı entegrasyonu:
        - DB session FastAPI'nin Depends(get_db) mekanizmasıyla alınır
        - SessionAgent hem RAM context'i hem DB session kaydını oluşturur
    """
    return _session_agent.start_session(request, db)


# ── POST /session/end ──────────────────────────────────────────────────────────

@router.post("/end")
def end_session(
    request: SessionEndRequest,
    db: Session = Depends(get_db),
) -> dict:
    """
    Aktif oturumu kapatır.

    Yapılanlar:
        1. RAM'deki kısa dönem context okunur
        2. Önemli olaylar MemoryEntry olarak çıkarılır
        3. Mem0'a yazılır
        4. Session DB'de kapatılır
        5. Session report üretilir

    Döner:
        status                 : "ended" veya "not_found"
        memory_entries_written : Mem0'a kaç kayıt yazıldı
        topics_covered         : bu oturumda işlenen konular
        summary_text           : oturum sonu kısa özet
    """
    result = _session_agent.end_session(request, db)

    if result.get("status") == "not_found":
        raise HTTPException(
            status_code=404,
            detail=(
                f"session_id '{request.session_id}' bulunamadı. "
                "Oturum hiç başlatılmamış veya zaten kapatılmış olabilir."
            ),
        )

    return result