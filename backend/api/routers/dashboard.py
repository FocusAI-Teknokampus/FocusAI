# backend/api/routers/dashboard.py
#
# Dashboard endpoint'leri.
# Şu an temel bilgi döner — Hafta 2'de SQLite/Mem0'dan gerçek veri gelecek.
#
# GET /dashboard/{session_id}  → oturum özeti
# GET /dashboard/profile/{user_id}  → kullanıcı profili


from fastapi import APIRouter, HTTPException

from backend.agents.session_agent import SessionAgent, _active_sessions
from backend.core.schemas import SessionSummary

router = APIRouter()
_session_agent = SessionAgent()


# ── GET /dashboard/{session_id} ───────────────────────────────────────────────

@router.get("/{session_id}")
def get_session_summary(session_id: str) -> dict:
    """
    Aktif ya da yakın zamanda kapanmış oturumun özetini döner.

    Şu an döndürdükleri:
        session_id       : oturum kimliği
        user_id          : kullanıcı kimliği
        topic            : çalışılan konu
        message_count    : kaç mesaj gönderildi
        topics_covered   : işlenen konular
        current_state    : son tespit edilen durum
        retry_count      : takılma sayacı

    Hafta 2'de eklenecekler:
        focus_timeline   : odak skoru grafiği için zaman serisi
        avg_focus_score  : ortalama odak skoru
        intervention_count : kaç kez müdahale edildi
        recommended_topics : yarın için önerilen konular
    """
    context = _active_sessions.get(session_id)

    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"session_id '{session_id}' bulunamadı veya oturum kapatılmış.",
        )

    return {
        "session_id":     context.session_id,
        "user_id":        context.user_id,
        "topic":          context.topic,
        "message_count":  len(context.messages),
        "topics_covered": context.topics_covered,
        "current_state":  context.current_state.value,
        "retry_count":    context.retry_count,
    }


# ── GET /dashboard/profile/{user_id} ─────────────────────────────────────────

@router.get("/profile/{user_id}")
def get_user_profile(user_id: str) -> dict:
    """
    Kullanıcının long-term profilini döner.
    Hafta 2'de Mem0'dan gerçek veri gelecek, şimdilik varsayılan profil.
    """
    profile = _session_agent.load_profile(user_id)

    return {
        "user_id":                    profile.user_id,
        "preferred_explanation_style": profile.preferred_explanation_style,
        "weak_topics":                profile.weak_topics,
        "strong_topics":              profile.strong_topics,
        "recurring_misconceptions":   profile.recurring_misconceptions,
        "adaptive_threshold":         profile.adaptive_threshold,
        "total_sessions":             profile.total_sessions,
    }