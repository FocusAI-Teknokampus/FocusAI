# backend/api/routers/dashboard.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.agents.session_agent import SessionAgent, _active_sessions
from backend.core.database import get_db, SessionRecord
from backend.services.analytics_service import AnalyticsService

router = APIRouter()
_session_agent = SessionAgent()


def _empty_dashboard_response(session_id: str, user_id: str | None = None) -> dict:
    """
    Dashboard response formatını her durumda sabit tutmak için
    boş/default yapı döner.
    """
    return {
        "session_id": session_id,
        "user_id": user_id,
        "topic": None,
        "subtopic": None,
        "study_mode": None,
        "camera_used": False,
        "started_at": None,
        "ended_at": None,
        "current_state": None,
        "average_focus_score": None,
        "retry_count": 0,
        "intervention_count": 0,
        "focus_timeline": [],
        "behavior_timeline": [],
        "latest_state_analysis": None,
        "latest_intervention": None,
        "report": {
            "message_count": 0,
            "intervention_count": 0,
            "retry_count": 0,
            "topics_covered": [],
            "focus_score": None,
            "summary_text": None,
            "behavior_summary": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "next_session_plan": {},
        },
        "source": None,
    }


@router.get("/{session_id}")
def get_session_summary(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Session dashboard verisini döner.

    Öncelik:
        1. Aktif session ise RAM fallback
        2. DB'de report/dashboard verisi varsa onu döner
        3. Sadece session kaydı varsa minimum özet döner
    """
    # 1) Aktif session RAM'de ise
    context = _active_sessions.get(session_id)
    if context is not None:
        analytics = AnalyticsService(db)
        dashboard = analytics.get_session_dashboard(session_id)
        if dashboard is not None:
            dashboard["current_state"] = context.current_state.value
            dashboard["retry_count"] = context.retry_count
            dashboard["source"] = "active_database"
            return dashboard

        payload = _empty_dashboard_response(
            session_id=context.session_id,
            user_id=context.user_id,
        )
        payload.update({
            "topic": context.topic,
            "current_state": context.current_state.value,
            "retry_count": context.retry_count,
            "report": {
                **payload["report"],
                "message_count": len(context.messages),
                "retry_count": context.retry_count,
                "topics_covered": context.topics_covered,
            },
            "source": "ram",
        })
        return payload

    # 2) DB dashboard
    analytics = AnalyticsService(db)
    dashboard = analytics.get_session_dashboard(session_id)
    if dashboard is not None:
        dashboard["source"] = "database"
        return dashboard

    # 3) En azından session kaydı varsa tek tip response dön
    row = (
        db.query(SessionRecord)
        .filter(SessionRecord.session_id == session_id)
        .first()
    )
    if row:
        payload = _empty_dashboard_response(
            session_id=row.session_id,
            user_id=row.user_id,
        )
        payload.update({
            "topic": row.topic,
            "subtopic": row.subtopic,
            "current_state": row.current_state,
            "retry_count": row.retry_count,
            "intervention_count": row.intervention_count,
            "average_focus_score": row.average_focus_score,
            "source": "database_session_only",
        })
        payload["report"]["retry_count"] = row.retry_count
        payload["report"]["intervention_count"] = row.intervention_count
        payload["report"]["focus_score"] = row.average_focus_score
        return payload

    raise HTTPException(
        status_code=404,
        detail=f"session_id '{session_id}' bulunamadı.",
    )


@router.get("/profile/{user_id}")
def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Kullanıcı profilini döner.
    """
    profile = _session_agent.load_profile(user_id, db)

    return {
        "user_id": profile.user_id,
        "preferred_explanation_style": profile.preferred_explanation_style,
        "weak_topics": profile.weak_topics,
        "strong_topics": profile.strong_topics,
        "recurring_misconceptions": profile.recurring_misconceptions,
        "adaptive_threshold": profile.adaptive_threshold,
        "total_sessions": profile.total_sessions,
    }
