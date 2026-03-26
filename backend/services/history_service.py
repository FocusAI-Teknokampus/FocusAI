import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    FocusEventRecord,
    InterventionRecord,
    MessageRecord,
    SessionRecord,
    SessionReportRecord,
    BehaviorEventRecord,
)


class HistoryService:
    """
    Kullanıcının geçmiş oturum, mesaj ve odak geçmişi sorgularını üretir.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_user_sessions(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        sessions = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.started_at.desc())
            .limit(limit)
            .all()
        )

        if not sessions:
            return []

        session_ids = [row.session_id for row in sessions]
        reports = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id.in_(session_ids))
            .all()
        )
        report_map = {row.session_id: row for row in reports}

        message_counts = defaultdict(int)
        message_rows = (
            self.db.query(MessageRecord.session_id)
            .filter(MessageRecord.session_id.in_(session_ids))
            .all()
        )
        for row in message_rows:
            message_counts[row.session_id] += 1

        return [
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "topic": session.topic,
                "subtopic": session.subtopic,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "current_state": session.current_state,
                "average_focus_score": session.average_focus_score,
                "retry_count": session.retry_count,
                "intervention_count": session.intervention_count,
                "message_count": report_map[session.session_id].message_count
                if session.session_id in report_map
                else message_counts.get(session.session_id, 0),
                "summary_text": report_map[session.session_id].summary_text
                if session.session_id in report_map
                else None,
            }
            for session in sessions
        ]

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        messages = (
            self.db.query(MessageRecord)
            .filter(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.timestamp.asc())
            .all()
        )

        return [
            {
                "id": message.id,
                "session_id": message.session_id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "user_state": message.user_state,
                "detected_topic": message.detected_topic,
                "message_type": message.message_type,
                "llm_confidence": message.llm_confidence,
            }
            for message in messages
        ]

    def get_focus_history(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        sessions = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.started_at.desc())
            .limit(limit)
            .all()
        )

        if not sessions:
            return []

        session_ids = [row.session_id for row in sessions]
        reports = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id.in_(session_ids))
            .all()
        )
        report_map = {row.session_id: row for row in reports}

        return [
            {
                "session_id": session.session_id,
                "topic": session.topic,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "current_state": session.current_state,
                "average_focus_score": session.average_focus_score,
                "focus_score": report_map[session.session_id].focus_score
                if session.session_id in report_map
                else session.average_focus_score,
                "summary_text": report_map[session.session_id].summary_text
                if session.session_id in report_map
                else None,
                "intervention_count": session.intervention_count,
                "retry_count": session.retry_count,
            }
            for session in sessions
        ]

    def get_focus_trend(self, user_id: str, days: int = 7) -> dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(days=max(1, days))
        sessions = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.user_id == user_id, SessionRecord.started_at >= cutoff)
            .order_by(SessionRecord.started_at.asc())
            .all()
        )

        if not sessions:
            return {
                "user_id": user_id,
                "days": days,
                "total_sessions": 0,
                "average_focus_score": None,
                "trend_direction": "stable",
                "points": [],
            }

        points_by_date: dict[str, list[float]] = defaultdict(list)
        scores: list[float] = []

        for session in sessions:
            score = session.average_focus_score
            if score is None:
                continue
            key = session.started_at.date().isoformat() if session.started_at else "unknown"
            points_by_date[key].append(score)
            scores.append(score)

        points = [
            {
                "date": day,
                "focus_score": round(sum(day_scores) / len(day_scores), 3),
                "session_count": len(day_scores),
            }
            for day, day_scores in sorted(points_by_date.items())
        ]

        trend_direction = "stable"
        if len(points) >= 2:
            delta = points[-1]["focus_score"] - points[0]["focus_score"]
            if delta > 0.05:
                trend_direction = "up"
            elif delta < -0.05:
                trend_direction = "down"

        avg_focus = round(sum(scores) / len(scores), 3) if scores else None
        return {
            "user_id": user_id,
            "days": days,
            "total_sessions": len(sessions),
            "average_focus_score": avg_focus,
            "trend_direction": trend_direction,
            "points": points,
        }

    def get_last_session_report(self, user_id: str) -> tuple[Optional[SessionRecord], Optional[SessionReportRecord]]:
        session = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.user_id == user_id, SessionRecord.ended_at.isnot(None))
            .order_by(SessionRecord.ended_at.desc())
            .first()
        )

        if session is None:
            session = (
                self.db.query(SessionRecord)
                .filter(SessionRecord.user_id == user_id)
                .order_by(SessionRecord.started_at.desc())
                .first()
            )

        if session is None:
            return None, None

        report = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id == session.session_id)
            .first()
        )
        return session, report

    def get_latest_state_snapshot(self, session_id: str) -> Optional[dict[str, Any]]:
        row = (
            self.db.query(BehaviorEventRecord)
            .filter(
                BehaviorEventRecord.session_id == session_id,
                BehaviorEventRecord.event_type == "state_snapshot",
            )
            .order_by(BehaviorEventRecord.created_at.desc())
            .first()
        )
        if row is None:
            return None

        metadata = self.parse_report_dict(row.metadata_json)
        return {
            "timestamp": row.created_at.isoformat() if row.created_at else None,
            "state_after": row.state_after,
            "severity": row.severity,
            **metadata,
        }

    def get_latest_intervention(self, session_id: str) -> Optional[dict[str, Any]]:
        row = (
            self.db.query(InterventionRecord)
            .filter(InterventionRecord.session_id == session_id)
            .order_by(InterventionRecord.timestamp.desc())
            .first()
        )
        if row is None:
            return None

        return {
            "intervention_type": row.intervention_type,
            "message": row.message,
            "triggered_by": row.triggered_by,
            "reason": row.reason,
            "confidence": row.confidence,
            "was_successful": row.was_successful,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "feedback_impact": self.get_latest_feedback_impact(
                session_id=session_id,
                intervention_type=row.intervention_type,
            ),
        }

    def get_latest_feedback_impact(
        self,
        session_id: str,
        intervention_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        rows = (
            self.db.query(BehaviorEventRecord)
            .filter(
                BehaviorEventRecord.session_id == session_id,
                BehaviorEventRecord.event_type == "intervention_feedback_outcome",
            )
            .order_by(BehaviorEventRecord.created_at.desc())
            .all()
        )
        if not rows:
            return None

        for row in rows:
            metadata = self.parse_report_dict(row.metadata_json)
            if intervention_type and metadata.get("intervention_type") != intervention_type:
                continue
            return {
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                **metadata,
            }
        return None

    def parse_report_list(self, raw: Optional[str]) -> list[Any]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def parse_report_dict(self, raw: Optional[str]) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def get_focus_events_for_user(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        events = (
            self.db.query(FocusEventRecord)
            .filter(FocusEventRecord.user_id == user_id)
            .order_by(FocusEventRecord.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "session_id": event.session_id,
                "focus_score": event.focus_score,
                "source": event.source,
                "state_label": event.state_label,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in reversed(events)
        ]
