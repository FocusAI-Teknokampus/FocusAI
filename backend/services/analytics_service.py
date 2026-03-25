# backend/services/analytics_service.py

import json
from collections import Counter
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    SessionRecord,
    MessageRecord,
    InterventionRecord,
    BehaviorEventRecord,
    FocusEventRecord,
    SessionReportRecord,
)


class AnalyticsService:
    """
    Session raporu ve dashboard analitiği üreten servis.
    """

    def __init__(self, db: Session):
        self.db = db

    # ============================================================
    # SESSION REPORT
    # ============================================================

    def finalize_session_report(self, session_id: str) -> Optional[SessionReportRecord]:
        """
        Verilen session için rapor üretir veya günceller.
        """
        session = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.session_id == session_id)
            .first()
        )
        if not session:
            return None

        messages = (
            self.db.query(MessageRecord)
            .filter(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.timestamp.asc())
            .all()
        )

        interventions = (
            self.db.query(InterventionRecord)
            .filter(InterventionRecord.session_id == session_id)
            .all()
        )

        behavior_events = (
            self.db.query(BehaviorEventRecord)
            .filter(BehaviorEventRecord.session_id == session_id)
            .all()
        )

        focus_events = (
            self.db.query(FocusEventRecord)
            .filter(FocusEventRecord.session_id == session_id)
            .order_by(FocusEventRecord.created_at.asc())
            .all()
        )

        message_count = len(messages)
        intervention_count = len(interventions)
        retry_count = self._count_retry_events(behavior_events)
        topics_covered = self._extract_topics(messages, session.topic)
        focus_score = self._average_focus(focus_events)

        behavior_summary = self._build_behavior_summary(behavior_events)
        strengths = self._infer_strengths(messages, behavior_events)
        weaknesses = self._infer_weaknesses(behavior_events)
        recommendations = self._build_recommendations(
            behavior_summary=behavior_summary,
            focus_score=focus_score,
            weaknesses=weaknesses,
        )
        next_session_plan = self._build_next_session_plan(
            session=session,
            focus_score=focus_score,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )
        summary_text = self._build_summary_text(
            session=session,
            message_count=message_count,
            intervention_count=intervention_count,
            focus_score=focus_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

        existing = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id == session_id)
            .first()
        )

        if existing:
            report = existing
        else:
            report = SessionReportRecord(
                session_id=session.session_id,
                user_id=session.user_id,
            )
            self.db.add(report)

        report.topic = session.topic
        report.message_count = message_count
        report.intervention_count = intervention_count
        report.retry_count = retry_count
        report.topics_covered = json.dumps(topics_covered, ensure_ascii=False)
        report.focus_score = focus_score
        report.summary_text = summary_text

        # Yeni: zengin alanları da kaydet
        report.behavior_summary = json.dumps(behavior_summary, ensure_ascii=False)
        report.strengths = json.dumps(strengths, ensure_ascii=False)
        report.weaknesses = json.dumps(weaknesses, ensure_ascii=False)
        report.recommendations = json.dumps(recommendations, ensure_ascii=False)
        report.next_session_plan = json.dumps(next_session_plan, ensure_ascii=False)

        self.db.commit()
        self.db.refresh(report)
        return report

    # ============================================================
    # DASHBOARD
    # ============================================================

    def get_session_dashboard(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Session dashboard verisini döner.
        """
        session = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.session_id == session_id)
            .first()
        )
        if not session:
            return None

        report = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id == session_id)
            .first()
        )

        focus_events = (
            self.db.query(FocusEventRecord)
            .filter(FocusEventRecord.session_id == session_id)
            .order_by(FocusEventRecord.created_at.asc())
            .all()
        )

        behavior_events = (
            self.db.query(BehaviorEventRecord)
            .filter(BehaviorEventRecord.session_id == session_id)
            .order_by(BehaviorEventRecord.created_at.asc())
            .all()
        )

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "topic": session.topic,
            "subtopic": session.subtopic,
            "study_mode": session.study_mode,
            "camera_used": session.camera_used,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "current_state": session.current_state,
            "average_focus_score": session.average_focus_score,
            "retry_count": session.retry_count,
            "intervention_count": session.intervention_count,
            "focus_timeline": [
                {
                    "timestamp": e.created_at.isoformat(),
                    "focus_score": e.focus_score,
                    "source": e.source,
                    "state_label": e.state_label,
                }
                for e in focus_events
            ],
            "behavior_timeline": [
                {
                    "timestamp": e.created_at.isoformat(),
                    "event_type": e.event_type,
                    "topic": e.topic,
                    "severity": e.severity,
                    "state_before": e.state_before,
                    "state_after": e.state_after,
                }
                for e in behavior_events
            ],
            "report": {
                "message_count": report.message_count if report else 0,
                "intervention_count": report.intervention_count if report else 0,
                "retry_count": report.retry_count if report else 0,
                "topics_covered": self._safe_load_list(report.topics_covered) if report else [],
                "focus_score": report.focus_score if report else None,
                "summary_text": report.summary_text if report else None,
                "behavior_summary": self._safe_load_dict(report.behavior_summary) if report else {},
                "strengths": self._safe_load_list(report.strengths) if report else [],
                "weaknesses": self._safe_load_list(report.weaknesses) if report else [],
                "recommendations": self._safe_load_list(report.recommendations) if report else [],
                "next_session_plan": self._safe_load_dict(report.next_session_plan) if report else {},
            },
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def _count_retry_events(self, events: List[BehaviorEventRecord]) -> int:
        retry_like = {
            "question_repeat",
            "rapid_short_questions",
            "same_misconception_again",
        }
        return sum(1 for e in events if e.event_type in retry_like)

    def _extract_topics(
        self,
        messages: List[MessageRecord],
        fallback_topic: Optional[str],
    ) -> List[str]:
        topics = []

        if fallback_topic:
            topics.append(fallback_topic)

        for msg in messages:
            if msg.detected_topic:
                topics.append(msg.detected_topic)

        unique_topics = []
        seen = set()

        for topic in topics:
            normalized = topic.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_topics.append(topic)

        return unique_topics

    def _average_focus(self, focus_events: List[FocusEventRecord]) -> Optional[float]:
        if not focus_events:
            return None
        scores = [e.focus_score for e in focus_events]
        return round(sum(scores) / len(scores), 3)

    def _build_behavior_summary(self, events: List[BehaviorEventRecord]) -> Dict[str, Any]:
        if not events:
            return {
                "dominant_behavior": None,
                "event_counts": {},
                "high_severity_events": 0,
            }

        event_counts = Counter([e.event_type for e in events])
        dominant_behavior = event_counts.most_common(1)[0][0]
        high_severity_events = sum(1 for e in events if (e.severity or 0) >= 0.7)

        return {
            "dominant_behavior": dominant_behavior,
            "event_counts": dict(event_counts),
            "high_severity_events": high_severity_events,
        }

    def _infer_strengths(
        self,
        messages: List[MessageRecord],
        behavior_events: List[BehaviorEventRecord],
    ) -> List[str]:
        strengths = []

        recovery_count = sum(1 for e in behavior_events if e.event_type == "recovery_after_hint")
        independent_work = sum(1 for e in behavior_events if e.event_type == "worked_without_help")

        if recovery_count > 0:
            strengths.append("İpucu sonrası toparlanma gösterebildi.")
        if independent_work > 0:
            strengths.append("Bazı noktalarda yardım almadan ilerleyebildi.")
        if len(messages) >= 6:
            strengths.append("Oturum boyunca etkileşimi sürdürebildi.")

        if not strengths:
            strengths.append("Çalışma oturumunu tamamladı.")

        return strengths

    def _infer_weaknesses(self, behavior_events: List[BehaviorEventRecord]) -> List[str]:
        weaknesses = []

        repeat_count = sum(1 for e in behavior_events if e.event_type == "question_repeat")
        misconception_count = sum(1 for e in behavior_events if e.event_type == "same_misconception_again")
        pause_count = sum(1 for e in behavior_events if e.event_type == "long_pause")

        if repeat_count >= 2:
            weaknesses.append("Aynı tip soruya tekrar dönme eğilimi görüldü.")
        if misconception_count >= 1:
            weaknesses.append("Benzer yanlış anlama kalıbı tekrar etti.")
        if pause_count >= 2:
            weaknesses.append("Uzun duraksamalar odak kaybına işaret ediyor olabilir.")

        return weaknesses

    def _build_recommendations(
        self,
        behavior_summary: Dict[str, Any],
        focus_score: Optional[float],
        weaknesses: List[str],
    ) -> List[str]:
        recommendations = []

        dominant = behavior_summary.get("dominant_behavior")

        if dominant == "rapid_short_questions":
            recommendations.append("Bir sonraki oturumda daha az ama daha derin soru sormayı dene.")
        if dominant == "same_misconception_again":
            recommendations.append("Temel kavrama dönüp kısa bir özet tekrar yap.")
        if dominant == "long_pause":
            recommendations.append("Daha kısa çalışma blokları ve planlı molalar faydalı olabilir.")

        if focus_score is not None and focus_score < 0.5:
            recommendations.append("Odak seviyesi düşük olduğu için 15-20 dakikalık mikro çalışma blokları önerilir.")

        if not recommendations and weaknesses:
            recommendations.append("Zorlanılan kavramları örnek sorularla tekrar et.")
        if not recommendations:
            recommendations.append("Benzer tempoda devam edilebilir.")

        return recommendations

    def _build_next_session_plan(
        self,
        session: SessionRecord,
        focus_score: Optional[float],
        weaknesses: List[str],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        suggested_minutes = 25

        if focus_score is not None and focus_score < 0.5:
            suggested_minutes = 15
        elif focus_score is not None and focus_score > 0.75:
            suggested_minutes = 30

        return {
            "topic": session.topic,
            "subtopic": session.subtopic,
            "suggested_duration_minutes": suggested_minutes,
            "goal": "Zorlanılan kavramları kısa tekrar ve örnek çözümle pekiştirmek.",
            "priority_weaknesses": weaknesses[:2],
            "recommended_actions": recommendations[:3],
        }

    def _build_summary_text(
        self,
        session: SessionRecord,
        message_count: int,
        intervention_count: int,
        focus_score: Optional[float],
        strengths: List[str],
        weaknesses: List[str],
        recommendations: List[str],
    ) -> str:
        topic = session.topic or "genel konu"
        focus_str = f"{focus_score:.2f}" if focus_score is not None else "ölçülemedi"

        text = (
            f"Bu oturumda '{topic}' konusu üzerinde çalışıldı. "
            f"Toplam {message_count} mesajlık etkileşim gerçekleşti ve "
            f"{intervention_count} mentor müdahalesi yapıldı. "
            f"Ortalama odak skoru: {focus_str}. "
        )

        if strengths:
            text += f"Güçlü taraflar: {', '.join(strengths)}. "
        if weaknesses:
            text += f"Geliştirilmesi gereken alanlar: {', '.join(weaknesses)}. "
        if recommendations:
            text += f"Öneri: {recommendations[0]}"

        return text.strip()

    def _safe_load_list(self, raw: Optional[str]) -> List[Any]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else []
        except Exception:
            return []

    def _safe_load_dict(self, raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}