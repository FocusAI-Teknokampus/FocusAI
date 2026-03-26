import json
from collections import Counter
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    BehaviorEventRecord,
    FocusEventRecord,
    InterventionRecord,
    MessageRecord,
    SessionRecord,
    SessionReportRecord,
)
from backend.services.history_service import HistoryService


class AnalyticsService:
    """
    Builds session reports and dashboard payloads.
    """

    def __init__(self, db: Session):
        self.db = db

    def finalize_session_report(self, session_id: str) -> Optional[SessionReportRecord]:
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
            behavior_summary=behavior_summary,
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
            behavior_summary=behavior_summary,
        )

        report = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id == session_id)
            .first()
        )
        if report is None:
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
        report.behavior_summary = json.dumps(behavior_summary, ensure_ascii=False)
        report.strengths = json.dumps(strengths, ensure_ascii=False)
        report.weaknesses = json.dumps(weaknesses, ensure_ascii=False)
        report.recommendations = json.dumps(recommendations, ensure_ascii=False)
        report.next_session_plan = json.dumps(next_session_plan, ensure_ascii=False)

        self.db.commit()
        self.db.refresh(report)
        return report

    def get_session_dashboard(self, session_id: str) -> Optional[Dict[str, Any]]:
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
        history = HistoryService(self.db)

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
                    "timestamp": row.created_at.isoformat(),
                    "focus_score": row.focus_score,
                    "source": row.source,
                    "state_label": row.state_label,
                }
                for row in focus_events
            ],
            "behavior_timeline": [
                {
                    "timestamp": row.created_at.isoformat(),
                    "event_type": row.event_type,
                    "topic": row.topic,
                    "severity": row.severity,
                    "state_before": row.state_before,
                    "state_after": row.state_after,
                }
                for row in behavior_events
            ],
            "latest_state_analysis": history.get_latest_state_snapshot(session_id),
            "latest_intervention": history.get_latest_intervention(session_id),
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

    def _count_retry_events(self, events: List[BehaviorEventRecord]) -> int:
        retry_like = {
            "question_repeat",
            "rapid_short_questions",
            "same_misconception_again",
            "semantic_retry",
        }
        return sum(1 for event in events if event.event_type in retry_like)

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
        scores = [event.focus_score for event in focus_events]
        return round(sum(scores) / len(scores), 3)

    def _build_behavior_summary(self, events: List[BehaviorEventRecord]) -> Dict[str, Any]:
        filtered_events = [
            event
            for event in events
            if event.event_type not in {"state_snapshot", "intervention_feedback_outcome"}
        ]
        feedback_events = [
            event for event in events if event.event_type == "intervention_feedback_outcome"
        ]
        latest_feedback_impact = self._extract_latest_feedback_impact(feedback_events)

        if not filtered_events:
            return {
                "dominant_behavior": None,
                "event_counts": {},
                "high_severity_events": 0,
                "feedback_outcome_counts": self._feedback_outcome_counts(feedback_events),
                "latest_feedback_impact": latest_feedback_impact,
            }

        event_counts = Counter(event.event_type for event in filtered_events)
        dominant_behavior = event_counts.most_common(1)[0][0]
        high_severity_events = sum(1 for event in filtered_events if (event.severity or 0) >= 0.7)

        return {
            "dominant_behavior": dominant_behavior,
            "event_counts": dict(event_counts),
            "high_severity_events": high_severity_events,
            "feedback_outcome_counts": self._feedback_outcome_counts(feedback_events),
            "latest_feedback_impact": latest_feedback_impact,
        }

    def _infer_strengths(
        self,
        messages: List[MessageRecord],
        behavior_events: List[BehaviorEventRecord],
    ) -> List[str]:
        strengths: List[str] = []

        recovery_count = sum(1 for event in behavior_events if event.event_type == "recovery_after_hint")
        independent_work = sum(1 for event in behavior_events if event.event_type == "worked_without_help")
        improved_after_feedback = sum(
            1
            for event in behavior_events
            if event.event_type == "intervention_feedback_outcome"
            and self._safe_load_dict(event.metadata_json).get("measurement_status") == "improved"
        )

        if recovery_count > 0:
            strengths.append("Ipucu sonrasi toparlanma gosterebildi.")
        if independent_work > 0:
            strengths.append("Bazi noktalarda yardim almadan ilerleyebildi.")
        if improved_after_feedback > 0:
            strengths.append("Feedback sonrasi gozlenebilir toparlanma sinyali olustu.")
        if len(messages) >= 6:
            strengths.append("Oturum boyunca etkilesimi surdurebildi.")

        if not strengths:
            strengths.append("Calisma oturumunu tamamladi.")
        return strengths

    def _infer_weaknesses(self, behavior_events: List[BehaviorEventRecord]) -> List[str]:
        weaknesses: List[str] = []

        repeat_count = sum(1 for event in behavior_events if event.event_type == "question_repeat")
        misconception_count = sum(1 for event in behavior_events if event.event_type == "same_misconception_again")
        pause_count = sum(1 for event in behavior_events if event.event_type == "long_pause")
        confusion_count = sum(1 for event in behavior_events if event.event_type == "confusion_signal")
        drift_count = sum(1 for event in behavior_events if event.event_type == "topic_drift")
        worsened_after_feedback = sum(
            1
            for event in behavior_events
            if event.event_type == "intervention_feedback_outcome"
            and self._safe_load_dict(event.metadata_json).get("measurement_status") == "worsened"
        )

        if repeat_count >= 2:
            weaknesses.append("Ayni tip soruya tekrar donme egilimi goruldu.")
        if misconception_count >= 1:
            weaknesses.append("Benzer yanlis anlama kalibi tekrar etti.")
        if pause_count >= 2:
            weaknesses.append("Uzun duraksamalar odak kaybina isaret ediyor olabilir.")
        if confusion_count >= 2:
            weaknesses.append("Karisiklik sinyalleri arka arkaya geldi; kavram temeli zayif olabilir.")
        if drift_count >= 1:
            weaknesses.append("Konu surekliligi bozuldu; birden fazla alt probleme dagilma var.")
        if worsened_after_feedback >= 1:
            weaknesses.append("Son mudahale sonrasi davranis metrikleri iyilesmedi; farkli bir destek tipi gerekebilir.")
        return weaknesses

    def _build_recommendations(
        self,
        behavior_summary: Dict[str, Any],
        focus_score: Optional[float],
        weaknesses: List[str],
    ) -> List[str]:
        recommendations: List[str] = []

        dominant = behavior_summary.get("dominant_behavior")
        latest_feedback_impact = behavior_summary.get("latest_feedback_impact") or {}
        latest_feedback_status = latest_feedback_impact.get("measurement_status")
        latest_intervention_type = latest_feedback_impact.get("intervention_type")

        if dominant == "rapid_short_questions":
            recommendations.append("Bir sonraki oturumda daha az ama daha derin soru sormayi dene.")
        if dominant == "semantic_retry":
            recommendations.append("Yeni soruya gecmeden once ayni problemi tek cumleyle yeniden cercevele.")
        if dominant == "same_misconception_again":
            recommendations.append("Temel kavrama donup kisa bir ozet tekrar yap.")
        if dominant == "long_pause":
            recommendations.append("Daha kisa calisma bloklari ve planli molalar faydali olabilir.")
        if dominant == "topic_drift":
            recommendations.append("Tek oturumda tek alt hedefe bagli kal; yan sorulari sona park et.")
        if focus_score is not None and focus_score < 0.5:
            recommendations.append("Odak seviyesi dusuk oldugu icin 15-20 dakikalik mikro calisma bloklari onerilir.")
        if latest_feedback_status == "improved" and latest_intervention_type:
            recommendations.append(
                f"Son feedback'e gore '{latest_intervention_type}' mudahalesi ise yaradi; benzer durumda tekrar kullan."
            )
        if latest_feedback_status == "worsened" and latest_intervention_type:
            recommendations.append(
                f"Son feedback'e gore '{latest_intervention_type}' yeterli degildi; alternatif mudahale tipi dene."
            )
        if not recommendations and weaknesses:
            recommendations.append("Zorlanilan kavramlari ornek sorularla tekrar et.")
        if not recommendations:
            recommendations.append("Benzer tempoda devam edilebilir.")
        return recommendations

    def _build_next_session_plan(
        self,
        session: SessionRecord,
        focus_score: Optional[float],
        behavior_summary: Dict[str, Any],
        weaknesses: List[str],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        suggested_minutes = 25
        if focus_score is not None and focus_score < 0.5:
            suggested_minutes = 15
        elif focus_score is not None and focus_score > 0.75:
            suggested_minutes = 30

        dominant_behavior = behavior_summary.get("dominant_behavior")
        latest_feedback_impact = behavior_summary.get("latest_feedback_impact") or {}
        feedback_status = latest_feedback_impact.get("measurement_status")
        preferred_intervention = latest_feedback_impact.get("intervention_type")

        if dominant_behavior in {"long_pause", "topic_drift"} and suggested_minutes > 20:
            suggested_minutes = 20

        if dominant_behavior == "same_misconception_again":
            goal = "Tekrarlayan yanlis modeli tek bir temsil uzerinden duzeltmek."
            opening_action = "Bir onceki yanlis adimi 2 cumlede yaz ve hangi noktada koptugunu isaretle."
        elif dominant_behavior in {"rapid_short_questions", "semantic_retry"}:
            goal = "Soru-kirma dongusunu azaltip problemi tek tam baglamla ifade etmek."
            opening_action = "Bugun cozecegin tek problemi sec ve once tek mesajda tam baglamini yaz."
        elif dominant_behavior == "long_pause" or (focus_score is not None and focus_score < 0.5):
            goal = "Kisa bloklarla odagi koruyup tek alt probleme ilerlemek."
            opening_action = "5 dakikalik mini hedef yaz ve ilk ornege hemen basla."
        else:
            goal = "Kaldigin alt basligi ornek cozerek istikrarli sekilde ilerletmek."
            opening_action = "Son oturumdan kalan alt basligi 1 cumleyle hatirlat ve ilk ornegi ac."

        why_now = weaknesses[0] if weaknesses else "Son oturumdaki sinyaller takip aksiyonu gerektiriyor."
        mentor_tactic = preferred_intervention or self._default_mentor_tactic(dominant_behavior, feedback_status)
        success_criteria = self._build_success_criteria(weaknesses, focus_score)
        risk_watchouts = self._build_risk_watchouts(dominant_behavior, weaknesses, feedback_status)
        core_minutes = max(8, suggested_minutes - 10)

        return {
            "topic": session.topic,
            "subtopic": session.subtopic,
            "suggested_duration_minutes": suggested_minutes,
            "goal": goal,
            "why_now": why_now,
            "opening_action": opening_action,
            "first_prompt": self._build_first_prompt(session, dominant_behavior, weaknesses),
            "mentor_tactic": mentor_tactic,
            "session_structure": [
                {
                    "step": "Warm start",
                    "minutes": 5,
                    "action": opening_action,
                    "success_signal": "Kullanici hedefi ve zorlandigi noktayi netlestirir.",
                },
                {
                    "step": "Core drill",
                    "minutes": core_minutes,
                    "action": recommendations[0] if recommendations else "Tek kavrama odakli ornek coz.",
                    "success_signal": success_criteria[0] if success_criteria else "Ayni hata tekrar etmez.",
                },
                {
                    "step": "Close out",
                    "minutes": suggested_minutes - 5 - core_minutes,
                    "action": "Oturum sonunda 2 maddelik ozet ve bir sonraki soru listesi cikar.",
                    "success_signal": "Bir sonraki oturuma devredilecek net not olusur.",
                },
            ],
            "priority_weaknesses": weaknesses[:2],
            "recommended_actions": recommendations[:3],
            "success_criteria": success_criteria,
            "risk_watchouts": risk_watchouts,
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
        behavior_summary: Dict[str, Any],
    ) -> str:
        topic = session.topic or "genel konu"
        focus_str = f"{focus_score:.2f}" if focus_score is not None else "olculemedi"
        text = (
            f"Bu oturumda '{topic}' konusu uzerinde calisildi. "
            f"Toplam {message_count} mesajlik etkilesim gerceklesti ve "
            f"{intervention_count} mentor mudahalesi yapildi. "
            f"Ortalama odak skoru: {focus_str}. "
        )
        if strengths:
            text += f"Guclu taraflar: {', '.join(strengths)}. "
        if weaknesses:
            text += f"Gelistirilmesi gereken alanlar: {', '.join(weaknesses)}. "

        latest_feedback_impact = behavior_summary.get("latest_feedback_impact") or {}
        if latest_feedback_impact.get("measurement_status") == "improved":
            text += "Son feedback sonrasi olumlu davranis degisimi izlendi. "
        elif latest_feedback_impact.get("measurement_status") == "worsened":
            text += "Son feedback sonrasi beklenen toparlanma gorulmedi. "

        if recommendations:
            text += f"Oneri: {recommendations[0]}"
        return text.strip()

    def _extract_latest_feedback_impact(
        self,
        feedback_events: List[BehaviorEventRecord],
    ) -> Optional[Dict[str, Any]]:
        if not feedback_events:
            return None
        latest = max(feedback_events, key=lambda event: event.created_at)
        payload = self._safe_load_dict(latest.metadata_json)
        if not payload:
            return None
        return {
            "timestamp": latest.created_at.isoformat() if latest.created_at else None,
            **payload,
        }

    def _feedback_outcome_counts(self, feedback_events: List[BehaviorEventRecord]) -> Dict[str, int]:
        counts = Counter()
        for event in feedback_events:
            status = self._safe_load_dict(event.metadata_json).get("measurement_status")
            if status:
                counts[status] += 1
        return dict(counts)

    def _default_mentor_tactic(
        self,
        dominant_behavior: Optional[str],
        feedback_status: Optional[str],
    ) -> str:
        if feedback_status == "worsened":
            return "question"
        if dominant_behavior in {"same_misconception_again", "semantic_retry"}:
            return "hint"
        if dominant_behavior in {"long_pause", "topic_drift"}:
            return "strategy"
        return "question"

    def _build_first_prompt(
        self,
        session: SessionRecord,
        dominant_behavior: Optional[str],
        weaknesses: List[str],
    ) -> str:
        topic = session.topic or "bugunku konu"
        if dominant_behavior == "same_misconception_again":
            return f"{topic} icin once en cok karistigin adimi yaz; sonra tek bir ornek uzerinden ilerleyelim."
        if dominant_behavior in {"rapid_short_questions", "semantic_retry"}:
            return f"{topic} icin bugun tek bir soruya odaklan. Tum baglami tek mesajda yaz, parcali sorma."
        if dominant_behavior == "long_pause":
            return f"{topic} oturumunu 15 dakikalik tek blok olarak ac; ilk 5 dakikada sadece baslangic adimini tamamla."
        if weaknesses:
            return f"{topic} oturumunu '{weaknesses[0]}' noktasini duzeltecek sekilde baslatalim."
        return f"{topic} icin kaldigin yerden devam et ve ilk ornegi sec."

    def _build_success_criteria(
        self,
        weaknesses: List[str],
        focus_score: Optional[float],
    ) -> List[str]:
        criteria: List[str] = []
        if any("tekrar" in item.lower() for item in weaknesses):
            criteria.append("Ayni soru tipi arka arkaya tekrar edilmez.")
        if any("duraksama" in item.lower() for item in weaknesses):
            criteria.append("10 dakika icinde aktif ilerleme surer, uzun sessizlik olmaz.")
        if any("karisiklik" in item.lower() for item in weaknesses):
            criteria.append("Kullanici kavrami kendi cumlesiyle dogru ozetler.")
        if focus_score is not None and focus_score < 0.5:
            criteria.append("Oturum sonunda odak ritmi korunur ve mikro blok tamamlanir.")
        if not criteria:
            criteria.append("Bir alt hedef tamamlanir ve sonraki soru netlesir.")
        return criteria[:3]

    def _build_risk_watchouts(
        self,
        dominant_behavior: Optional[str],
        weaknesses: List[str],
        feedback_status: Optional[str],
    ) -> List[str]:
        watchouts: List[str] = []
        if dominant_behavior == "topic_drift":
            watchouts.append("Yan sorular ana akisi bozabilir; parking lot listesi tut.")
        if dominant_behavior == "long_pause":
            watchouts.append("Sessizlik uzarsa oturumu daha kucuk alt gorevlere bol.")
        if dominant_behavior in {"rapid_short_questions", "semantic_retry"}:
            watchouts.append("Eksik baglamli tek satir sorular tekrar retry dongusu yaratabilir.")
        if feedback_status == "worsened":
            watchouts.append("Son kullanilan mudahale tipini varsayilan secenek yapma.")
        if not watchouts and weaknesses:
            watchouts.append("Zayif alan ortaya cikarsa tempo dusurulup tek ornege donulmeli.")
        return watchouts[:3]

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
