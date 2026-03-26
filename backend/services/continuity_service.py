from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from backend.services.baseline_service import BaselineService
from backend.services.history_service import HistoryService
from backend.services.intervention_policy_service import InterventionPolicyService


class ContinuityService:
    """
    Produces welcome/continuity data when a user returns.
    """

    def __init__(self, db: Session):
        self.db = db
        self.history_service = HistoryService(db)
        self.baseline_service = BaselineService(db)
        self.policy_service = InterventionPolicyService(db)

    def get_welcome(self, user_id: str) -> dict[str, Any]:
        baseline = self.baseline_service.refresh_user_baseline(user_id)
        policy = self.policy_service.get_policy_summary(user_id)
        session, report = self.history_service.get_last_session_report(user_id)

        if session is None:
            return {
                "user_id": user_id,
                "has_history": False,
                "last_session": None,
                "last_report": None,
                "last_worked_topic": None,
                "continue_suggestion": "Ilk oturumunu baslat ve kisa bir hedef belirle.",
                "continue_reason": "Henuz onceki oturum verisi yok.",
                "baseline": baseline,
                "intervention_policy": policy,
                "latest_feedback_impact": None,
                "operational_next_session_plan": None,
                "personalization_insights": [],
            }

        latest_state_analysis = self.history_service.get_latest_state_snapshot(session.session_id)
        latest_intervention = self.history_service.get_latest_intervention(session.session_id)
        latest_feedback_impact = self.history_service.get_latest_feedback_impact(session.session_id)

        last_report = None
        next_session_plan = {}
        continue_suggestion = "Bir sonraki oturuma kaldigin yerden devam edebilirsin."
        continue_reason = "Son oturum verisi mevcut."

        if report is not None:
            weaknesses = self.history_service.parse_report_list(report.weaknesses)
            recommendations = self.history_service.parse_report_list(report.recommendations)
            next_session_plan = self.history_service.parse_report_dict(report.next_session_plan)

            continue_suggestion = (
                next_session_plan.get("opening_action")
                or next_session_plan.get("goal")
                or (recommendations[0] if recommendations else continue_suggestion)
            )
            continue_reason = (
                next_session_plan.get("why_now")
                or (weaknesses[0] if weaknesses else None)
                or (recommendations[0] if recommendations else continue_reason)
            )
            last_report = {
                "summary_text": report.summary_text,
                "focus_score": report.focus_score,
                "message_count": report.message_count,
                "intervention_count": report.intervention_count,
                "retry_count": report.retry_count,
                "topics_covered": self.history_service.parse_report_list(report.topics_covered),
                "strengths": self.history_service.parse_report_list(report.strengths),
                "weaknesses": weaknesses,
                "recommendations": recommendations,
                "next_session_plan": next_session_plan,
                "created_at": report.created_at.isoformat() if report.created_at else None,
            }

        operational_plan = self._build_operational_plan(
            session=session,
            baseline=baseline,
            policy=policy,
            latest_state_analysis=latest_state_analysis,
            latest_feedback_impact=latest_feedback_impact,
            next_session_plan=next_session_plan,
        )

        return {
            "user_id": user_id,
            "has_history": True,
            "last_session": {
                "session_id": session.session_id,
                "topic": session.topic,
                "subtopic": session.subtopic,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "current_state": session.current_state,
                "average_focus_score": session.average_focus_score,
                "intervention_count": session.intervention_count,
                "retry_count": session.retry_count,
            },
            "last_report": last_report,
            "last_worked_topic": session.topic,
            "continue_suggestion": continue_suggestion,
            "continue_reason": continue_reason,
            "baseline": baseline,
            "intervention_policy": policy,
            "latest_state_analysis": latest_state_analysis,
            "latest_intervention": latest_intervention,
            "latest_feedback_impact": latest_feedback_impact,
            "operational_next_session_plan": operational_plan,
            "personalization_insights": self._build_personalization_insights(
                baseline=baseline,
                policy=policy,
                latest_state_analysis=latest_state_analysis,
                latest_feedback_impact=latest_feedback_impact,
            ),
        }

    def _build_personalization_insights(
        self,
        baseline: dict[str, Any],
        policy: dict[str, Any],
        latest_state_analysis: dict[str, Any] | None,
        latest_feedback_impact: dict[str, Any] | None,
    ) -> list[str]:
        insights: list[str] = []

        if baseline.get("avg_response_time_seconds", 0) >= 45:
            insights.append(
                "Sistem seni yavas cevap veren bir kullanici olarak kalibre ediyor; gecikme tek basina risk sayilmiyor."
            )

        if baseline.get("question_style") == "short_questions":
            insights.append(
                "Kisa soru ritmin normal kabul ediliyor; kisa mesaj ancak baseline'dan saparsa risk sinyali oluyor."
            )
        elif baseline.get("question_style") == "detailed_questions":
            insights.append(
                "Detayli soru tarzina gore kalibre edildin; ani kisa ve yuzeysel mesajlar daha guclu sinyal sayiliyor."
            )

        best_intervention = policy.get("best_intervention_type")
        if best_intervention:
            insights.append(
                f"Feedback kayitlarina gore sende en iyi calisan mudahale su an '{best_intervention}'."
            )

        reason_summary = (latest_state_analysis or {}).get("reason_summary")
        if reason_summary:
            insights.append(f"Son state karari ozeti: {reason_summary}")

        if latest_feedback_impact:
            status = latest_feedback_impact.get("measurement_status")
            intervention_type = latest_feedback_impact.get("intervention_type")
            if status == "improved" and intervention_type:
                insights.append(
                    f"Son feedback sonrasi '{intervention_type}' mudahalesi gozlenebilir iyilesme uretmis."
                )
            elif status == "worsened" and intervention_type:
                insights.append(
                    f"Son feedback sonrasi '{intervention_type}' mudahalesi yeterli olmamis; alternatif dene."
                )

        return insights[:5]

    def _build_operational_plan(
        self,
        session,
        baseline: dict[str, Any],
        policy: dict[str, Any],
        latest_state_analysis: dict[str, Any] | None,
        latest_feedback_impact: dict[str, Any] | None,
        next_session_plan: dict[str, Any],
    ) -> dict[str, Any]:
        if not next_session_plan:
            return {
                "topic": session.topic,
                "subtopic": session.subtopic,
                "start_with": "Son kaldigin noktayi 1-2 cumlede ozetle.",
                "target_outcome": "Yeni oturum hedefini netlestir.",
                "mentor_tactic": policy.get("best_intervention_type") or "question",
                "checkpoints": [],
            }

        checkpoints = list(next_session_plan.get("success_criteria") or [])
        if baseline.get("question_style") == "short_questions":
            checkpoints.insert(0, "Ilk soruyu tek satir yerine tam baglamla kur.")
        if baseline.get("avg_response_time_seconds", 0) >= 45:
            checkpoints.append("Cevaplari acele etmeden dusun; gecikme tek basina risk degil.")

        mentor_tactic = (
            (latest_feedback_impact or {}).get("intervention_type")
            if (latest_feedback_impact or {}).get("measurement_status") == "improved"
            else None
        ) or next_session_plan.get("mentor_tactic") or policy.get("best_intervention_type")

        return {
            "topic": next_session_plan.get("topic") or session.topic,
            "subtopic": next_session_plan.get("subtopic") or session.subtopic,
            "duration_minutes": next_session_plan.get("suggested_duration_minutes"),
            "start_with": next_session_plan.get("opening_action") or next_session_plan.get("goal"),
            "first_prompt": next_session_plan.get("first_prompt"),
            "target_outcome": next_session_plan.get("goal"),
            "why_now": next_session_plan.get("why_now"),
            "mentor_tactic": mentor_tactic,
            "session_structure": next_session_plan.get("session_structure", []),
            "checkpoints": checkpoints[:4],
            "risk_watchouts": next_session_plan.get("risk_watchouts", []),
            "state_carryover": (latest_state_analysis or {}).get("state_after"),
            "feedback_carryover": (latest_feedback_impact or {}).get("measurement_status"),
        }
