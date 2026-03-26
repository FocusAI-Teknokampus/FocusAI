from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    BehaviorEventRecord,
    FocusEventRecord,
    InterventionRecord,
    UserFeedbackRecord,
    UserProfileRecord,
)
from backend.core.schemas import FeedbackRequest, FeedbackResponse
from backend.services.intervention_policy_service import InterventionPolicyService
from backend.services.session_service import SessionService


class FeedbackService:
    """
    Stores user feedback and turns it into adaptive and measurable outcomes.
    """

    def __init__(self, db: Session):
        self.db = db
        self.session_service = SessionService(db)
        self.policy_service = InterventionPolicyService(db)

    def submit_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        self.session_service.get_or_create_user(payload.user_id)

        feedback = UserFeedbackRecord(
            user_id=payload.user_id,
            session_id=payload.session_id,
            message_id=payload.message_id,
            feedback_type=payload.feedback_type,
            target_type=payload.target_type,
            target_id=payload.target_id,
            intervention_type=self._normalize_intervention_type(
                payload.feedback_type,
                payload.intervention_type,
            ),
            value=payload.value,
            notes=payload.notes,
        )
        self.db.add(feedback)
        self.db.flush()

        normalized_intervention = feedback.intervention_type
        outcome = self._map_feedback_outcome(payload.feedback_type)
        adaptive_threshold = self._apply_threshold_adjustment(
            user_id=payload.user_id,
            feedback_type=payload.feedback_type,
        )

        success_rate = None
        behavior_change = None
        if normalized_intervention and outcome is not None:
            row, triggered_state = self._apply_intervention_outcome(
                user_id=payload.user_id,
                session_id=payload.session_id,
                intervention_type=normalized_intervention,
                was_successful=outcome,
            )
            self.policy_service.record_feedback(
                user_id=payload.user_id,
                intervention_type=normalized_intervention,
                triggered_state=triggered_state or "unknown",
                was_successful=outcome,
                feedback_type=payload.feedback_type,
            )
            success_rate = self.policy_service.get_success_rate(
                user_id=payload.user_id,
                intervention_type=normalized_intervention,
                triggered_state=triggered_state,
            )
            behavior_change = self._measure_behavior_change(
                feedback=feedback,
                intervention=row,
                user_marked_success=outcome,
            )

        self.db.commit()

        return FeedbackResponse(
            feedback_id=feedback.id,
            adaptive_threshold=adaptive_threshold,
            intervention_type=normalized_intervention,
            intervention_success_rate=success_rate,
            behavior_change=behavior_change,
        )

    def _apply_intervention_outcome(
        self,
        user_id: str,
        session_id: Optional[str],
        intervention_type: str,
        was_successful: bool,
    ) -> tuple[Optional[InterventionRecord], Optional[str]]:
        if not session_id:
            return None, None

        row = (
            self.db.query(InterventionRecord)
            .filter(
                InterventionRecord.user_id == user_id,
                InterventionRecord.session_id == session_id,
                InterventionRecord.intervention_type == intervention_type,
            )
            .order_by(InterventionRecord.timestamp.desc())
            .first()
        )
        if row is None:
            return None, None

        row.was_successful = was_successful
        return row, row.triggered_by

    def _measure_behavior_change(
        self,
        feedback: UserFeedbackRecord,
        intervention: Optional[InterventionRecord],
        user_marked_success: bool,
    ) -> dict[str, Any]:
        if intervention is None or not feedback.session_id:
            return {
                "measurement_status": "pending_followup",
                "user_feedback_outcome": "helpful" if user_marked_success else "not_helpful",
                "reason": "Intervention kaydi bulunamadigi icin davranis degisimi olculemedi.",
            }

        focus_before_rows = (
            self.db.query(FocusEventRecord)
            .filter(
                FocusEventRecord.session_id == feedback.session_id,
                FocusEventRecord.created_at <= intervention.timestamp,
            )
            .order_by(FocusEventRecord.created_at.desc())
            .limit(3)
            .all()
        )
        focus_after_rows = (
            self.db.query(FocusEventRecord)
            .filter(
                FocusEventRecord.session_id == feedback.session_id,
                FocusEventRecord.created_at > intervention.timestamp,
            )
            .order_by(FocusEventRecord.created_at.asc())
            .limit(3)
            .all()
        )
        snapshot_before_rows = (
            self.db.query(BehaviorEventRecord)
            .filter(
                BehaviorEventRecord.session_id == feedback.session_id,
                BehaviorEventRecord.event_type == "state_snapshot",
                BehaviorEventRecord.created_at <= intervention.timestamp,
            )
            .order_by(BehaviorEventRecord.created_at.desc())
            .limit(3)
            .all()
        )
        snapshot_after_rows = (
            self.db.query(BehaviorEventRecord)
            .filter(
                BehaviorEventRecord.session_id == feedback.session_id,
                BehaviorEventRecord.event_type == "state_snapshot",
                BehaviorEventRecord.created_at > intervention.timestamp,
            )
            .order_by(BehaviorEventRecord.created_at.asc())
            .limit(3)
            .all()
        )

        focus_before = self._average_focus(focus_before_rows)
        focus_after = self._average_focus(focus_after_rows)
        focus_delta = (
            round(focus_after - focus_before, 3)
            if focus_before is not None and focus_after is not None
            else None
        )

        before_snapshot = snapshot_before_rows[0] if snapshot_before_rows else None
        after_snapshot = snapshot_after_rows[-1] if snapshot_after_rows else None
        before_meta = self._parse_json(getattr(before_snapshot, "metadata_json", None))
        after_meta = self._parse_json(getattr(after_snapshot, "metadata_json", None))
        before_feature = self._parse_feature(before_meta)
        after_feature = self._parse_feature(after_meta)

        retry_before = before_feature.get("retry_count")
        retry_after = after_feature.get("retry_count")
        confusion_before = before_feature.get("confusion_score")
        confusion_after = after_feature.get("confusion_score")
        confidence_before = before_meta.get("confidence")
        confidence_after = after_meta.get("confidence")
        retry_delta = self._delta(retry_before, retry_after, digits=2)
        confusion_delta = self._delta(confusion_before, confusion_after, digits=3)
        confidence_delta = self._delta(confidence_before, confidence_after, digits=3)

        state_before = before_snapshot.state_after if before_snapshot else intervention.triggered_by
        state_after = after_snapshot.state_after if after_snapshot else None
        severity_before = self._state_severity(state_before)
        severity_after = self._state_severity(state_after)
        severity_delta = (
            severity_after - severity_before
            if severity_before is not None and severity_after is not None
            else None
        )

        post_timestamps = [
            row.created_at
            for row in [*(focus_after_rows or []), *(snapshot_after_rows or [])]
            if getattr(row, "created_at", None) is not None
        ]
        first_post_seconds = None
        if post_timestamps:
            first_post_seconds = round(
                min(
                    (timestamp - intervention.timestamp).total_seconds()
                    for timestamp in post_timestamps
                ),
                2,
            )

        observed_signals: list[str] = []
        if focus_delta is not None:
            if focus_delta >= 0.08:
                observed_signals.append("focus_up")
            elif focus_delta <= -0.08:
                observed_signals.append("focus_down")
        if retry_delta is not None:
            if retry_delta <= -1:
                observed_signals.append("retry_down")
            elif retry_delta >= 1:
                observed_signals.append("retry_up")
        if confusion_delta is not None:
            if confusion_delta <= -0.15:
                observed_signals.append("confusion_down")
            elif confusion_delta >= 0.15:
                observed_signals.append("confusion_up")
        if severity_delta is not None:
            if severity_delta < 0:
                observed_signals.append("state_improved")
            elif severity_delta > 0:
                observed_signals.append("state_worsened")

        if not focus_after_rows and not snapshot_after_rows:
            measurement_status = "pending_followup"
        elif any(signal in observed_signals for signal in {"focus_up", "retry_down", "confusion_down", "state_improved"}):
            measurement_status = "improved"
        elif any(signal in observed_signals for signal in {"focus_down", "retry_up", "confusion_up", "state_worsened"}):
            measurement_status = "worsened"
        else:
            measurement_status = "unchanged"

        result = {
            "measurement_status": measurement_status,
            "user_feedback_outcome": "helpful" if user_marked_success else "not_helpful",
            "feedback_matches_observed_change": (
                (user_marked_success and measurement_status in {"improved", "pending_followup"})
                or ((not user_marked_success) and measurement_status in {"worsened", "pending_followup"})
            ),
            "focus_before": focus_before,
            "focus_after": focus_after,
            "focus_delta": focus_delta,
            "retry_before": retry_before,
            "retry_after": retry_after,
            "retry_delta": retry_delta,
            "confusion_before": confusion_before,
            "confusion_after": confusion_after,
            "confusion_delta": confusion_delta,
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "confidence_delta": confidence_delta,
            "state_before": state_before,
            "state_after": state_after,
            "state_severity_delta": severity_delta,
            "post_signal_count": len(focus_after_rows) + len(snapshot_after_rows),
            "latency_to_first_post_signal_seconds": first_post_seconds,
            "observed_signals": observed_signals,
            "recommended_next_observation": self._recommended_observation(measurement_status),
        }

        behavior_event = BehaviorEventRecord(
            session_id=feedback.session_id,
            user_id=feedback.user_id,
            event_type="intervention_feedback_outcome",
            state_before=state_before,
            state_after=state_after,
            topic=None,
            severity=self._behavior_change_severity(result),
            metadata_json=json.dumps(
                {
                    "feedback_id": feedback.id,
                    "intervention_id": intervention.id,
                    "intervention_type": intervention.intervention_type,
                    "triggered_by": intervention.triggered_by,
                    **result,
                },
                ensure_ascii=False,
            ),
        )
        self.db.add(behavior_event)
        self.db.flush()
        return result

    def _apply_threshold_adjustment(self, user_id: str, feedback_type: str) -> Optional[float]:
        delta_map = {
            "correct_detection": -0.02,
            "wrong_detection": 0.03,
            "break_helpful": -0.02,
            "break_not_helpful": 0.02,
            "intervention_helpful": -0.01,
            "intervention_not_helpful": 0.02,
        }
        delta = delta_map.get(feedback_type)
        if delta is None:
            return None

        profile = (
            self.db.query(UserProfileRecord)
            .filter(UserProfileRecord.user_id == user_id)
            .first()
        )
        if profile is None:
            return None

        current = profile.adaptive_threshold or 0.75
        next_value = max(0.55, min(0.95, round(current + delta, 2)))
        profile.adaptive_threshold = next_value
        self.db.flush()
        return next_value

    def _normalize_intervention_type(
        self,
        feedback_type: str,
        intervention_type: Optional[str],
    ) -> Optional[str]:
        if intervention_type:
            return intervention_type
        if feedback_type in {"break_helpful", "break_not_helpful"}:
            return "break"
        return None

    def _map_feedback_outcome(self, feedback_type: str) -> Optional[bool]:
        mapping = {
            "correct_detection": True,
            "wrong_detection": False,
            "break_helpful": True,
            "break_not_helpful": False,
            "intervention_helpful": True,
            "intervention_not_helpful": False,
        }
        return mapping.get(feedback_type)

    def _average_focus(self, rows: list[FocusEventRecord]) -> Optional[float]:
        if not rows:
            return None
        return round(sum(row.focus_score for row in rows) / len(rows), 3)

    def _parse_json(self, raw: Optional[str]) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _parse_feature(self, metadata: dict[str, Any]) -> dict[str, Any]:
        feature_vector = metadata.get("feature_vector")
        return feature_vector if isinstance(feature_vector, dict) else {}

    def _delta(self, before: Any, after: Any, digits: int) -> Optional[float]:
        if before is None or after is None:
            return None
        return round(float(after) - float(before), digits)

    def _state_severity(self, state_label: Optional[str]) -> Optional[int]:
        if not state_label:
            return None
        severity_map = {
            "focused": 0,
            "unknown": 1,
            "distracted": 2,
            "stuck": 3,
            "fatigued": 4,
        }
        return severity_map.get(state_label)

    def _behavior_change_severity(self, result: dict[str, Any]) -> float:
        status = result.get("measurement_status")
        if status == "improved":
            return 0.25
        if status == "worsened":
            return 0.85
        if status == "unchanged":
            return 0.5
        return 0.35

    def _recommended_observation(self, measurement_status: str) -> str:
        if measurement_status == "improved":
            return "Benzer durumda ayni mudahale tipini one cikar."
        if measurement_status == "worsened":
            return "Bir sonraki benzer durumda alternatif mudahale tipi dene."
        if measurement_status == "unchanged":
            return "Ek 1-2 mesaj daha izleyip tekrar degerlendir."
        return "Yeni mesajlar geldikce before/after farkini tekrar hesapla."
