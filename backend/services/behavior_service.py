import json
from typing import Optional

from sqlalchemy.orm import Session

from backend.core.schemas import FeatureVector, StateEstimate, UserState
from backend.services.session_service import SessionService


class BehaviorService:
    """
    Persists FeatureVector and StateEstimate outputs as history events.
    """

    def __init__(self, db: Session):
        self.db = db
        self.session_service = SessionService(db)

    def persist_analysis(
        self,
        session_id: str,
        user_id: str,
        feature_vector: Optional[FeatureVector],
        state_estimate: Optional[StateEstimate],
    ) -> None:
        if feature_vector is None or state_estimate is None:
            return

        self.session_service.update_session_state(
            session_id=session_id,
            state=state_estimate.state.value,
            retry_count=feature_vector.retry_count,
        )

        focus_score = self._estimate_focus_score(state_estimate)
        has_camera_signal = any(
            value is not None
            for value in (
                feature_vector.ear_score,
                feature_vector.gaze_on_screen,
                feature_vector.hand_on_chin,
                feature_vector.head_tilt_angle,
            )
        )
        self.session_service.log_focus_event(
            session_id=session_id,
            user_id=user_id,
            focus_score=focus_score,
            source="camera_text" if has_camera_signal else "text",
            state_label=state_estimate.state.value,
        )

        events = self._infer_behavior_events(feature_vector, state_estimate)
        for event in events:
            self.session_service.log_behavior_event(
                session_id=session_id,
                user_id=user_id,
                event_type=event["event_type"],
                state_before=event.get("state_before"),
                state_after=event.get("state_after"),
                topic=event.get("topic"),
                severity=event.get("severity"),
                metadata_json=json.dumps(event.get("metadata", {}), ensure_ascii=False),
            )

        snapshot = {
            "confidence": state_estimate.confidence,
            "threshold": state_estimate.threshold,
            "decision_margin": state_estimate.decision_margin,
            "uncertainty_signal": state_estimate.uncertainty_signal,
            "learning_pattern": state_estimate.learning_pattern.value,
            "response_policy": state_estimate.response_policy.value,
            "dominant_signals": state_estimate.dominant_signals,
            "reasons": state_estimate.reasons,
            "policy_path": state_estimate.policy_path,
            "feature_vector": {
                "idle_time_seconds": feature_vector.idle_time_seconds,
                "retry_count": feature_vector.retry_count,
                "response_time_seconds": feature_vector.response_time_seconds,
                "message_length": feature_vector.message_length,
                "topic": feature_vector.topic,
                "question_density": feature_vector.question_density,
                "confusion_score": feature_vector.confusion_score,
                "topic_stability": feature_vector.topic_stability,
                "topic_confidence": feature_vector.topic_confidence,
                "semantic_retry_score": feature_vector.semantic_retry_score,
                "help_seeking_score": feature_vector.help_seeking_score,
                "help_seeking_semantic_score": feature_vector.help_seeking_semantic_score,
                "help_seeking_classifier_score": feature_vector.help_seeking_classifier_score,
                "answer_commitment_score": feature_vector.answer_commitment_score,
                "answer_commitment_semantic_score": feature_vector.answer_commitment_semantic_score,
                "answer_commitment_classifier_score": feature_vector.answer_commitment_classifier_score,
                "fatigue_text_score": feature_vector.fatigue_text_score,
                "frustration_text_score": feature_vector.frustration_text_score,
                "confidence_text_score": feature_vector.confidence_text_score,
                "overwhelm_text_score": feature_vector.overwhelm_text_score,
                "urgency_text_score": feature_vector.urgency_text_score,
                "ear_score": feature_vector.ear_score,
                "gaze_on_screen": feature_vector.gaze_on_screen,
                "hand_on_chin": feature_vector.hand_on_chin,
                "head_tilt_angle": feature_vector.head_tilt_angle,
            },
            "deviation_features": state_estimate.deviation_features,
            "state_scores": state_estimate.state_scores,
            "state_probabilities": state_estimate.state_probabilities,
            "fatigue_text_score": feature_vector.fatigue_text_score,
            "frustration_text_score": feature_vector.frustration_text_score,
            "confidence_text_score": feature_vector.confidence_text_score,
            "overwhelm_text_score": feature_vector.overwhelm_text_score,
            "urgency_text_score": feature_vector.urgency_text_score,
            "reason_summary": self._build_reason_summary(state_estimate),
        }
        self.session_service.log_behavior_event(
            session_id=session_id,
            user_id=user_id,
            event_type="state_snapshot",
            state_before=None,
            state_after=state_estimate.state.value,
            topic=feature_vector.topic,
            severity=state_estimate.confidence,
            metadata_json=json.dumps(snapshot, ensure_ascii=False),
        )

    def _estimate_focus_score(self, estimate: StateEstimate) -> float:
        base_map = {
            UserState.FOCUSED: 0.90,
            UserState.UNKNOWN: 0.50,
            UserState.STUCK: 0.40,
            UserState.DISTRACTED: 0.25,
            UserState.FATIGUED: 0.20,
        }
        base = base_map.get(estimate.state, 0.50)
        conf = estimate.confidence or 0.0
        score = base * 0.8 + conf * 0.2
        return round(max(0.0, min(1.0, score)), 3)

    def _infer_behavior_events(
        self,
        f: FeatureVector,
        estimate: StateEstimate,
    ) -> list[dict]:
        events: list[dict] = []

        if f.retry_count >= 2:
            events.append(
                {
                    "event_type": "question_repeat",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": min(1.0, 0.3 + 0.1 * f.retry_count),
                    "metadata": {
                        "retry_count": f.retry_count,
                        "message_length": f.message_length,
                    },
                }
            )

        if f.retry_count >= 3 and f.response_time_seconds < 5:
            events.append(
                {
                    "event_type": "rapid_short_questions",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": 0.75,
                    "metadata": {
                        "retry_count": f.retry_count,
                        "response_time_seconds": f.response_time_seconds,
                        "question_density": f.question_density,
                    },
                }
            )

        if f.semantic_retry_score >= 0.55:
            events.append(
                {
                    "event_type": "semantic_retry",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.semantic_retry_score), 3),
                    "metadata": {
                        "semantic_retry_score": f.semantic_retry_score,
                        "retry_count": f.retry_count,
                        "topic_confidence": f.topic_confidence,
                    },
                }
            )

        if f.confusion_score >= 0.4:
            events.append(
                {
                    "event_type": "confusion_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.confusion_score), 3),
                    "metadata": {
                        "confusion_score": f.confusion_score,
                        "question_density": f.question_density,
                    },
                }
            )

        if f.help_seeking_score >= 0.5:
            events.append(
                {
                    "event_type": "help_seeking_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.help_seeking_score), 3),
                    "metadata": {
                        "help_seeking_score": f.help_seeking_score,
                        "help_seeking_semantic_score": f.help_seeking_semantic_score,
                        "help_seeking_classifier_score": f.help_seeking_classifier_score,
                        "answer_commitment_score": f.answer_commitment_score,
                    },
                }
            )

        if f.answer_commitment_score >= 0.5:
            events.append(
                {
                    "event_type": "self_attempt_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.answer_commitment_score), 3),
                    "metadata": {
                        "answer_commitment_score": f.answer_commitment_score,
                        "answer_commitment_semantic_score": f.answer_commitment_semantic_score,
                        "answer_commitment_classifier_score": f.answer_commitment_classifier_score,
                    },
                }
            )

        if f.fatigue_text_score >= 0.45:
            match_type = "hybrid_text_match"
            if f.fatigue_text_score >= 0.75:
                match_type = "explicit_fatigue_phrase"
            elif f.fatigue_text_score < 0.58:
                match_type = "semantic_fatigue_match"

            events.append(
                {
                    "event_type": "fatigue_text_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.fatigue_text_score), 3),
                    "metadata": {
                        "fatigue_text_score": f.fatigue_text_score,
                        "match_type": match_type,
                        "confusion_score": f.confusion_score,
                        "answer_commitment_score": f.answer_commitment_score,
                    },
                }
            )

        if f.frustration_text_score >= 0.45:
            match_type = "hybrid_text_match"
            if f.frustration_text_score >= 0.75:
                match_type = "explicit_frustration_phrase"
            elif f.frustration_text_score < 0.58:
                match_type = "semantic_frustration_match"

            events.append(
                {
                    "event_type": "frustration_text_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.frustration_text_score), 3),
                    "metadata": {
                        "frustration_text_score": f.frustration_text_score,
                        "match_type": match_type,
                        "confusion_score": f.confusion_score,
                        "semantic_retry_score": f.semantic_retry_score,
                    },
                }
            )

        if f.confidence_text_score >= 0.45:
            events.append(
                {
                    "event_type": "confidence_text_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.confidence_text_score), 3),
                    "metadata": {
                        "confidence_text_score": f.confidence_text_score,
                        "answer_commitment_score": f.answer_commitment_score,
                    },
                }
            )

        if f.overwhelm_text_score >= 0.45:
            events.append(
                {
                    "event_type": "overwhelm_text_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.overwhelm_text_score), 3),
                    "metadata": {
                        "overwhelm_text_score": f.overwhelm_text_score,
                        "confusion_score": f.confusion_score,
                        "fatigue_text_score": f.fatigue_text_score,
                    },
                }
            )

        if f.urgency_text_score >= 0.45:
            events.append(
                {
                    "event_type": "urgency_text_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, f.urgency_text_score), 3),
                    "metadata": {
                        "urgency_text_score": f.urgency_text_score,
                        "help_seeking_score": f.help_seeking_score,
                    },
                }
            )

        if f.idle_time_seconds > 180:
            events.append(
                {
                    "event_type": "long_pause",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": 0.70,
                    "metadata": {
                        "idle_time_seconds": f.idle_time_seconds,
                    },
                }
            )

        if f.retry_count >= 5 and f.message_length < 30:
            events.append(
                {
                    "event_type": "same_misconception_again",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": 0.85,
                    "metadata": {
                        "retry_count": f.retry_count,
                        "message_length": f.message_length,
                    },
                }
            )

        if f.topic_stability <= 0.35:
            events.append(
                {
                    "event_type": "topic_drift",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, 1.0 - f.topic_stability), 3),
                    "metadata": {
                        "topic_stability": f.topic_stability,
                    },
                }
            )

        if f.topic:
            events.append(
                {
                    "event_type": "topic_signal",
                    "state_before": None,
                    "state_after": estimate.state.value,
                    "topic": f.topic,
                    "severity": round(min(1.0, max(0.2, f.topic_confidence)), 3),
                    "metadata": {
                        "topic_confidence": f.topic_confidence,
                    },
                }
            )

        return events

    def _build_reason_summary(self, estimate: StateEstimate) -> str:
        if estimate.reasons:
            signals = ", ".join(estimate.dominant_signals[:2]) if estimate.dominant_signals else "global sinyaller"
            return f"{estimate.reasons[0]} Baskin sinyaller: {signals}. Policy={estimate.response_policy.value}."

        deviations = estimate.deviation_features or {}
        high_signals = []

        for key, payload in deviations.items():
            severity = float((payload or {}).get("severity", 0.0))
            if severity >= 0.45:
                high_signals.append((severity, key))

        high_signals.sort(reverse=True)
        top_labels = [name for _, name in high_signals[:2]]
        if top_labels:
            readable = ", ".join(top_labels)
            return (
                f"Karari en cok kullanici baseline'ina gore sapma gosteren sinyaller etkiledi: {readable}. "
                f"Margin={estimate.decision_margin}, uncertainty={estimate.uncertainty_signal}."
            )

        state_scores = estimate.state_scores or {}
        dominant_states = sorted(
            state_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if dominant_states:
            top_state, top_score = dominant_states[0]
            return (
                f"Karar agirlikla global sinyallere dayandi; en baskin durum {top_state} "
                f"(score={top_score}). Margin={estimate.decision_margin}, "
                f"uncertainty={estimate.uncertainty_signal}."
            )

        return (
            "Karar daha cok genel kural sinyallerine dayandi. "
            f"Margin={estimate.decision_margin}, uncertainty={estimate.uncertainty_signal}."
        )
