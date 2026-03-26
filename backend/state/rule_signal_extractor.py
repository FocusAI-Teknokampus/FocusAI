from __future__ import annotations

from backend.core.schemas import UserState


class RuleSignalExtractor:
    """
    Global kural tabanli ham state sinyallerini uretir.
    """

    def extract(self, features) -> dict[UserState, float]:
        fatigue_text_score = getattr(features, "fatigue_text_score", 0.0) or 0.0
        frustration_text_score = getattr(features, "frustration_text_score", 0.0) or 0.0
        confidence_text_score = getattr(features, "confidence_text_score", 0.0) or 0.0
        overwhelm_text_score = getattr(features, "overwhelm_text_score", 0.0) or 0.0
        confusion_score = getattr(features, "confusion_score", 0.0) or 0.0
        help_seeking_score = getattr(features, "help_seeking_score", 0.0) or 0.0
        answer_commitment_score = getattr(features, "answer_commitment_score", 0.0) or 0.0

        scores: dict[UserState, float] = {
            UserState.STUCK: 0.0,
            UserState.FATIGUED: 0.0,
            UserState.DISTRACTED: 0.0,
            UserState.FOCUSED: 0.0,
        }

        if features.retry_count >= 3:
            scores[UserState.STUCK] += 1.0
        if features.retry_count >= 5:
            scores[UserState.STUCK] += 0.5
        if features.semantic_retry_score >= 0.6:
            scores[UserState.STUCK] += 0.9
        if confusion_score >= 0.45:
            scores[UserState.STUCK] += 0.8
        if help_seeking_score >= 0.65:
            scores[UserState.STUCK] += 0.45
        if answer_commitment_score <= 0.2 and features.retry_count >= 2:
            scores[UserState.STUCK] += 0.35
        if frustration_text_score >= 0.55:
            scores[UserState.STUCK] += 0.8
        if frustration_text_score >= 0.75:
            scores[UserState.STUCK] += 0.4
        if frustration_text_score >= 0.45 and confusion_score >= 0.3:
            scores[UserState.STUCK] += 0.3
        if features.idle_time_seconds > 120:
            scores[UserState.STUCK] += 0.8
        if overwhelm_text_score >= 0.5 and confusion_score >= 0.28:
            scores[UserState.STUCK] += 0.25
        if features.idle_time_seconds > 300:
            scores[UserState.FATIGUED] += 0.8
        if fatigue_text_score >= 0.55:
            scores[UserState.FATIGUED] += 1.0
        if fatigue_text_score >= 0.75:
            scores[UserState.FATIGUED] += 0.5
        if overwhelm_text_score >= 0.55:
            scores[UserState.FATIGUED] += 0.7
        if fatigue_text_score >= 0.45 and confusion_score >= 0.35:
            scores[UserState.FATIGUED] += 0.4
        if fatigue_text_score >= 0.45 and answer_commitment_score <= 0.35:
            scores[UserState.FATIGUED] += 0.3
        if overwhelm_text_score >= 0.45 and answer_commitment_score <= 0.35:
            scores[UserState.FATIGUED] += 0.25
        if features.gaze_on_screen is False:
            scores[UserState.DISTRACTED] += 1.0
        if features.hand_on_chin is True:
            scores[UserState.DISTRACTED] += 0.7
        if features.message_length < 5 and features.idle_time_seconds > 30:
            scores[UserState.DISTRACTED] += 0.5
        if features.topic_stability <= 0.35:
            scores[UserState.DISTRACTED] += 0.6
        if features.question_density >= 0.45 and features.message_length <= 30:
            scores[UserState.DISTRACTED] += 0.4
        if help_seeking_score >= 0.7 and features.message_length <= 20:
            scores[UserState.DISTRACTED] += 0.25
        if frustration_text_score >= 0.45 and features.topic_stability <= 0.45:
            scores[UserState.DISTRACTED] += 0.25
        if (
            features.retry_count == 0
            and features.idle_time_seconds < 60
            and confusion_score < 0.2
            and fatigue_text_score < 0.25
            and frustration_text_score < 0.25
            and overwhelm_text_score < 0.25
            and help_seeking_score < 0.45
        ):
            scores[UserState.FOCUSED] += 1.0

        if features.ear_score is not None and features.ear_score < 0.20:
            scores[UserState.FATIGUED] += 1.0
        if features.ear_score is not None and features.ear_score < 0.25:
            scores[UserState.FATIGUED] += 0.5
        if features.head_tilt_angle is not None and abs(features.head_tilt_angle) > 25:
            scores[UserState.FATIGUED] += 0.6
        if features.ear_score is not None and features.ear_score >= 0.28:
            scores[UserState.FOCUSED] += 0.8
        if features.gaze_on_screen is True and features.hand_on_chin is False:
            scores[UserState.FOCUSED] += 0.7
        if (
            features.topic_stability >= 0.75
            and confusion_score < 0.2
            and fatigue_text_score < 0.25
            and frustration_text_score < 0.25
            and overwhelm_text_score < 0.25
        ):
            scores[UserState.FOCUSED] += 0.3
        if answer_commitment_score >= 0.6:
            scores[UserState.FOCUSED] += 0.45
        if confidence_text_score >= 0.55 and confusion_score < 0.3:
            scores[UserState.FOCUSED] += 0.45

        return scores
