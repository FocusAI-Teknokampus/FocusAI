from __future__ import annotations

from backend.core.schemas import UserState


class RuleSignalExtractor:
    """
    Global kural tabanli ham state sinyallerini uretir.
    """

    def extract(self, features) -> dict[UserState, float]:
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
        if features.confusion_score >= 0.45:
            scores[UserState.STUCK] += 0.8
        if features.idle_time_seconds > 120:
            scores[UserState.STUCK] += 0.8
        if features.idle_time_seconds > 300:
            scores[UserState.FATIGUED] += 0.8
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
        if features.retry_count == 0 and features.idle_time_seconds < 60:
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
        if features.topic_stability >= 0.75 and features.confusion_score < 0.2:
            scores[UserState.FOCUSED] += 0.3

        return scores
