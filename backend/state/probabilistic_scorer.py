from __future__ import annotations

import math

from backend.core.schemas import LearningPattern, UserState


class ProbabilisticScorer:
    """
    Global sinyalleri ve baseline sapmalarini birlestirip normalize eder.
    """

    def score(
        self,
        features,
        deviation_features: dict,
        rule_scores: dict[UserState, float],
    ) -> tuple[dict[UserState, float], dict[UserState, float]]:
        raw_scores = dict(rule_scores)

        retry_spike = deviation_features["retry_count"]["severity"]
        idle_spike = deviation_features["idle_time_seconds"]["severity"]
        slow_response = deviation_features["response_time_seconds"]["severity"]
        short_message = deviation_features["message_length"]["severity"]
        help_spike = deviation_features.get("help_seeking_score", {}).get("severity", 0.0)
        low_commitment = deviation_features.get("answer_commitment_score", {}).get("severity", 0.0)
        confusion_score = getattr(features, "confusion_score", 0.0) or 0.0
        semantic_retry = getattr(features, "semantic_retry_score", 0.0) or 0.0
        topic_stability = getattr(features, "topic_stability", 1.0) or 0.0
        question_density = getattr(features, "question_density", 0.0) or 0.0
        help_seeking = getattr(features, "help_seeking_score", 0.0) or 0.0
        answer_commitment = getattr(features, "answer_commitment_score", 0.0) or 0.0
        fatigue_text_score = getattr(features, "fatigue_text_score", 0.0) or 0.0

        raw_scores[UserState.STUCK] += retry_spike * 1.4
        raw_scores[UserState.STUCK] += idle_spike * 0.9
        raw_scores[UserState.STUCK] += slow_response * 0.8
        raw_scores[UserState.STUCK] += semantic_retry * 1.1
        raw_scores[UserState.STUCK] += confusion_score * 0.9
        raw_scores[UserState.STUCK] += help_seeking * 0.35
        raw_scores[UserState.STUCK] += low_commitment * 0.55

        raw_scores[UserState.FATIGUED] += idle_spike * 1.0
        raw_scores[UserState.FATIGUED] += slow_response * 0.7
        raw_scores[UserState.FATIGUED] += fatigue_text_score * 1.2
        raw_scores[UserState.FATIGUED] += max(0.0, slow_response - 0.3) * 0.4
        raw_scores[UserState.FATIGUED] += max(0.0, low_commitment - 0.2) * 0.3

        raw_scores[UserState.DISTRACTED] += short_message * 0.9
        raw_scores[UserState.DISTRACTED] += max(0.0, idle_spike - 0.2) * 0.5
        raw_scores[UserState.DISTRACTED] += max(0.0, 0.65 - topic_stability) * 1.2
        raw_scores[UserState.DISTRACTED] += question_density * 0.35 if features.message_length < 40 else 0.0
        raw_scores[UserState.DISTRACTED] += help_spike * 0.25
        if features.gaze_on_screen is False:
            raw_scores[UserState.DISTRACTED] += 0.3
        if features.hand_on_chin is True:
            raw_scores[UserState.DISTRACTED] += 0.2

        focus_penalty = max(
            retry_spike,
            idle_spike,
            short_message,
            slow_response,
            semantic_retry * 0.8,
            confusion_score * 0.7,
            help_spike * 0.5,
            low_commitment * 0.6,
            fatigue_text_score * 0.8,
        )
        if focus_penalty < 0.2 and features.retry_count == 0:
            raw_scores[UserState.FOCUSED] += 1.0
        raw_scores[UserState.FOCUSED] += max(0.0, 0.8 - focus_penalty)
        raw_scores[UserState.FOCUSED] += max(0.0, topic_stability - 0.55) * 0.4
        raw_scores[UserState.FOCUSED] += answer_commitment * 0.85
        raw_scores[UserState.FOCUSED] -= max(0.0, help_seeking - 0.55) * 0.2
        raw_scores[UserState.FOCUSED] -= fatigue_text_score * 0.6
        if features.gaze_on_screen is False:
            raw_scores[UserState.FOCUSED] -= 0.15

        probabilities = self._normalize(raw_scores)
        return raw_scores, probabilities

    def detect_learning_pattern(
        self,
        features,
        deviation_features: dict,
    ) -> LearningPattern:
        retry_spike = deviation_features["retry_count"]["severity"]
        short_message = deviation_features["message_length"]["severity"]
        slow_response = deviation_features["response_time_seconds"]["severity"]
        idle_spike = deviation_features["idle_time_seconds"]["severity"]
        question_density = getattr(features, "question_density", 0.0) or 0.0
        confusion_score = getattr(features, "confusion_score", 0.0) or 0.0
        semantic_retry = getattr(features, "semantic_retry_score", 0.0) or 0.0
        help_seeking = getattr(features, "help_seeking_score", 0.0) or 0.0
        answer_commitment = getattr(features, "answer_commitment_score", 0.0) or 0.0

        if (
            (retry_spike >= 0.6 or semantic_retry >= 0.65)
            and short_message >= 0.5
            and slow_response < 0.3
            and answer_commitment < 0.4
        ):
            return LearningPattern.SHALLOW_LEARNING
        if (
            (idle_spike >= 0.5 or answer_commitment >= 0.55)
            and (features.retry_count >= 1 or slow_response >= 0.4 or confusion_score >= 0.35)
        ):
            return LearningPattern.DEEP_ATTEMPT
        if (
            (retry_spike >= 0.8 or semantic_retry >= 0.8)
            and (short_message >= 0.6 or question_density >= 0.35 or help_seeking >= 0.6)
        ):
            return LearningPattern.MISCONCEPTION
        return LearningPattern.NORMAL

    def _normalize(self, raw_scores: dict[UserState, float]) -> dict[UserState, float]:
        positive = {state: max(score, 0.0) for state, score in raw_scores.items()}
        if sum(positive.values()) <= 0:
            uniform = round(1 / len(positive), 4)
            return {state: uniform for state in positive}

        exp_scores = {
            state: math.exp(score)
            for state, score in positive.items()
        }
        total = sum(exp_scores.values())
        return {
            state: round(score / total, 4)
            for state, score in exp_scores.items()
        }
