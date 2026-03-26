from __future__ import annotations

from typing import Optional

from backend.core.schemas import FeatureVector, ResponsePolicyMode, StateEstimate, UserProfile, UserState


class ResponsePolicyService:
    """
    State tahminini cevap moduna cevirir ve aciklanabilir sinyaller uretir.
    """

    def decide(
        self,
        estimate: StateEstimate,
        feature_vector: FeatureVector,
        user_profile: Optional[UserProfile] = None,
        baseline_profile: Optional[dict] = None,
    ) -> dict:
        dominant_signals = self._dominant_signals(estimate, feature_vector)
        reasons = self._reasons(estimate, feature_vector, dominant_signals)
        mode = self._select_mode(
            estimate=estimate,
            feature_vector=feature_vector,
            user_profile=user_profile,
            baseline_profile=baseline_profile or {},
        )
        policy_path = [
            f"state:{estimate.state.value}",
            f"confidence:{round(estimate.confidence, 2)}",
            f"policy:{mode.value}",
            f"delivery:{'clarify_node' if mode == ResponsePolicyMode.CLARIFY else 'mentor_node'}",
        ]
        return {
            "response_policy": mode,
            "dominant_signals": dominant_signals,
            "reasons": reasons,
            "policy_path": policy_path,
        }

    def _select_mode(
        self,
        estimate: StateEstimate,
        feature_vector: FeatureVector,
        user_profile: Optional[UserProfile],
        baseline_profile: dict,
    ) -> ResponsePolicyMode:
        work_style = baseline_profile.get("work_style", {}) if baseline_profile else {}
        challenge_tolerance = (
            user_profile.challenge_tolerance
            if user_profile
            else work_style.get("challenge_tolerance", 0.5)
        ) or 0.5
        intervention_sensitivity = (
            user_profile.intervention_sensitivity
            if user_profile
            else work_style.get("intervention_sensitivity", 0.5)
        ) or 0.5
        prefers_hint_first = bool(
            user_profile.prefers_hint_first if user_profile else work_style.get("prefers_hint_first", False)
        )
        prefers_direct_explanation = bool(
            user_profile.prefers_direct_explanation
            if user_profile
            else work_style.get("prefers_direct_explanation", False)
        )

        if estimate.confidence < estimate.threshold or estimate.uncertainty_signal >= 0.6:
            return ResponsePolicyMode.CLARIFY

        if estimate.state == UserState.FATIGUED:
            return ResponsePolicyMode.RECOVERY

        if estimate.state == UserState.DISTRACTED:
            if feature_vector.topic_stability <= 0.4 or feature_vector.idle_time_seconds >= 150:
                return ResponsePolicyMode.RECOVERY
            if challenge_tolerance >= 0.72 and feature_vector.answer_commitment_score >= 0.6:
                return ResponsePolicyMode.CHALLENGE
            return ResponsePolicyMode.CLARIFY if intervention_sensitivity >= 0.75 else ResponsePolicyMode.RECOVERY

        if estimate.state == UserState.STUCK:
            if prefers_direct_explanation and feature_vector.help_seeking_score >= 0.55:
                return ResponsePolicyMode.DIRECT_HELP
            if feature_vector.answer_commitment_score >= 0.5 or prefers_hint_first:
                return ResponsePolicyMode.GUIDED_HINT
            if challenge_tolerance >= 0.75 and feature_vector.confusion_score < 0.45:
                return ResponsePolicyMode.CHALLENGE
            return ResponsePolicyMode.DIRECT_HELP if feature_vector.help_seeking_score >= 0.65 else ResponsePolicyMode.GUIDED_HINT

        if estimate.state == UserState.FOCUSED:
            if (challenge_tolerance >= 0.65 or feature_vector.answer_commitment_score >= 0.65) and feature_vector.answer_commitment_score >= 0.55:
                return ResponsePolicyMode.CHALLENGE
            if feature_vector.help_seeking_score >= 0.6 and feature_vector.answer_commitment_score < 0.35:
                return ResponsePolicyMode.DIRECT_HELP
            return ResponsePolicyMode.GUIDED_HINT if prefers_hint_first else ResponsePolicyMode.DIRECT_HELP

        question_style = baseline_profile.get("question_style")
        if question_style == "short_questions" and feature_vector.message_length <= 25:
            return ResponsePolicyMode.DIRECT_HELP
        return ResponsePolicyMode.CLARIFY

    def _dominant_signals(self, estimate: StateEstimate, feature_vector: FeatureVector) -> list[str]:
        signals: list[tuple[float, str]] = []
        deviations = estimate.deviation_features or {}

        for key, label in [
            ("retry_count", "retry spike"),
            ("idle_time_seconds", "long idle gap"),
            ("response_time_seconds", "slow response"),
            ("message_length", "short message"),
            ("help_seeking_score", "high help seeking"),
            ("answer_commitment_score", "low answer commitment"),
        ]:
            severity = float((deviations.get(key) or {}).get("severity", 0.0))
            if severity >= 0.35:
                signals.append((severity, label))

        if feature_vector.semantic_retry_score >= 0.55:
            signals.append((feature_vector.semantic_retry_score, "high semantic retry"))
        if feature_vector.topic_stability <= 0.45:
            signals.append((1.0 - feature_vector.topic_stability, "low topic stability"))
        if feature_vector.confusion_score >= 0.4:
            signals.append((feature_vector.confusion_score, "high confusion"))
        if feature_vector.answer_commitment_score >= 0.55:
            signals.append((feature_vector.answer_commitment_score, "strong answer commitment"))

        signals.sort(key=lambda item: item[0], reverse=True)
        return [label for _, label in signals[:3]]

    def _reasons(
        self,
        estimate: StateEstimate,
        feature_vector: FeatureVector,
        dominant_signals: list[str],
    ) -> list[str]:
        reasons: list[str] = []

        if estimate.confidence < estimate.threshold:
            reasons.append(
                "State tahmini kullanici esiginin altinda kaldigi icin once netlestirme gerekiyor."
            )

        if feature_vector.semantic_retry_score >= 0.6 and feature_vector.topic_stability >= 0.5:
            reasons.append(
                "Son mesajlar ayni kavram cevresinde farkli ifadelerle tekrar ediyor; kullanici ayni yere geri donuyor."
            )

        if feature_vector.help_seeking_score >= 0.55 and feature_vector.answer_commitment_score <= 0.3:
            reasons.append(
                "Mesaj dogrudan yardim talebine kayiyor; kendi cozum denemesi sinyali zayif."
            )

        if feature_vector.answer_commitment_score >= 0.5 and feature_vector.confusion_score >= 0.35:
            reasons.append(
                "Kullanici kendi denemesini surduruyor ama kavrami netlestirmekte zorlaniyor."
            )

        if feature_vector.topic_stability <= 0.35:
            reasons.append(
                "Mesajlar alt problemler arasinda sicrıyor; konu surekliligi zayif."
            )

        if not reasons and dominant_signals:
            reasons.append(
                f"Karari en cok su sinyaller etkiledi: {', '.join(dominant_signals)}."
            )

        if not reasons:
            reasons.append("Global state olasiliklari bu mesaji dogrudan yardim akisina yakin buldu.")

        return reasons[:3]
