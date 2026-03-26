from __future__ import annotations

from backend.core.schemas import UserState


class DecisionController:
    """
    Olasilik dagilimindan final state, confidence ve belirsizlik cikarir.
    """

    def select(
        self,
        state_probabilities: dict[UserState, float],
    ) -> dict:
        ranked = sorted(
            state_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        winner, winner_probability = ranked[0]
        runner_probability = ranked[1][1] if len(ranked) > 1 else 0.0

        decision_margin = max(0.0, winner_probability - runner_probability)
        uncertainty_signal = round(max(0.0, 1.0 - (winner_probability + decision_margin)), 4)
        confidence = min(
            1.0,
            round(0.7 * winner_probability + 0.3 * decision_margin, 4),
        )

        if winner_probability < 0.42 and decision_margin < 0.12:
            return {
                "predicted_state": UserState.UNKNOWN,
                "confidence": round(min(confidence, 0.45), 4),
                "decision_margin": round(decision_margin, 4),
                "uncertainty_signal": max(uncertainty_signal, 0.55),
            }

        return {
            "predicted_state": winner,
            "confidence": round(confidence, 4),
            "decision_margin": round(decision_margin, 4),
            "uncertainty_signal": round(uncertainty_signal, 4),
        }
