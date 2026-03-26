from __future__ import annotations

from datetime import datetime
from typing import Optional

from backend.core.config import settings
from backend.core.schemas import (
    InterventionType,
    LearningPattern,
    MentorIntervention,
    StateEstimate,
    UserProfile,
    UserState,
)


class UncertaintyEngine:
    """
    Confidence, cooldown ve kullanıcıya özel müdahale başarısına göre karar verir.
    """

    def __init__(self):
        self._last_intervention: dict[str, datetime] = {}
        self._recent_interventions: dict[str, list[str]] = {}

    def decide(
        self,
        estimate: StateEstimate,
        profile: Optional[UserProfile] = None,
        session_id: str = "",
        policy_summary: Optional[dict[str, dict]] = None,
    ) -> Optional[MentorIntervention]:
        threshold = settings.default_uncertainty_threshold
        if profile and profile.adaptive_threshold:
            threshold = profile.adaptive_threshold

        if not self._should_intervene(estimate, threshold, session_id):
            return None

        candidates = self._candidate_interventions(estimate)
        intervention_type, policy_snapshot, decision_reason = self._select_intervention_type(
            estimate=estimate,
            candidates=candidates,
            policy_summary=policy_summary or {},
            session_id=session_id,
        )

        if intervention_type == InterventionType.NONE:
            return None

        message = self._generate_message(
            intervention_type, estimate.state, estimate.learning_pattern
        )

        if session_id:
            self._last_intervention[session_id] = datetime.now()
            recent = self._recent_interventions.get(session_id, [])
            recent.append(intervention_type.value)
            self._recent_interventions[session_id] = recent[-5:]

        return MentorIntervention(
            intervention_type=intervention_type,
            message=message,
            triggered_by=estimate.state,
            learning_pattern=estimate.learning_pattern,
            confidence=estimate.confidence,
            decision_reason=decision_reason,
            policy_snapshot=policy_snapshot,
        )

    def _should_intervene(
        self,
        estimate: StateEstimate,
        threshold: float,
        session_id: str,
    ) -> bool:
        if estimate.state in [UserState.FOCUSED, UserState.UNKNOWN]:
            return False

        if estimate.confidence < threshold:
            return False

        if session_id and session_id in self._last_intervention:
            elapsed = (
                datetime.now() - self._last_intervention[session_id]
            ).total_seconds()
            if elapsed < settings.intervention_cooldown_seconds:
                return False

        return True

    def _candidate_interventions(self, estimate: StateEstimate) -> list[InterventionType]:
        state = estimate.state
        pattern = estimate.learning_pattern

        if estimate.confidence < estimate.threshold:
            return [InterventionType.QUESTION]

        if state == UserState.FATIGUED:
            return [InterventionType.BREAK, InterventionType.QUESTION, InterventionType.MODE_SWITCH]

        if state == UserState.STUCK:
            if pattern == LearningPattern.SHALLOW_LEARNING:
                return [InterventionType.STRATEGY, InterventionType.HINT, InterventionType.QUESTION]
            if pattern == LearningPattern.DEEP_ATTEMPT:
                return [InterventionType.HINT, InterventionType.STRATEGY, InterventionType.BREAK]
            if pattern == LearningPattern.MISCONCEPTION:
                return [InterventionType.QUESTION, InterventionType.HINT, InterventionType.MODE_SWITCH]
            return [InterventionType.HINT, InterventionType.STRATEGY, InterventionType.QUESTION]

        if state == UserState.DISTRACTED:
            return [InterventionType.QUESTION, InterventionType.MODE_SWITCH, InterventionType.BREAK]

        return [InterventionType.NONE]

    def _select_intervention_type(
        self,
        estimate: StateEstimate,
        candidates: list[InterventionType],
        policy_summary: dict[str, dict],
        session_id: str,
    ) -> tuple[InterventionType, dict, str]:
        scored_candidates: list[tuple[float, InterventionType]] = []
        snapshots: dict[str, dict] = {}

        for index, candidate in enumerate(candidates):
            if candidate == InterventionType.NONE:
                continue

            base_score = max(0.2, 1.0 - index * 0.15)
            policy = policy_summary.get(candidate.value, {})
            success_rate = policy.get("success_rate")
            recent_success_rate = policy.get("recent_success_rate")
            total_count = policy.get("total_count", 0)

            if success_rate is not None:
                base_score += success_rate * 1.2
            if recent_success_rate is not None:
                base_score += recent_success_rate * 1.4

            if total_count == 0:
                base_score += 0.05

            recent_penalty = self._recent_penalty(session_id, candidate.value)
            base_score -= recent_penalty

            if estimate.state == UserState.FATIGUED and candidate == InterventionType.BREAK:
                base_score += 0.2
            if estimate.state == UserState.STUCK and candidate == InterventionType.HINT:
                base_score += 0.1

            scored_candidates.append((base_score, candidate))
            snapshots[candidate.value] = {
                "candidate_rank": index + 1,
                "success_rate": success_rate,
                "recent_success_rate": recent_success_rate,
                "total_count": total_count,
                "repeat_penalty": round(recent_penalty, 3),
                "final_score": round(base_score, 3),
            }

        if not scored_candidates:
            return (
                InterventionType.NONE,
                {},
                "Uygun aday mudahale bulunamadi.",
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        winner_score, winner = scored_candidates[0]
        winner_snapshot = snapshots.get(winner.value, {})
        success_rate = winner_snapshot.get("success_rate")
        recent_success_rate = winner_snapshot.get("recent_success_rate")
        repeat_penalty = winner_snapshot.get("repeat_penalty", 0.0)
        reason = (
            f"{estimate.state.value} durumunda adaylar arasinda "
            f"'{winner.value}' en yuksek skoru aldi. "
            f"Success rate={success_rate}, recent success rate={recent_success_rate}, "
            f"repeat penalty={repeat_penalty}, final score={round(winner_score, 3)}."
        )
        return winner, winner_snapshot, reason

    def _recent_penalty(self, session_id: str, intervention_type: str) -> float:
        if not session_id:
            return 0.0

        recent = self._recent_interventions.get(session_id, [])
        if not recent:
            return 0.0

        repeat_count = sum(1 for item in recent[-3:] if item == intervention_type)
        if repeat_count >= 2:
            return 0.45
        if repeat_count == 1:
            return 0.15
        return 0.0

    def _generate_message(
        self,
        intervention_type: InterventionType,
        state: UserState,
        pattern: LearningPattern,
    ) -> str:
        messages = {
            InterventionType.HINT: (
                "Bu konuda biraz takildin gibi gorunuyor. "
                "Problemi farkli bir acidan ele almayi dener misin?"
            ),
            InterventionType.STRATEGY: (
                "Cok sayida kisa soru soruyorsun. "
                "Bir adim geri cekilip genel yapinin ozetine bakalim."
            ),
            InterventionType.BREAK: (
                "Bir suredir yogun calisiyorsun. "
                "5 dakikalik kisa bir mola verip geri donmek faydali olabilir."
            ),
            InterventionType.QUESTION: (
                "Su an nasil hissettigini biraz daha anlatir misin? "
                "Takildigin noktayi birlikte netlestirebiliriz."
            ),
            InterventionType.MODE_SWITCH: (
                "Bu konuyu farkli bir yontemle deneyelim. "
                "Istersen kisa ozet veya adim adim moduna gecebiliriz."
            ),
        }
        return messages.get(
            intervention_type,
            "Nasil gidiyor? Yardimci olabilecegim bir sey var mi?"
        )
