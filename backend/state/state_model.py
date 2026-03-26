from __future__ import annotations

from backend.core.config import settings
from backend.core.schemas import StateEstimate
from backend.services.response_policy_service import ResponsePolicyService
from backend.state.decision_controller import DecisionController
from backend.state.feature_deviation_scorer import FeatureDeviationScorer
from backend.state.probabilistic_scorer import ProbabilisticScorer
from backend.state.rule_signal_extractor import RuleSignalExtractor


class StateModel:
    """
    Alt bilesenleri orkestre ederek state tahmini uretir.
    """

    def __init__(self):
        self.deviation_scorer = FeatureDeviationScorer()
        self.rule_extractor = RuleSignalExtractor()
        self.probabilistic_scorer = ProbabilisticScorer()
        self.decision_controller = DecisionController()
        self.response_policy_service = ResponsePolicyService()

    def predict(
        self,
        features,
        adaptive_threshold: float = None,
        baseline_profile: dict | None = None,
    ) -> StateEstimate:
        threshold = adaptive_threshold or settings.default_uncertainty_threshold

        deviation_features = self.deviation_scorer.build(features, baseline_profile)
        rule_scores = self.rule_extractor.extract(features)
        raw_scores, state_probabilities = self.probabilistic_scorer.score(
            features=features,
            deviation_features=deviation_features,
            rule_scores=rule_scores,
        )
        decision = self.decision_controller.select(state_probabilities)
        learning_pattern = self.probabilistic_scorer.detect_learning_pattern(
            features=features,
            deviation_features=deviation_features,
        )

        predicted_state = decision["predicted_state"]

        estimate = StateEstimate(
            session_id=features.session_id,
            state=predicted_state,
            predicted_state=predicted_state,
            confidence=round(decision["confidence"], 2),
            decision_margin=round(decision["decision_margin"], 3),
            uncertainty_signal=round(decision["uncertainty_signal"], 3),
            learning_pattern=learning_pattern,
            threshold=threshold,
            deviation_features=deviation_features,
            state_scores={
                key.value: round(value, 3)
                for key, value in raw_scores.items()
            },
            state_probabilities={
                key.value: round(value, 4)
                for key, value in state_probabilities.items()
            },
        )
        policy = self.response_policy_service.decide(
            estimate=estimate,
            feature_vector=features,
            baseline_profile=baseline_profile,
        )
        return estimate.model_copy(update=policy)
