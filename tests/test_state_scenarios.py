import unittest
from datetime import datetime

from backend.core.schemas import FeatureVector, InterventionType, ResponsePolicyMode, UserState
from backend.state.state_model import StateModel
from backend.state.uncertainty_engine import UncertaintyEngine


def make_feature(
    *,
    session_id: str,
    idle_time_seconds: float,
    retry_count: int,
    response_time_seconds: float,
    message_length: int,
    help_seeking_score: float = 0.0,
    answer_commitment_score: float = 0.0,
    confusion_score: float = 0.0,
    semantic_retry_score: float = 0.0,
    fatigue_text_score: float = 0.0,
    ear_score: float | None = None,
    gaze_on_screen: bool | None = None,
    hand_on_chin: bool | None = None,
    head_tilt_angle: float | None = None,
) -> FeatureVector:
    return FeatureVector(
        session_id=session_id,
        timestamp=datetime.utcnow(),
        idle_time_seconds=idle_time_seconds,
        retry_count=retry_count,
        response_time_seconds=response_time_seconds,
        message_length=message_length,
        topic="programlama",
        confusion_score=confusion_score,
        semantic_retry_score=semantic_retry_score,
        help_seeking_score=help_seeking_score,
        answer_commitment_score=answer_commitment_score,
        fatigue_text_score=fatigue_text_score,
        ear_score=ear_score,
        gaze_on_screen=gaze_on_screen,
        hand_on_chin=hand_on_chin,
        head_tilt_angle=head_tilt_angle,
    )


def make_baseline(
    *,
    response_mean: float,
    response_std: float,
    idle_mean: float,
    idle_std: float,
    length_mean: float,
    length_std: float,
    retry_mean: float,
    retry_std: float,
    enough_data: bool = True,
) -> dict:
    return {
        "enough_data": enough_data,
        "metrics": {
            "response_time_seconds": {
                "mean": response_mean,
                "stddev": response_std,
            },
            "idle_time_seconds": {
                "mean": idle_mean,
                "stddev": idle_std,
            },
            "message_length": {
                "mean": length_mean,
                "stddev": length_std,
            },
            "retry_count": {
                "mean": retry_mean,
                "stddev": retry_std,
            },
        },
    }


class StateScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = StateModel()
        self.engine = UncertaintyEngine()

    def test_slow_but_focused_student(self) -> None:
        baseline = make_baseline(
            response_mean=55,
            response_std=15,
            idle_mean=20,
            idle_std=10,
            length_mean=95,
            length_std=25,
            retry_mean=0.2,
            retry_std=1,
        )
        feature = make_feature(
            session_id="slow-focused",
            idle_time_seconds=28,
            retry_count=0,
            response_time_seconds=70,
            message_length=105,
            answer_commitment_score=0.65,
            gaze_on_screen=True,
            hand_on_chin=False,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)

        self.assertEqual(estimate.predicted_state, UserState.FOCUSED)
        self.assertGreaterEqual(estimate.confidence, 0.55)
        self.assertEqual(estimate.response_policy.value, "challenge")
        self.assertAlmostEqual(
            sum(estimate.state_probabilities.values()),
            1.0,
            places=3,
        )

    def test_fast_but_distracted_student(self) -> None:
        baseline = make_baseline(
            response_mean=8,
            response_std=3,
            idle_mean=10,
            idle_std=4,
            length_mean=28,
            length_std=8,
            retry_mean=0.2,
            retry_std=1,
        )
        feature = make_feature(
            session_id="fast-distracted",
            idle_time_seconds=42,
            retry_count=0,
            response_time_seconds=4,
            message_length=3,
            help_seeking_score=0.45,
            gaze_on_screen=False,
            hand_on_chin=True,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)
        intervention = self.engine.decide(estimate=estimate, session_id="fast-distracted")

        self.assertEqual(estimate.predicted_state, UserState.DISTRACTED)
        self.assertGreaterEqual(estimate.confidence, 0.5)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.intervention_type, InterventionType.BREAK)

    def test_stuck_student(self) -> None:
        baseline = make_baseline(
            response_mean=14,
            response_std=4,
            idle_mean=18,
            idle_std=8,
            length_mean=70,
            length_std=18,
            retry_mean=0.0,
            retry_std=1,
        )
        feature = make_feature(
            session_id="stuck-student",
            idle_time_seconds=240,
            retry_count=4,
            response_time_seconds=22,
            message_length=52,
            help_seeking_score=0.5,
            answer_commitment_score=0.72,
            gaze_on_screen=True,
            hand_on_chin=False,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)
        intervention = self.engine.decide(estimate=estimate, session_id="stuck-student")

        self.assertEqual(estimate.predicted_state, UserState.STUCK)
        self.assertEqual(estimate.learning_pattern.value, "deep_attempt")
        self.assertGreaterEqual(estimate.confidence, 0.45)
        self.assertEqual(estimate.response_policy.value, "guided_hint")
        self.assertTrue(estimate.dominant_signals)
        self.assertTrue(estimate.reasons)
        self.assertTrue(estimate.policy_path)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.intervention_type, InterventionType.HINT)

    def test_fatigued_student(self) -> None:
        baseline = make_baseline(
            response_mean=16,
            response_std=5,
            idle_mean=25,
            idle_std=10,
            length_mean=80,
            length_std=20,
            retry_mean=0.3,
            retry_std=1,
        )
        feature = make_feature(
            session_id="fatigued-student",
            idle_time_seconds=360,
            retry_count=0,
            response_time_seconds=28,
            message_length=75,
            ear_score=0.18,
            gaze_on_screen=True,
            hand_on_chin=False,
            head_tilt_angle=31,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)
        intervention = self.engine.decide(estimate=estimate, session_id="fatigued-student")

        self.assertEqual(estimate.predicted_state, UserState.FATIGUED)
        self.assertGreaterEqual(estimate.confidence, 0.5)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.intervention_type, InterventionType.BREAK)

    def test_hint_policy_can_override_default_strategy_for_shallow_stuck_student(self) -> None:
        baseline = make_baseline(
            response_mean=10,
            response_std=4,
            idle_mean=14,
            idle_std=6,
            length_mean=35,
            length_std=10,
            retry_mean=0.1,
            retry_std=1,
        )
        feature = make_feature(
            session_id="policy-before",
            idle_time_seconds=40,
            retry_count=5,
            response_time_seconds=3,
            message_length=10,
            help_seeking_score=0.75,
            answer_commitment_score=0.1,
            gaze_on_screen=True,
            hand_on_chin=False,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)
        estimate = estimate.model_copy(update={"response_policy": ResponsePolicyMode.GUIDED_HINT})
        before = self.engine.decide(estimate=estimate, session_id="policy-before")

        self.assertEqual(estimate.predicted_state, UserState.STUCK)
        self.assertEqual(estimate.learning_pattern.value, "shallow_learning")
        self.assertEqual(estimate.response_policy.value, "guided_hint")
        self.assertIsNotNone(before)
        self.assertEqual(before.intervention_type, InterventionType.STRATEGY)

        summary = {
            "hint": {
                "intervention_type": "hint",
                "total_count": 5,
                "success_rate": 1.0,
                "recent_success_rate": 1.0,
            },
            "strategy": {
                "intervention_type": "strategy",
                "total_count": 5,
                "success_rate": 0.0,
                "recent_success_rate": 0.0,
            },
        }

        after = self.engine.decide(
            estimate=estimate,
            session_id="policy-after",
            policy_summary=summary,
        )

        self.assertIsNotNone(after)
        self.assertEqual(after.intervention_type, InterventionType.HINT)
        self.assertIn("hint", after.decision_reason.lower())
        self.assertGreater(summary["hint"]["success_rate"], summary["strategy"]["success_rate"])

    def test_explicit_fatigue_language_blocks_focused_bias(self) -> None:
        baseline = make_baseline(
            response_mean=12,
            response_std=4,
            idle_mean=18,
            idle_std=6,
            length_mean=52,
            length_std=18,
            retry_mean=0.1,
            retry_std=1,
        )
        feature = make_feature(
            session_id="fatigue-text-student",
            idle_time_seconds=24,
            retry_count=0,
            response_time_seconds=18,
            message_length=34,
            confusion_score=0.42,
            fatigue_text_score=0.82,
            answer_commitment_score=0.18,
            gaze_on_screen=True,
            hand_on_chin=False,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)

        self.assertNotEqual(estimate.predicted_state, UserState.FOCUSED)
        self.assertGreater(
            estimate.state_probabilities[UserState.FATIGUED.value],
            estimate.state_probabilities[UserState.FOCUSED.value],
        )

    def test_explicit_fatigue_language_prefers_recovery_policy(self) -> None:
        baseline = make_baseline(
            response_mean=10,
            response_std=4,
            idle_mean=12,
            idle_std=5,
            length_mean=40,
            length_std=14,
            retry_mean=0.2,
            retry_std=1,
        )
        feature = make_feature(
            session_id="fatigue-recovery",
            idle_time_seconds=35,
            retry_count=0,
            response_time_seconds=14,
            message_length=32,
            confusion_score=0.28,
            fatigue_text_score=0.9,
            answer_commitment_score=0.16,
            gaze_on_screen=True,
            hand_on_chin=False,
        )

        estimate = self.model.predict(feature, baseline_profile=baseline)

        self.assertEqual(estimate.response_policy, ResponsePolicyMode.RECOVERY)


if __name__ == "__main__":
    unittest.main()
