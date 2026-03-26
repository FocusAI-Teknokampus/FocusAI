import unittest
from datetime import datetime

from backend.state.feature_extractor import FeatureExtractor
from backend.state.state_model import StateModel


class FatigueRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FeatureExtractor()
        self.model = StateModel()
        self.timestamp = datetime(2026, 3, 26, 14, 0, 0)

    def test_normal_message_does_not_raise_fatigue(self) -> None:
        feature = self.extractor.extract(
            session_id="reg-normal",
            message_content="Bu limitte once paydada eslenik ile sadeleştirme denedim, sonra sonucu 2 buldum.",
            message_timestamp=self.timestamp,
        )
        estimate = self.model.predict(feature)

        self.assertEqual(feature.fatigue_text_score, 0.0)
        self.assertLess(estimate.state_probabilities["fatigued"], 0.1)
        self.assertGreater(estimate.state_probabilities["focused"], 0.8)

    def test_confusion_and_fatigue_are_separated(self) -> None:
        confusion_feature = self.extractor.extract(
            session_id="reg-confusion",
            message_content="Anlamadim, bu adim neden boyle oldu?",
            message_timestamp=self.timestamp,
        )
        confusion_estimate = self.model.predict(confusion_feature)

        fatigue_feature = self.extractor.extract(
            session_id="reg-fatigue",
            message_content="Yoruldum, su an tam anlayamiyorum.",
            message_timestamp=self.timestamp,
        )
        fatigue_estimate = self.model.predict(fatigue_feature)

        self.assertEqual(confusion_feature.fatigue_text_score, 0.0)
        self.assertNotEqual(confusion_estimate.state.value, "fatigued")
        self.assertGreater(fatigue_feature.fatigue_text_score, 0.5)
        self.assertEqual(fatigue_estimate.state.value, "fatigued")
        self.assertGreater(
            fatigue_estimate.state_probabilities["fatigued"],
            confusion_estimate.state_probabilities["fatigued"],
        )


if __name__ == "__main__":
    unittest.main()
