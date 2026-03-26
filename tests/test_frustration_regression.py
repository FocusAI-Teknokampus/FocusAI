import unittest
from datetime import datetime

from backend.state.feature_extractor import FeatureExtractor
from backend.state.state_model import StateModel


class FrustrationRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FeatureExtractor()
        self.model = StateModel()
        self.timestamp = datetime(2026, 3, 26, 16, 0, 0)

    def test_normal_message_does_not_raise_frustration(self) -> None:
        feature = self.extractor.extract(
            session_id="fr-normal",
            message_content="Bu ornekte once paydada eslenik ile sadeleştirme yaptim ve sonucu 2 buldum.",
            message_timestamp=self.timestamp,
        )
        estimate = self.model.predict(feature)

        self.assertEqual(feature.frustration_text_score, 0.0)
        self.assertLess(estimate.state_probabilities["stuck"], 0.15)
        self.assertGreater(estimate.state_probabilities["focused"], 0.8)

    def test_fatigue_and_frustration_are_separated(self) -> None:
        fatigue_feature = self.extractor.extract(
            session_id="fr-fatigue",
            message_content="Yoruldum, su an anlayamiyorum.",
            message_timestamp=self.timestamp,
        )
        frustration_feature = self.extractor.extract(
            session_id="fr-frustration",
            message_content="Of ya, bu cok sinir bozucu, yine olmadi.",
            message_timestamp=self.timestamp,
        )

        self.assertGreater(fatigue_feature.fatigue_text_score, 0.4)
        self.assertEqual(fatigue_feature.frustration_text_score, 0.0)
        self.assertGreater(frustration_feature.frustration_text_score, 0.4)
        self.assertEqual(frustration_feature.fatigue_text_score, 0.0)


if __name__ == "__main__":
    unittest.main()
