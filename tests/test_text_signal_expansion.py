import unittest
from datetime import datetime

from backend.state.feature_extractor import FeatureExtractor
from backend.state.state_model import StateModel


class TextSignalExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FeatureExtractor()
        self.model = StateModel()
        self.timestamp = datetime(2026, 3, 26, 17, 0, 0)

    def test_confidence_signal_boosts_focused_without_fatigue_bleed(self) -> None:
        feature = self.extractor.extract(
            session_id="exp-confidence",
            message_content="Bence cozdum, sonuc 2 cikiyor ve bu adimdan eminim.",
            message_timestamp=self.timestamp,
        )
        estimate = self.model.predict(feature)

        self.assertGreater(feature.confidence_text_score, 0.4)
        self.assertEqual(feature.fatigue_text_score, 0.0)
        self.assertGreater(estimate.state_probabilities["focused"], estimate.state_probabilities["fatigued"])

    def test_overwhelm_signal_prefers_fatigued_over_focused(self) -> None:
        feature = self.extractor.extract(
            session_id="exp-overwhelm",
            message_content="Bunaldim, her sey birbirine girdi, nereden baslayacagimi bilmiyorum.",
            message_timestamp=self.timestamp,
        )
        estimate = self.model.predict(feature)

        self.assertGreater(feature.overwhelm_text_score, 0.4)
        self.assertGreater(estimate.state_probabilities["fatigued"], estimate.state_probabilities["focused"])

    def test_urgency_signal_changes_policy_more_than_state(self) -> None:
        feature = self.extractor.extract(
            session_id="exp-urgency",
            message_content="Acele cevap lazim, sinavim var, hizlica ozetler misin?",
            message_timestamp=self.timestamp,
        )
        estimate = self.model.predict(feature)

        self.assertGreater(feature.urgency_text_score, 0.4)
        self.assertEqual(estimate.response_policy.value, "direct_help")


if __name__ == "__main__":
    unittest.main()
