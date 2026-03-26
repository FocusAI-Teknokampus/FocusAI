import unittest
from datetime import datetime, timedelta

from backend.state.feature_extractor import FeatureExtractor


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FeatureExtractor()

    def test_repeated_question_builds_semantic_retry_and_retry_count(self) -> None:
        base_time = datetime(2026, 3, 26, 10, 0, 0)

        first = self.extractor.extract(
            session_id="s1",
            message_content="Python API neden calismiyor?",
            message_timestamp=base_time,
        )
        second = self.extractor.extract(
            session_id="s1",
            message_content="Python API neden calismiyor?",
            message_timestamp=base_time + timedelta(seconds=20),
        )

        self.assertEqual(first.retry_count, 0)
        self.assertGreaterEqual(second.semantic_retry_score, 0.9)
        self.assertGreaterEqual(second.retry_count, 1)
        self.assertEqual(second.topic, "programlama")

    def test_confused_follow_up_exposes_confusion_and_topic_stability(self) -> None:
        base_time = datetime(2026, 3, 26, 10, 0, 0)

        self.extractor.extract(
            session_id="s2",
            message_content="Integral ornegini beraber cozelim.",
            message_timestamp=base_time,
        )
        follow_up = self.extractor.extract(
            session_id="s2",
            message_content="Hala anlamadim, integral neden boyle olmadi?",
            message_timestamp=base_time + timedelta(seconds=45),
        )

        self.assertEqual(follow_up.topic, "matematik")
        self.assertGreaterEqual(follow_up.confusion_score, 0.5)
        self.assertGreaterEqual(follow_up.topic_stability, 0.9)
        self.assertGreater(follow_up.question_density, 0.1)


if __name__ == "__main__":
    unittest.main()
