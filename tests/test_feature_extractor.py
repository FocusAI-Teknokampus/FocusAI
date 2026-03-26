import unittest
from datetime import datetime, timedelta

from backend.core.schemas import CameraSignal
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
        self.assertGreaterEqual(second.help_seeking_score, 0.1)
        self.assertGreaterEqual(second.help_seeking_classifier_score, 0.45)
        self.assertLessEqual(second.answer_commitment_score, 0.2)

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

    def test_attempt_message_raises_answer_commitment(self) -> None:
        feature = self.extractor.extract(
            session_id="s3",
            message_content="Integralde once parcali integrasyon denedim, x^2 * e^x tarafinda takildim.",
            message_timestamp=datetime(2026, 3, 26, 11, 0, 0),
        )

        self.assertGreaterEqual(feature.answer_commitment_score, 0.3)
        self.assertGreaterEqual(feature.answer_commitment_semantic_score, 0.1)
        self.assertGreaterEqual(feature.answer_commitment_classifier_score, 0.45)
        self.assertGreaterEqual(feature.help_seeking_score, 0.0)

    def test_topic_bank_detects_related_math_concepts_without_exact_keyword(self) -> None:
        feature = self.extractor.extract(
            session_id="s4",
            message_content="Diferansiyel denklemde baslangic kosulunu nereye uygulayacagim?",
            message_timestamp=datetime(2026, 3, 26, 11, 5, 0),
        )

        self.assertEqual(feature.topic, "matematik")
        self.assertGreaterEqual(feature.topic_confidence, 0.34)
        self.assertGreaterEqual(feature.help_seeking_score, 0.15)

    def test_semantic_retry_handles_close_rephrase(self) -> None:
        base_time = datetime(2026, 3, 26, 11, 10, 0)

        self.extractor.extract(
            session_id="s5",
            message_content="Python API neden 500 hatasi veriyor?",
            message_timestamp=base_time,
        )
        second = self.extractor.extract(
            session_id="s5",
            message_content="Bu API neden yine server hatasi donuyor?",
            message_timestamp=base_time + timedelta(seconds=25),
        )

        self.assertEqual(second.topic, "programlama")
        self.assertGreaterEqual(second.semantic_retry_score, 0.55)
        self.assertGreaterEqual(second.retry_count, 1)

    def test_camera_signal_is_mapped_into_feature_vector(self) -> None:
        feature = self.extractor.extract(
            session_id="s6",
            message_content="Fonksiyon sorusunu cozuyorum.",
            message_timestamp=datetime(2026, 3, 26, 11, 20, 0),
            camera_signal=CameraSignal(
                ear_score=0.19,
                gaze_on_screen=False,
                hand_on_chin=True,
                head_tilt_angle=27.0,
            ),
        )

        self.assertEqual(feature.ear_score, 0.19)
        self.assertFalse(feature.gaze_on_screen)
        self.assertTrue(feature.hand_on_chin)
        self.assertEqual(feature.head_tilt_angle, 27.0)

    def test_explicit_fatigue_language_raises_fatigue_text_score(self) -> None:
        feature = self.extractor.extract(
            session_id="s7",
            message_content="Yoruldum, su an tam anlayamiyorum.",
            message_timestamp=datetime(2026, 3, 26, 11, 30, 0),
        )

        self.assertGreater(feature.fatigue_text_score, 0.0)
        self.assertGreaterEqual(feature.confusion_score, 0.1)

    def test_explicit_frustration_language_raises_frustration_text_score(self) -> None:
        feature = self.extractor.extract(
            session_id="s8",
            message_content="Of ya, bu soru cok sinir bozucu, yine olmadi.",
            message_timestamp=datetime(2026, 3, 26, 11, 35, 0),
        )

        self.assertGreater(feature.frustration_text_score, 0.0)
        self.assertEqual(feature.fatigue_text_score, 0.0)

    def test_confidence_overwhelm_and_urgency_scores_are_separated(self) -> None:
        confidence = self.extractor.extract(
            session_id="s9",
            message_content="Bence cozdum, sonuc 2 cikiyor ve bu adimdan eminim.",
            message_timestamp=datetime(2026, 3, 26, 11, 40, 0),
        )
        overwhelm = self.extractor.extract(
            session_id="s10",
            message_content="Bunaldim, her sey birbirine girdi, nereden baslayacagimi bilmiyorum.",
            message_timestamp=datetime(2026, 3, 26, 11, 41, 0),
        )
        urgency = self.extractor.extract(
            session_id="s11",
            message_content="Acele cevap lazim, sinavim var, hizlica ozetler misin?",
            message_timestamp=datetime(2026, 3, 26, 11, 42, 0),
        )

        self.assertGreater(confidence.confidence_text_score, 0.0)
        self.assertEqual(confidence.overwhelm_text_score, 0.0)
        self.assertEqual(confidence.urgency_text_score, 0.0)
        self.assertGreater(overwhelm.overwhelm_text_score, 0.0)
        self.assertEqual(overwhelm.confidence_text_score, 0.0)
        self.assertGreater(urgency.urgency_text_score, 0.0)


if __name__ == "__main__":
    unittest.main()
