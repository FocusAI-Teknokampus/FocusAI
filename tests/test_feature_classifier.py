import unittest
from pathlib import Path

from backend.state.feature_classifier import FeatureIntentClassifier, LabeledExample
from backend.state.semantic_features import SemanticFeatureProvider
from scripts.refresh_feature_classifier import build_datasets


class FeatureIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.semantic_provider = SemanticFeatureProvider.from_settings()
        self.classifier = FeatureIntentClassifier(self.semantic_provider)

    def test_help_seeking_scores_request_higher_than_attempt(self) -> None:
        request_score = self.classifier.score_help_seeking(
            "Burada takildim, bana kisa bir ipucu verir misin?"
        )
        attempt_score = self.classifier.score_help_seeking(
            "Once zincir kuralini denedim ve ikinci satirda isaret hatasi yaptim."
        )

        self.assertGreater(request_score, attempt_score)
        self.assertGreaterEqual(request_score, 0.55)

    def test_answer_commitment_scores_attempt_higher_than_direct_answer_request(self) -> None:
        attempt_score = self.classifier.score_answer_commitment(
            "Benim denemem su sekilde, once denklemi sadelestirdim sonra integral kurdum."
        )
        shortcut_score = self.classifier.score_answer_commitment(
            "Sadece cevabi ver."
        )

        self.assertGreater(attempt_score, shortcut_score)
        self.assertGreaterEqual(attempt_score, 0.55)
        self.assertLessEqual(shortcut_score, 0.5)

    def test_classifier_artifact_round_trip_preserves_scores(self) -> None:
        text = "Burada takildim, nasil ilerleyecegimi soyle."
        original_help = self.classifier.score_help_seeking(text)
        original_commitment = self.classifier.score_answer_commitment(text)

        artifact_dir = Path(".tmp_testdata") / "classifier_tests"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "feature_classifier_model.json"
        try:
            self.classifier.save_artifact(artifact_path=artifact_path)
            loaded = FeatureIntentClassifier(
                self.semantic_provider,
                artifact_path=artifact_path,
            )
        finally:
            artifact_path.unlink(missing_ok=True)

        self.assertEqual(original_help, loaded.score_help_seeking(text))
        self.assertEqual(original_commitment, loaded.score_answer_commitment(text))

    def test_build_datasets_adds_labeled_rows(self) -> None:
        rows = [
            {
                "content": "Bana ipucu verir misin?",
                "label_help_seeking": 1,
                "label_answer_commitment": 0,
            },
            {
                "content": "Benim denemem bu, integral kurdum.",
                "label_help_seeking": 0,
                "label_answer_commitment": 1,
            },
        ]

        help_dataset, commitment_dataset = build_datasets(rows=rows, include_seed=False)

        self.assertIn(LabeledExample(text="Bana ipucu verir misin?", label=1), help_dataset)
        self.assertIn(LabeledExample(text="Benim denemem bu, integral kurdum.", label=1), commitment_dataset)
        self.assertEqual({example.label for example in help_dataset}, {0, 1})
        self.assertEqual({example.label for example in commitment_dataset}, {0, 1})


if __name__ == "__main__":
    unittest.main()
