from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.core.config import settings
from backend.state.semantic_features import (
    SemanticFeatureProvider,
    average_vectors,
    cosine_similarity,
    normalize_text,
)


@dataclass(frozen=True)
class LabeledExample:
    text: str
    label: int


@dataclass(frozen=True)
class BinaryClassifierModel:
    name: str
    positive_centroid: list[float]
    negative_centroid: list[float]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    feature_weights: dict[str, float]
    feature_midpoints: dict[str, float]
    intercept: float


HELP_SEEKING_DATASET: tuple[LabeledExample, ...] = (
    LabeledExample("Bana sadece kisa bir ipucu verir misin?", 1),
    LabeledExample("Burada takildim, nasil ilerlemeliyim?", 1),
    LabeledExample("Bu adimi aciklayabilir misin?", 1),
    LabeledExample("Neden boyle oldugunu anlamadim.", 1),
    LabeledExample("Direkt cevabi verme, sadece yon ver.", 1),
    LabeledExample("Bu soruda yardima ihtiyacim var.", 1),
    LabeledExample("Hangi kuraldan devam etmem gerektigini soyle.", 1),
    LabeledExample("Bu kisim neden yanlis oldu, gosterir misin?", 1),
    LabeledExample("Once zincir kuralini denedim ve ikinci satirda hata yaptim.", 0),
    LabeledExample("Benim cozumumde integral siniri burada degisiyor.", 0),
    LabeledExample("Ilk olarak denklemi sadelestirdim sonra sonucu kontrol ettim.", 0),
    LabeledExample("Bence hata isaret degisiminde.", 0),
    LabeledExample("Python fonksiyonum state guncellemiyor, log ekledim.", 0),
    LabeledExample("Denklemin iki tarafini x ile carptim.", 0),
    LabeledExample("Sonucum 3 cikti ama emin degilim.", 0),
    LabeledExample("Bu benim su ana kadarki denemem.", 0),
)


ANSWER_COMMITMENT_DATASET: tuple[LabeledExample, ...] = (
    LabeledExample("Once zincir kuralini denedim ama ikinci satirda hata yaptim.", 1),
    LabeledExample("Benim denemem su sekilde, burada takildim.", 1),
    LabeledExample("Ilk olarak denklemi sadelestirdim sonra integral kurdum.", 1),
    LabeledExample("Bence hata turev alirken oldu.", 1),
    LabeledExample("x^2 * e^x icin parcali integrasyon yazdim.", 1),
    LabeledExample("Kodumda once request attim sonra response parse ettim.", 1),
    LabeledExample("My attempt gets 42 but I think the sign is wrong.", 1),
    LabeledExample("Su ana kadar yaptigim cozum bu.", 1),
    LabeledExample("Sadece cevabi soyle.", 0),
    LabeledExample("Direkt sonucu ver.", 0),
    LabeledExample("Bana cozum yolunu yaz.", 0),
    LabeledExample("Burada ne yapacagimi soyle.", 0),
    LabeledExample("Yardim eder misin?", 0),
    LabeledExample("Bu kisim neden olmadi?", 0),
    LabeledExample("Explain the answer directly.", 0),
    LabeledExample("Bana sadece sonucu ver gec.", 0),
)


class FeatureIntentClassifier:
    def __init__(
        self,
        semantic_provider: SemanticFeatureProvider,
        artifact_path: Optional[str | Path] = None,
    ):
        self._semantic_provider = semantic_provider
        self._artifact_path = Path(artifact_path) if artifact_path else self.default_artifact_path()

        loaded = self._load_models_from_artifact(self._artifact_path)
        if loaded is not None:
            self._help_model, self._commitment_model = loaded
            return

        self._help_model = self._fit_model("help_seeking", HELP_SEEKING_DATASET)
        self._commitment_model = self._fit_model(
            "answer_commitment",
            ANSWER_COMMITMENT_DATASET,
        )

    @classmethod
    def default_artifact_path(cls) -> Path:
        return Path(settings.feature_classifier_model_path)

    @classmethod
    def from_datasets(
        cls,
        semantic_provider: SemanticFeatureProvider,
        help_dataset: tuple[LabeledExample, ...],
        commitment_dataset: tuple[LabeledExample, ...],
    ) -> "FeatureIntentClassifier":
        classifier = cls.__new__(cls)
        classifier._semantic_provider = semantic_provider
        classifier._artifact_path = cls.default_artifact_path()
        classifier._help_model = classifier._fit_model("help_seeking", help_dataset)
        classifier._commitment_model = classifier._fit_model(
            "answer_commitment",
            commitment_dataset,
        )
        return classifier

    def score_help_seeking(self, text: str) -> float:
        return self._score(self._help_model, text)

    def score_answer_commitment(self, text: str) -> float:
        return self._score(self._commitment_model, text)

    def save_artifact(
        self,
        artifact_path: Optional[str | Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        path = Path(artifact_path) if artifact_path else self._artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "models": {
                "help_seeking": self._model_to_payload(self._help_model),
                "answer_commitment": self._model_to_payload(self._commitment_model),
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _fit_model(
        self,
        name: str,
        dataset: tuple[LabeledExample, ...],
    ) -> BinaryClassifierModel:
        texts = [item.text for item in dataset]
        embeddings = self._semantic_provider.embed_texts(texts)
        feature_rows = [self._extract_features(item.text) for item in dataset]

        positives = [item for item in dataset if item.label == 1]
        negatives = [item for item in dataset if item.label == 0]

        pos_embeddings = [embedding for item, embedding in zip(dataset, embeddings) if item.label == 1]
        neg_embeddings = [embedding for item, embedding in zip(dataset, embeddings) if item.label == 0]

        feature_names = tuple(feature_rows[0].keys()) if feature_rows else ()
        pos_means = {
            key: self._mean([row[key] for row, item in zip(feature_rows, dataset) if item.label == 1])
            for key in feature_names
        }
        neg_means = {
            key: self._mean([row[key] for row, item in zip(feature_rows, dataset) if item.label == 0])
            for key in feature_names
        }

        feature_weights = {
            key: round((pos_means[key] - neg_means[key]) * 2.6, 4)
            for key in feature_names
        }
        feature_midpoints = {
            key: (pos_means[key] + neg_means[key]) / 2
            for key in feature_names
        }
        intercept = math.log((len(positives) + 1) / (len(negatives) + 1)) * 0.25

        return BinaryClassifierModel(
            name=name,
            positive_centroid=average_vectors(pos_embeddings),
            negative_centroid=average_vectors(neg_embeddings),
            positive_examples=tuple(item.text for item in positives),
            negative_examples=tuple(item.text for item in negatives),
            feature_weights=feature_weights,
            feature_midpoints=feature_midpoints,
            intercept=intercept,
        )

    def _score(self, model: BinaryClassifierModel, text: str) -> float:
        embedding = self._semantic_provider.embed_text(text)
        features = self._extract_features(text)

        centroid_delta = cosine_similarity(embedding, model.positive_centroid) - cosine_similarity(
            embedding,
            model.negative_centroid,
        )
        example_delta = self._best_similarity(embedding, model.positive_examples) - self._best_similarity(
            embedding,
            model.negative_examples,
        )

        raw = model.intercept
        raw += centroid_delta * 3.4
        raw += example_delta * 1.8

        for name, weight in model.feature_weights.items():
            midpoint = model.feature_midpoints.get(name, 0.5)
            raw += (features.get(name, 0.0) - midpoint) * weight

        probability = 1.0 / (1.0 + math.exp(-raw))
        return round(max(0.0, min(1.0, probability)), 3)

    def _best_similarity(self, embedding: list[float], examples: tuple[str, ...]) -> float:
        if not examples:
            return 0.0
        vectors = self._semantic_provider.embed_texts(list(examples))
        return max(cosine_similarity(embedding, vector) for vector in vectors)

    def _extract_features(self, text: str) -> dict[str, float]:
        normalized = normalize_text(text)
        tokens = normalized.split()
        token_count = max(1, len(tokens))

        help_patterns = (
            "yardim",
            "ipucu",
            "anlat",
            "goster",
            "soyle",
            "ver",
            "coz",
            "help",
            "explain",
            "solution",
        )
        direct_patterns = (
            "sadece cevabi",
            "direkt cevabi",
            "cevabi soyle",
            "just answer",
            "just give",
            "sonucu ver",
        )
        effort_patterns = (
            "denedim",
            "yaptim",
            "hesapladim",
            "buldum",
            "bence",
            "benim",
            "kodum",
            "my attempt",
            "i tried",
            "i got",
            "cozumum",
        )
        confusion_patterns = (
            "takildim",
            "anlamadim",
            "olmadi",
            "karisti",
            "emin degilim",
            "neden boyle",
        )
        first_person_tokens = {"ben", "bence", "benim", "kodum", "cozumum", "i", "my"}
        interrogatives = {"neden", "nasil", "hangi", "nereye", "ne", "niye", "why", "how", "what"}

        prompt_like = 1.0 if normalized.startswith(tuple(interrogatives)) else 0.0
        question_density = min(1.0, ((text.count("?") * 2) + prompt_like) / token_count)
        help_density = sum(1 for pattern in help_patterns if pattern in normalized) / max(1, len(help_patterns) / 2)
        direct_density = sum(1 for pattern in direct_patterns if pattern in normalized) / max(1, len(direct_patterns) / 2)
        effort_density = sum(1 for pattern in effort_patterns if pattern in normalized) / max(1, len(effort_patterns) / 2)
        confusion_density = sum(1 for pattern in confusion_patterns if pattern in normalized) / max(1, len(confusion_patterns) / 2)
        first_person_density = sum(1 for token in tokens if token in first_person_tokens) / token_count
        interrogative_density = sum(1 for token in tokens if token in interrogatives) / token_count
        math_signal = 1.0 if re.search(r"[0-9_=<>/*+\-]", text) else 0.0
        short_signal = max(0.0, min(1.0, 1 - ((len(text) - 28) / 90)))
        long_signal = max(0.0, min(1.0, (len(text) - 45) / 120))

        return {
            "question_density": round(question_density, 4),
            "help_density": round(min(1.0, help_density), 4),
            "direct_density": round(min(1.0, direct_density), 4),
            "effort_density": round(min(1.0, effort_density), 4),
            "confusion_density": round(min(1.0, confusion_density), 4),
            "first_person_density": round(min(1.0, first_person_density * 3), 4),
            "interrogative_density": round(min(1.0, interrogative_density * 3), 4),
            "math_signal": math_signal,
            "short_signal": round(short_signal, 4),
            "long_signal": round(long_signal, 4),
        }

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _load_models_from_artifact(
        self,
        artifact_path: Path,
    ) -> Optional[tuple[BinaryClassifierModel, BinaryClassifierModel]]:
        if not artifact_path.exists():
            return None
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            models = payload.get("models", {})
            help_model = self._model_from_payload("help_seeking", models.get("help_seeking"))
            commitment_model = self._model_from_payload(
                "answer_commitment",
                models.get("answer_commitment"),
            )
            if help_model is None or commitment_model is None:
                return None
            return help_model, commitment_model
        except Exception:
            return None

    def _model_to_payload(self, model: BinaryClassifierModel) -> dict[str, Any]:
        return {
            "name": model.name,
            "positive_centroid": model.positive_centroid,
            "negative_centroid": model.negative_centroid,
            "positive_examples": list(model.positive_examples),
            "negative_examples": list(model.negative_examples),
            "feature_weights": model.feature_weights,
            "feature_midpoints": model.feature_midpoints,
            "intercept": model.intercept,
        }

    def _model_from_payload(
        self,
        name: str,
        payload: Any,
    ) -> Optional[BinaryClassifierModel]:
        if not isinstance(payload, dict):
            return None
        try:
            return BinaryClassifierModel(
                name=name,
                positive_centroid=[float(value) for value in payload["positive_centroid"]],
                negative_centroid=[float(value) for value in payload["negative_centroid"]],
                positive_examples=tuple(str(value) for value in payload["positive_examples"]),
                negative_examples=tuple(str(value) for value in payload["negative_examples"]),
                feature_weights={key: float(value) for key, value in payload["feature_weights"].items()},
                feature_midpoints={key: float(value) for key, value in payload["feature_midpoints"].items()},
                intercept=float(payload["intercept"]),
            )
        except Exception:
            return None
