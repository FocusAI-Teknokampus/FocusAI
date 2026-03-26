from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Optional

from backend.core.config import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass(frozen=True)
class TopicPrototype:
    label: str
    description: str
    examples: tuple[str, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return (self.description, *self.examples)


@dataclass(frozen=True)
class SemanticAnalysis:
    embedding: list[float]
    topic: Optional[str]
    topic_score: float


DEFAULT_TOPIC_BANK: tuple[TopicPrototype, ...] = (
    TopicPrototype(
        label="matematik",
        description=(
            "Matematik sorulari: turev, integral, limit, diferansiyel, lineer cebir, "
            "fonksiyon davranisi ve denklem cozumleri."
        ),
        examples=(
            "Bir fonksiyonun turevini nasil alirim?",
            "Integralde parcali integrasyon uyguluyorum.",
            "Diferansiyel denklemde baslangic kosulu nasil kullanilir?",
        ),
    ),
    TopicPrototype(
        label="fizik",
        description=(
            "Fizik sorulari: kuvvet, hareket, ivme, enerji, elektrik, dalga ve "
            "termodinamik yorumlari."
        ),
        examples=(
            "Net kuvvet sifirsa cisim neden hizini korur?",
            "Elektrik devresinde direncler nasil toplanir?",
            "Termodinamikte isi ve is arasindaki fark nedir?",
        ),
    ),
    TopicPrototype(
        label="kimya",
        description=(
            "Kimya sorulari: mol, reaksiyon, baglar, asit-baz, organik yapilar ve "
            "atomik orbital kavramlari."
        ),
        examples=(
            "Mol hesabinda avogadro sayisini ne zaman kullanirim?",
            "Asit baz titrasyonunda esitlik noktasi nasil bulunur?",
            "Orbitaller arasindaki enerji farki neden degisir?",
        ),
    ),
    TopicPrototype(
        label="biyoloji",
        description=(
            "Biyoloji sorulari: hucre, dna, protein sentezi, enzimler, genetik, "
            "evrim ve ekosistem iliskileri."
        ),
        examples=(
            "Protein sentezinde mrna ve trna ne yapiyor?",
            "Enzim hizi sicakliktan neden etkileniyor?",
            "Mitoz ile mayoz arasindaki temel farklar neler?",
        ),
    ),
    TopicPrototype(
        label="programlama",
        description=(
            "Programlama sorulari: kod, algoritma, api, hata ayiklama, class, fonksiyon, "
            "veri yapilari ve yazilim akisi."
        ),
        examples=(
            "Python API istegi neden 500 hatasi veriyor?",
            "Bu algoritmanin zaman karmasikligi nasil bulunur?",
            "Fonksiyonda state neden bekledigim gibi guncellenmiyor?",
        ),
    ),
)


HELP_SEEKING_EXAMPLES: tuple[str, ...] = (
    "Burada takildim, nasil ilerlemeliyim?",
    "Bana kisa bir ipucu verir misin?",
    "Bu adimi aciklar misin?",
    "Neden boyle oldugunu anlamadim.",
)


FATIGUE_EXAMPLES: tuple[str, ...] = (
    "Yoruldum, su an odaklanamiyorum.",
    "Cok yorgunum, sonra devam edelim.",
    "Kafam almiyor, biraz dinlenmem lazim.",
    "Gozum kapaniyor, anlamakta zorlanmaya basladim.",
)


FRUSTRATION_EXAMPLES: tuple[str, ...] = (
    "Of ya, yine olmadi, bu cok sinir bozucu.",
    "Biktim, ayni yerde kaliyorum.",
    "Yeter artik, bu soru beni delirtti.",
    "Asiri sacma geliyor, neden hala cozemiyorum?",
)


CONFIDENCE_EXAMPLES: tuple[str, ...] = (
    "Bence cozdum, sonuc 2 cikiyor.",
    "Bu adimdan eminim, zincir kurali gerekiyor.",
    "Cozumu buldum gibi, son kontrolu yapmak istiyorum.",
    "Burada mantigi oturttum, devam edebilirim.",
)


OVERWHELM_EXAMPLES: tuple[str, ...] = (
    "Bunaldim, konu ust uste biniyor.",
    "Cok fazla geldi, nereden baslayacagimi bilmiyorum.",
    "Her sey birbirine girdi, yetisemiyorum.",
    "Kafam cok doldu, parcalara ayirmam lazim.",
)


URGENCY_EXAMPLES: tuple[str, ...] = (
    "Acele cevap lazim, sinavim var.",
    "Hizlica kisaca anlatir misin?",
    "Cok vaktim yok, direkt ozet gec.",
    "Hemen bakmam gerekiyor, cabuk bir yon ver.",
)


ANSWER_COMMITMENT_EXAMPLES: tuple[str, ...] = (
    "Once zincir kuralini denedim ama ikinci adimda hata yaptim.",
    "Benim denemem su sekilde, yanlis olan yeri bulmaya calisiyorum.",
    "Ilk olarak denklemi sadelestirdim, sonra integralde takildim.",
    "Kendi cozumumde bu sonuca ulastim ama emin degilim.",
)


DIRECT_ANSWER_EXAMPLES: tuple[str, ...] = (
    "Direkt cevabi soyle.",
    "Sadece sonucu ver.",
    "Cozumu yaz gec.",
    "Just give me the answer.",
)


class EmbeddingBackend:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        normalized = normalize_text(text)
        vector = [0.0] * self.dimensions
        tokens = normalized.split()
        if not tokens:
            return vector

        for token in tokens:
            self._add_feature(vector, f"tok:{token}", 1.0 + min(1.5, len(token) / 8))
            for size in (3, 4):
                if len(token) < size:
                    continue
                for index in range(len(token) - size + 1):
                    gram = token[index : index + size]
                    self._add_feature(vector, f"ng:{gram}", 0.22)

        for left, right in zip(tokens, tokens[1:]):
            self._add_feature(vector, f"bg:{left}_{right}", 0.4)

        return l2_normalize(vector)

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "little") % self.dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[position] += sign * weight


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str, api_key: str):
        if OpenAI is None:
            raise RuntimeError("OpenAI dependency is not installed.")
        if not api_key:
            raise RuntimeError("OpenAI API key is not configured.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [l2_normalize(item.embedding) for item in response.data]


class SemanticFeatureProvider:
    def __init__(
        self,
        backend: EmbeddingBackend,
        fallback_backend: Optional[EmbeddingBackend] = None,
        topic_bank: tuple[TopicPrototype, ...] = DEFAULT_TOPIC_BANK,
    ):
        self._backend = backend
        self._fallback_backend = fallback_backend or HashingEmbeddingBackend(
            dimensions=settings.semantic_embedding_dimensions
        )
        self._topic_bank = topic_bank
        self._embedding_cache: dict[str, list[float]] = {}
        self._topic_centroids: Optional[dict[str, list[float]]] = None

    @classmethod
    def from_settings(cls) -> "SemanticFeatureProvider":
        dimensions = settings.semantic_embedding_dimensions
        fallback = HashingEmbeddingBackend(dimensions=dimensions)
        provider = settings.semantic_embedding_provider.lower()

        if provider == "local":
            return cls(backend=fallback, fallback_backend=fallback)

        if provider in {"auto", "openai"}:
            try:
                backend = OpenAIEmbeddingBackend(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key,
                )
                return cls(backend=backend, fallback_backend=fallback)
            except Exception:
                if provider == "openai":
                    raise

        return cls(backend=fallback, fallback_backend=fallback)

    def analyze_text(self, text: str) -> SemanticAnalysis:
        embedding = self.embed_text(text)
        topic, score = self.detect_topic_from_embedding(embedding)
        return SemanticAnalysis(
            embedding=embedding,
            topic=topic,
            topic_score=score,
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._embedding_cache]
        if missing:
            self._populate_cache(missing)
        return [self._embedding_cache.get(text, []) for text in texts]

    def detect_topic(self, text: str) -> tuple[Optional[str], float]:
        return self.detect_topic_from_embedding(self.embed_text(text))

    def detect_topic_from_embedding(
        self,
        embedding: list[float],
    ) -> tuple[Optional[str], float]:
        if not embedding or not any(abs(value) > 0.0 for value in embedding):
            return None, 0.0

        centroids = self._get_topic_centroids()
        best_label = None
        best_score = 0.0
        for label, centroid in centroids.items():
            score = cosine_similarity(embedding, centroid)
            if score > best_score:
                best_label = label
                best_score = score

        if best_score < settings.semantic_topic_threshold:
            return None, round(best_score, 3)
        return best_label, round(best_score, 3)

    def example_similarity_score(
        self,
        text: str,
        positive_examples: tuple[str, ...],
        negative_examples: tuple[str, ...] = (),
        floor: float = 0.22,
        ceiling: float = 0.74,
    ) -> float:
        base_embedding = self.embed_text(text)
        if not base_embedding or not any(abs(value) > 0.0 for value in base_embedding):
            return 0.0

        positive = self._best_similarity(base_embedding, positive_examples)
        negative = self._best_similarity(base_embedding, negative_examples)
        raw = max(0.0, positive - (negative * 0.55))
        return round(scale_similarity(raw, floor=floor, ceiling=ceiling), 3)

    def _best_similarity(
        self,
        base_embedding: list[float],
        examples: tuple[str, ...],
    ) -> float:
        if not examples:
            return 0.0
        embeddings = self.embed_texts(list(examples))
        return max(cosine_similarity(base_embedding, other) for other in embeddings)

    def _populate_cache(self, texts: list[str]) -> None:
        try:
            embeddings = self._backend.embed_texts(texts)
        except Exception:
            if self._backend is self._fallback_backend:
                raise
            self._backend = self._fallback_backend
            self._topic_centroids = None
            embeddings = self._backend.embed_texts(texts)

        for text, embedding in zip(texts, embeddings):
            self._embedding_cache[text] = l2_normalize(list(embedding))

    def _get_topic_centroids(self) -> dict[str, list[float]]:
        if self._topic_centroids is not None:
            return self._topic_centroids

        centroids: dict[str, list[float]] = {}
        for prototype in self._topic_bank:
            embeddings = self.embed_texts(list(prototype.texts))
            centroids[prototype.label] = average_vectors(embeddings)
        self._topic_centroids = centroids
        return centroids


def normalize_text(content: str) -> str:
    lowered = content.lower()
    transliteration = str.maketrans(
        {
            "\u0131": "i",
            "\u0130": "i",
            "\u015f": "s",
            "\u015e": "s",
            "\u011f": "g",
            "\u011e": "g",
            "\u00fc": "u",
            "\u00dc": "u",
            "\u00f6": "o",
            "\u00d6": "o",
            "\u00e7": "c",
            "\u00c7": "c",
        }
    )
    lowered = lowered.translate(transliteration)
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    averaged = [0.0] * dimensions
    for vector in vectors:
        for index, value in enumerate(vector):
            averaged[index] += value
    return l2_normalize([value / len(vectors) for value in averaged])


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    score = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(1.0, score))


def scale_similarity(score: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    scaled = (score - floor) / (ceiling - floor)
    return max(0.0, min(1.0, scaled))
