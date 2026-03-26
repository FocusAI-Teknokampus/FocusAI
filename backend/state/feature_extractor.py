from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from backend.core.config import settings
from backend.core.schemas import CameraSignal, FeatureVector, InputChannel
from backend.state.feature_classifier import FeatureIntentClassifier
from backend.state.semantic_features import (
    ANSWER_COMMITMENT_EXAMPLES,
    CONFIDENCE_EXAMPLES,
    DIRECT_ANSWER_EXAMPLES,
    FATIGUE_EXAMPLES,
    FRUSTRATION_EXAMPLES,
    HELP_SEEKING_EXAMPLES,
    OVERWHELM_EXAMPLES,
    SemanticFeatureProvider,
    URGENCY_EXAMPLES,
    cosine_similarity,
    normalize_text,
)


class FeatureExtractor:
    """
    Converts raw message activity into a richer feature vector for the state model.
    """

    def __init__(self, semantic_provider: Optional[SemanticFeatureProvider] = None):
        self._retry_counts: dict[str, int] = {}
        self._last_message_times: dict[str, datetime] = {}
        self._topic_history: dict[str, list[str]] = {}
        self._message_history: dict[str, list[dict[str, object]]] = {}
        self._semantic_provider = semantic_provider or SemanticFeatureProvider.from_settings()
        self._feature_classifier = (
            FeatureIntentClassifier(self._semantic_provider)
            if settings.feature_classifier_enabled
            else None
        )

    def extract(
        self,
        session_id: str,
        message_content: str,
        message_timestamp: datetime,
        channel: InputChannel = InputChannel.TEXT,
        camera_signal: Optional[CameraSignal] = None,
    ) -> FeatureVector:
        semantic = self._semantic_provider.analyze_text(message_content)
        topic = semantic.topic or self._detect_topic(message_content)
        idle_time = self._calculate_idle_time(session_id, message_timestamp)
        response_time = self._estimate_response_time(message_content, channel)
        question_density = self._question_density(message_content)
        confusion_score = self._confusion_score(message_content)
        help_seeking_semantic_score = self._help_seeking_semantic_score(message_content)
        help_seeking_classifier_score = self._help_seeking_classifier_score(message_content)
        help_seeking_score = self._help_seeking_score(
            message_content,
            semantic_score=help_seeking_semantic_score,
            classifier_score=help_seeking_classifier_score,
        )
        answer_commitment_semantic_score = self._answer_commitment_semantic_score(message_content)
        answer_commitment_classifier_score = self._answer_commitment_classifier_score(message_content)
        answer_commitment_score = self._answer_commitment_score(
            message_content,
            semantic_score=answer_commitment_semantic_score,
            classifier_score=answer_commitment_classifier_score,
        )
        fatigue_text_score = self._fatigue_text_score(message_content)
        frustration_text_score = self._frustration_text_score(message_content)
        confidence_text_score = self._confidence_text_score(message_content)
        overwhelm_text_score = self._overwhelm_text_score(message_content)
        urgency_text_score = self._urgency_text_score(message_content)
        semantic_retry_score = self._semantic_retry_score(
            session_id=session_id,
            content=message_content,
            topic=topic,
            embedding=semantic.embedding,
        )
        topic_stability = self._topic_stability(session_id, topic)
        retry_count = self._update_retry_count(
            session_id=session_id,
            content=message_content,
            topic=topic,
            question_density=question_density,
            confusion_score=confusion_score,
            semantic_retry_score=semantic_retry_score,
        )

        self._last_message_times[session_id] = message_timestamp
        self._remember_message(session_id, message_content, topic, semantic.embedding)

        cam = self._extract_camera_features(camera_signal)
        return FeatureVector(
            session_id=session_id,
            timestamp=message_timestamp,
            idle_time_seconds=idle_time,
            retry_count=retry_count,
            response_time_seconds=response_time,
            message_length=len(message_content),
            topic=topic,
            question_density=question_density,
            confusion_score=confusion_score,
            topic_stability=topic_stability,
            topic_confidence=semantic.topic_score if topic == semantic.topic else 0.0,
            semantic_retry_score=semantic_retry_score,
            help_seeking_score=help_seeking_score,
            help_seeking_semantic_score=help_seeking_semantic_score,
            help_seeking_classifier_score=help_seeking_classifier_score,
            answer_commitment_score=answer_commitment_score,
            answer_commitment_semantic_score=answer_commitment_semantic_score,
            answer_commitment_classifier_score=answer_commitment_classifier_score,
            fatigue_text_score=fatigue_text_score,
            frustration_text_score=frustration_text_score,
            confidence_text_score=confidence_text_score,
            overwhelm_text_score=overwhelm_text_score,
            urgency_text_score=urgency_text_score,
            ear_score=cam.get("ear_score"),
            gaze_on_screen=cam.get("gaze_on_screen"),
            hand_on_chin=cam.get("hand_on_chin"),
            head_tilt_angle=cam.get("head_tilt_angle"),
        )

    def reset_session(self, session_id: str) -> None:
        self._retry_counts.pop(session_id, None)
        self._last_message_times.pop(session_id, None)
        self._topic_history.pop(session_id, None)
        self._message_history.pop(session_id, None)

    def _calculate_idle_time(self, session_id: str, now: datetime) -> float:
        last = self._last_message_times.get(session_id)
        if last is None:
            return 0.0
        return (now - last).total_seconds()

    def _estimate_response_time(self, content: str, channel: InputChannel) -> float:
        word_count = max(1, len(re.findall(r"\w+", content, flags=re.UNICODE)))
        punctuation_penalty = content.count("?") + content.count("!") + content.count(":")
        complexity_bonus = len(re.findall(r"[0-9_=<>/*+\-]", content))
        chars_per_second = {
            InputChannel.TEXT: 3.8,
            InputChannel.IMAGE: 4.2,
            InputChannel.CAMERA: 3.8,
            InputChannel.VOICE: 7.0,
            InputChannel.SYSTEM: 4.0,
        }.get(channel, 3.8)
        estimated = (
            (len(content) / chars_per_second)
            + (word_count * 0.35)
            + min(4.0, punctuation_penalty * 0.4)
            + min(5.0, complexity_bonus * 0.12)
        )
        return round(estimated, 2)

    def _update_retry_count(
        self,
        session_id: str,
        content: str,
        topic: Optional[str],
        question_density: float,
        confusion_score: float,
        semantic_retry_score: float,
    ) -> int:
        current = self._retry_counts.get(session_id, 0)
        lowered = self._normalize_text(content)
        previous_topics = self._topic_history.get(session_id, [])
        same_topic_streak = bool(topic and previous_topics and previous_topics[-1] == topic)
        starts_like_question = lowered.startswith(
            ("ne ", "neden", "nasil", "niye", "kim", "hangi", "kac", "why ", "what ", "how ")
        )
        is_retry = any(
            [
                semantic_retry_score >= 0.62,
                confusion_score >= 0.55 and same_topic_streak,
                len(content) <= 80 and question_density >= 0.2 and same_topic_streak,
                len(content) <= 40 and starts_like_question and current >= 1,
            ]
        )
        reset_signal = len(content) >= 140 and confusion_score < 0.2 and question_density < 0.15

        if is_retry:
            current += 1
        elif reset_signal:
            current = 0
        elif current > 0:
            current = max(0, current - 1)

        self._retry_counts[session_id] = current
        return current

    def _detect_topic(self, content: str) -> Optional[str]:
        topic_keywords = {
            "matematik": ["turev", "integral", "limit", "fonksiyon", "denklem", "matris", "geometri"],
            "fizik": ["kuvvet", "hiz", "ivme", "enerji", "momentum", "termodinamik", "elektrik"],
            "kimya": ["mol", "reaksiyon", "asit", "baz", "element", "bilesik", "orbital"],
            "biyoloji": ["hucre", "dna", "protein", "evrim", "enzim", "mitoz", "ekosistem"],
            "programlama": ["kod", "fonksiyon", "algoritma", "dongu", "class", "api", "bug", "stack", "python"],
        }

        normalized = self._normalize_text(content)
        scores = {
            topic: sum(1 for keyword in keywords if keyword in normalized)
            for topic, keywords in topic_keywords.items()
        }
        best_topic, best_score = max(scores.items(), key=lambda item: item[1], default=(None, 0))
        if best_topic and best_score > 0:
            return best_topic
        return None

    def _question_density(self, content: str) -> float:
        tokens = max(1, len(re.findall(r"\w+", content, flags=re.UNICODE)))
        prompt_like_starts = 1 if self._normalize_text(content).startswith(
            ("ne ", "neden", "nasil", "hangi", "kac", "why ", "what ", "how ")
        ) else 0
        density = ((content.count("?") * 2) + prompt_like_starts) / tokens
        return round(min(1.0, density), 3)

    def _confusion_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        confusion_patterns = [
            "anlamadim",
            "anlayamiyorum",
            "tam anlayamiyorum",
            "anlamakta zorlaniyorum",
            "olmadi",
            "takildim",
            "karisti",
            "yanlis",
            "hata",
            "neden boyle",
            "emin degilim",
            "cozemiyorum",
            "yardim",
            "help",
            "stuck",
            "not working",
        ]
        score = 0.0
        for pattern in confusion_patterns:
            if pattern in normalized:
                score += 0.22
        if "?" in content:
            score += 0.1
        if len(content) <= 20:
            score += 0.08
        return round(min(1.0, score), 3)

    def _help_seeking_score(
        self,
        content: str,
        semantic_score: Optional[float] = None,
        classifier_score: Optional[float] = None,
    ) -> float:
        normalized = self._normalize_text(content)
        interrogatives = ("neden", "nasil", "hangi", "nereye", "ne zaman", "kac", "niye")
        patterns = [
            "yardim",
            "ipucu",
            "coz",
            "cevabi",
            "direkt soyle",
            "anlat",
            "goster",
            "help",
            "explain",
            "solution",
        ]
        score = 0.0
        for pattern in patterns:
            if pattern in normalized:
                score += 0.14
        if "?" in content:
            score += 0.08
        if any(word in normalized for word in interrogatives):
            score += 0.08
        if len(content) <= 40:
            score += 0.06
        lexical = min(1.0, score)
        semantic = (
            semantic_score
            if semantic_score is not None
            else self._help_seeking_semantic_score(content)
        )
        enriched = lexical + (semantic * 0.25)
        return round(
            min(1.0, self._blend_classifier_boost(enriched, classifier_score)),
            3,
        )

    def _help_seeking_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=HELP_SEEKING_EXAMPLES,
            negative_examples=ANSWER_COMMITMENT_EXAMPLES,
            floor=0.24,
            ceiling=0.76,
        )

    def _help_seeking_classifier_score(self, content: str) -> float:
        if self._feature_classifier is None:
            return 0.0
        return self._feature_classifier.score_help_seeking(content)

    def _answer_commitment_score(
        self,
        content: str,
        semantic_score: Optional[float] = None,
        classifier_score: Optional[float] = None,
    ) -> float:
        normalized = self._normalize_text(content)
        effort_patterns = [
            "denedim",
            "boyle yaptim",
            "su adimi yaptim",
            "buldum",
            "hesapladim",
            "dusundum",
            "deniyorum",
            "bence",
            "kodum",
            "ornek",
            "adim",
            "deneme",
            "yaklasim",
            "i tried",
            "my attempt",
            "i got",
        ]
        shortcut_patterns = [
            "sadece cevabi",
            "direkt cevabi",
            "cevabi soyle",
            "just answer",
            "just give",
        ]

        score = 0.0
        for pattern in effort_patterns:
            if pattern in normalized:
                score += 0.16
        if re.search(r"[0-9_=<>/*+\-]", content):
            score += 0.1
        if len(content) >= 70:
            score += 0.08
        if any(pattern in normalized for pattern in shortcut_patterns):
            score -= 0.25
        lexical = max(0.0, min(1.0, score))
        semantic = (
            semantic_score
            if semantic_score is not None
            else self._answer_commitment_semantic_score(content)
        )
        enriched = lexical + (semantic * 0.2)
        return round(
            max(0.0, min(1.0, self._blend_classifier_boost(enriched, classifier_score))),
            3,
        )

    def _answer_commitment_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=ANSWER_COMMITMENT_EXAMPLES,
            negative_examples=DIRECT_ANSWER_EXAMPLES,
            floor=0.2,
            ceiling=0.74,
        )

    def _answer_commitment_classifier_score(self, content: str) -> float:
        if self._feature_classifier is None:
            return 0.0
        return self._feature_classifier.score_answer_commitment(content)

    def _blend_classifier_boost(
        self,
        base_score: float,
        classifier_score: Optional[float],
    ) -> float:
        if classifier_score is None:
            return base_score
        boost = max(0.0, classifier_score - base_score) * settings.feature_classifier_blend_weight
        return base_score + boost

    def _fatigue_text_score(self, content: str) -> float:
        lexical = self._fatigue_text_lexical_score(content)
        semantic = self._fatigue_text_semantic_score(content)
        score = lexical + (semantic * 0.35)
        return round(min(1.0, score), 3)

    def _fatigue_text_lexical_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        patterns = [
            ("cok yorgunum", 0.42),
            ("yorgunum", 0.28),
            ("yoruldum", 0.34),
            ("kafam almiyor", 0.42),
            ("dinlenmem lazim", 0.4),
            ("uykum var", 0.3),
            ("gozum kapaniyor", 0.46),
            ("odaklanamiyorum", 0.28),
            ("beynim durdu", 0.42),
            ("anlamakta zorlaniyorum", 0.18),
            ("anlamakta zorlaniyorum cunku yoruldum", 0.24),
            ("anlayamiyorum cunku yoruldum", 0.24),
            ("sonra devam edelim", 0.14),
            ("mola vermem lazim", 0.22),
            ("ara vermem lazim", 0.22),
        ]
        score = 0.0
        matched_patterns: list[str] = []
        for pattern, weight in patterns:
            if pattern in normalized:
                score += weight
                matched_patterns.append(pattern)

        has_explicit_fatigue = any(
            token in normalized
            for token in (
                "yoruldum",
                "yorgunum",
                "kafam almiyor",
                "dinlenmem lazim",
                "uykum var",
                "gozum kapaniyor",
                "beynim durdu",
            )
        )
        has_cognitive_load = any(
            token in normalized
            for token in (
                "odaklanamiyorum",
                "anlamakta zorlaniyorum",
                "anlayamiyorum",
            )
        )
        if has_explicit_fatigue and has_cognitive_load:
            score += 0.18
        if has_explicit_fatigue and any(token in normalized for token in ("sonra devam edelim", "mola", "ara ver")):
            score += 0.08
        if len(matched_patterns) >= 2:
            score += 0.06
        return min(1.0, score)

    def _fatigue_text_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=FATIGUE_EXAMPLES,
            negative_examples=ANSWER_COMMITMENT_EXAMPLES + DIRECT_ANSWER_EXAMPLES,
            floor=0.18,
            ceiling=0.72,
        )

    def _frustration_text_score(self, content: str) -> float:
        lexical = self._frustration_text_lexical_score(content)
        semantic = self._frustration_text_semantic_score(content)
        score = lexical + (semantic * 0.32)
        return round(min(1.0, score), 3)

    def _frustration_text_lexical_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        patterns = [
            ("of ya", 0.34),
            ("biktim", 0.34),
            ("sinir bozucu", 0.36),
            ("sinir oldum", 0.38),
            ("yeter artik", 0.4),
            ("deliriyorum", 0.42),
            ("delirtecek", 0.38),
            ("beni delirtti", 0.42),
            ("sacma", 0.22),
            ("cok sacma", 0.3),
            ("nefret ettim", 0.36),
            ("cildiracagim", 0.42),
            ("cok bozucu", 0.32),
            ("asiri sinir bozucu", 0.42),
        ]
        score = 0.0
        matched_patterns: list[str] = []
        for pattern, weight in patterns:
            if pattern in normalized:
                score += weight
                matched_patterns.append(pattern)

        if "!" in content:
            score += min(0.12, content.count("!") * 0.06)
        if len(matched_patterns) >= 2:
            score += 0.1
        if any(token in normalized for token in ("olmadi", "cozemiyorum", "takildim")) and matched_patterns:
            score += 0.08
        return min(1.0, score)

    def _frustration_text_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=FRUSTRATION_EXAMPLES,
            negative_examples=ANSWER_COMMITMENT_EXAMPLES + FATIGUE_EXAMPLES,
            floor=0.2,
            ceiling=0.74,
        )

    def _confidence_text_score(self, content: str) -> float:
        lexical = self._confidence_text_lexical_score(content)
        semantic = self._confidence_text_semantic_score(content)
        score = lexical + (semantic * 0.28)
        return round(min(1.0, score), 3)

    def _confidence_text_lexical_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        positive_patterns = [
            ("eminim", 0.34),
            ("bundan eminim", 0.42),
            ("bence", 0.16),
            ("cozdum", 0.26),
            ("buldum", 0.24),
            ("mantigi oturttum", 0.4),
            ("dogru gibi", 0.18),
            ("sonuc", 0.1),
            ("cikiyor", 0.1),
            ("bunu anladim", 0.3),
            ("emin gibiyim", 0.22),
        ]
        negative_patterns = (
            "emin degilim",
            "galiba degil",
            "sanirim yanlis",
            "pek emin degilim",
        )
        score = 0.0
        for pattern, weight in positive_patterns:
            if pattern in normalized:
                score += weight
        if re.search(r"[0-9_=<>/*+\-]", content):
            score += 0.08
        if any(pattern in normalized for pattern in negative_patterns):
            score -= 0.3
        return max(0.0, min(1.0, score))

    def _confidence_text_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=CONFIDENCE_EXAMPLES,
            negative_examples=HELP_SEEKING_EXAMPLES + FATIGUE_EXAMPLES + OVERWHELM_EXAMPLES,
            floor=0.22,
            ceiling=0.74,
        )

    def _overwhelm_text_score(self, content: str) -> float:
        lexical = self._overwhelm_text_lexical_score(content)
        semantic = self._overwhelm_text_semantic_score(content)
        score = lexical + (semantic * 0.34)
        return round(min(1.0, score), 3)

    def _overwhelm_text_lexical_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        patterns = [
            ("bunaldim", 0.38),
            ("bunaliyorum", 0.34),
            ("cok fazla geldi", 0.42),
            ("yetisemiyorum", 0.38),
            ("her sey birbirine girdi", 0.44),
            ("kafam cok doldu", 0.42),
            ("nereden baslayacagimi bilmiyorum", 0.38),
            ("ust uste geliyor", 0.34),
            ("karmasa oldu", 0.28),
            ("hepsi birikti", 0.34),
        ]
        score = 0.0
        matched = 0
        for pattern, weight in patterns:
            if pattern in normalized:
                score += weight
                matched += 1
        if matched >= 2:
            score += 0.08
        if any(token in normalized for token in ("yoruldum", "yorgunum", "dinlenmem lazim")):
            score += 0.08
        return min(1.0, score)

    def _overwhelm_text_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=OVERWHELM_EXAMPLES,
            negative_examples=CONFIDENCE_EXAMPLES + DIRECT_ANSWER_EXAMPLES,
            floor=0.18,
            ceiling=0.72,
        )

    def _urgency_text_score(self, content: str) -> float:
        lexical = self._urgency_text_lexical_score(content)
        semantic = self._urgency_text_semantic_score(content)
        score = lexical + (semantic * 0.3)
        return round(min(1.0, score), 3)

    def _urgency_text_lexical_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
        patterns = [
            ("acil", 0.34),
            ("acele", 0.28),
            ("acelem var", 0.42),
            ("sinavim var", 0.42),
            ("hizlica", 0.28),
            ("cabuk", 0.24),
            ("hemen", 0.2),
            ("kisaca", 0.22),
            ("cok vaktim yok", 0.4),
            ("direkt ozet", 0.3),
            ("kisa cevap", 0.24),
        ]
        score = 0.0
        for pattern, weight in patterns:
            if pattern in normalized:
                score += weight
        if len(content) <= 32:
            score += 0.06
        return min(1.0, score)

    def _urgency_text_semantic_score(self, content: str) -> float:
        return self._semantic_provider.example_similarity_score(
            text=content,
            positive_examples=URGENCY_EXAMPLES,
            negative_examples=ANSWER_COMMITMENT_EXAMPLES + CONFIDENCE_EXAMPLES,
            floor=0.2,
            ceiling=0.74,
        )

    def _semantic_retry_score(
        self,
        session_id: str,
        content: str,
        topic: Optional[str],
        embedding: list[float],
    ) -> float:
        history = self._message_history.get(session_id, [])
        if not history:
            return 0.0

        current_tokens = self._content_tokens(content)
        if not current_tokens and not embedding:
            return 0.0

        best_similarity = 0.0
        for item in history[-5:]:
            previous_embedding = item.get("embedding", [])
            semantic_similarity = 0.0
            if isinstance(previous_embedding, list):
                semantic_similarity = cosine_similarity(embedding, previous_embedding)

            previous_tokens = item.get("tokens", set())
            if not isinstance(previous_tokens, set):
                previous_tokens = set()

            overlap = len(current_tokens & previous_tokens)
            union = len(current_tokens | previous_tokens)
            lexical_similarity = overlap / union if union else 0.0
            similarity = max(semantic_similarity, lexical_similarity)

            if topic and item.get("topic") == topic:
                similarity += 0.14
            best_similarity = max(best_similarity, similarity)

        return round(min(1.0, best_similarity), 3)

    def _topic_stability(self, session_id: str, topic: Optional[str]) -> float:
        recent_topics = [value for value in self._topic_history.get(session_id, []) if value]
        if not recent_topics:
            return 1.0 if topic else 0.5
        if topic is None:
            return 0.4

        window = recent_topics[-4:]
        same_count = sum(1 for value in window if value == topic)
        return round(min(1.0, same_count / max(1, len(window))), 3)

    def _remember_message(
        self,
        session_id: str,
        content: str,
        topic: Optional[str],
        embedding: list[float],
    ) -> None:
        message_history = self._message_history.setdefault(session_id, [])
        message_history.append(
            {
                "content": content,
                "topic": topic,
                "tokens": self._content_tokens(content),
                "embedding": embedding,
            }
        )
        self._message_history[session_id] = message_history[-6:]

        topic_history = self._topic_history.setdefault(session_id, [])
        if topic:
            topic_history.append(topic)
        self._topic_history[session_id] = topic_history[-6:]

    def _normalize_text(self, content: str) -> str:
        return normalize_text(content)

    def _content_tokens(self, content: str) -> set[str]:
        stopwords = {
            "ve",
            "ile",
            "ama",
            "bu",
            "bir",
            "icin",
            "mi",
            "mu",
            "the",
            "is",
            "a",
            "an",
            "to",
            "of",
            "da",
            "de",
        }
        normalized = self._normalize_text(content)
        return {
            token
            for token in normalized.split()
            if len(token) >= 3 and token not in stopwords
        }

    def _extract_camera_features(self, signal: Optional[CameraSignal]) -> dict:
        if signal is None:
            return {}
        return {
            "ear_score": signal.ear_score,
            "gaze_on_screen": signal.gaze_on_screen,
            "hand_on_chin": signal.hand_on_chin,
            "head_tilt_angle": signal.head_tilt_angle,
        }
