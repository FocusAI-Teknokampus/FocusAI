from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from backend.core.schemas import CameraSignal, FeatureVector, InputChannel


class FeatureExtractor:
    """
    Converts raw message activity into a richer feature vector for the state model.
    """

    def __init__(self):
        self._retry_counts: dict[str, int] = {}
        self._last_message_times: dict[str, datetime] = {}
        self._topic_history: dict[str, list[str]] = {}
        self._message_history: dict[str, list[dict[str, object]]] = {}

    def extract(
        self,
        session_id: str,
        message_content: str,
        message_timestamp: datetime,
        channel: InputChannel = InputChannel.TEXT,
        camera_signal: Optional[CameraSignal] = None,
    ) -> FeatureVector:
        topic = self._detect_topic(message_content)
        idle_time = self._calculate_idle_time(session_id, message_timestamp)
        response_time = self._estimate_response_time(message_content, channel)
        question_density = self._question_density(message_content)
        confusion_score = self._confusion_score(message_content)
        help_seeking_score = self._help_seeking_score(message_content)
        answer_commitment_score = self._answer_commitment_score(message_content)
        semantic_retry_score = self._semantic_retry_score(session_id, message_content, topic)
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
        self._remember_message(session_id, message_content, topic)

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
            semantic_retry_score=semantic_retry_score,
            help_seeking_score=help_seeking_score,
            answer_commitment_score=answer_commitment_score,
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

    def _help_seeking_score(self, content: str) -> float:
        normalized = self._normalize_text(content)
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
        if len(content) <= 40:
            score += 0.06
        return round(min(1.0, score), 3)

    def _answer_commitment_score(self, content: str) -> float:
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
        return round(max(0.0, min(1.0, score)), 3)

    def _semantic_retry_score(
        self,
        session_id: str,
        content: str,
        topic: Optional[str],
    ) -> float:
        history = self._message_history.get(session_id, [])
        if not history:
            return 0.0

        current_tokens = self._content_tokens(content)
        if not current_tokens:
            return 0.0

        best_similarity = 0.0
        for item in history[-4:]:
            previous_tokens = item.get("tokens", set())
            if not isinstance(previous_tokens, set) or not previous_tokens:
                continue

            overlap = len(current_tokens & previous_tokens)
            union = len(current_tokens | previous_tokens)
            similarity = overlap / union if union else 0.0
            if topic and item.get("topic") == topic:
                similarity += 0.12
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

    def _remember_message(self, session_id: str, content: str, topic: Optional[str]) -> None:
        message_history = self._message_history.setdefault(session_id, [])
        message_history.append(
            {
                "content": content,
                "topic": topic,
                "tokens": self._content_tokens(content),
            }
        )
        self._message_history[session_id] = message_history[-6:]

        topic_history = self._topic_history.setdefault(session_id, [])
        if topic:
            topic_history.append(topic)
        self._topic_history[session_id] = topic_history[-6:]

    def _normalize_text(self, content: str) -> str:
        lowered = content.lower()
        lowered = (
            lowered.replace("ı", "i")
            .replace("ş", "s")
            .replace("ğ", "g")
            .replace("ü", "u")
            .replace("ö", "o")
            .replace("ç", "c")
        )
        cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _content_tokens(self, content: str) -> set[str]:
        stopwords = {
            "ve",
            "ile",
            "ama",
            "bu",
            "bir",
            "icin",
            "mi",
            "mı",
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
