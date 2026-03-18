# backend/state/feature_extractor.py
#
# Ham sinyallerden anlamlı özellik vektörü çıkarır.
# Kamera açık: CameraSignal + BehaviorSignal → FeatureVector
# Kamera kapalı: sadece BehaviorSignal → FeatureVector
#
# Sahip: K3
# Bağımlılıklar: schemas.py (K1+K2 ile Gün 1'de yazıldı)

from datetime import datetime
from typing import Optional

from backend.core.schemas import (
    BehaviorSignal,
    CameraSignal,
    FeatureVector,
    InputChannel,
)


class FeatureExtractor:
    """
    Her mesaj olayında çağrılır.
    State Model (K1) bu sınıfın çıktısını alır.

    Kullanım:
        extractor = FeatureExtractor()
        feature = extractor.extract(session_id, behavior, camera)
    """

    def __init__(self):
        # Oturum başına sayaçlar — session_id → değer
        self._retry_counts: dict[str, int] = {}
        self._last_message_times: dict[str, datetime] = {}
        self._topic_history: dict[str, list[str]] = {}

    def extract(
        self,
        session_id: str,
        message_content: str,
        message_timestamp: datetime,
        channel: InputChannel = InputChannel.TEXT,
        camera_signal: Optional[CameraSignal] = None,
    ) -> FeatureVector:
        """
        Ana metod. Her chat mesajında K3'ün endpoint'i bunu çağırır.

        Parametreler:
            session_id       : aktif oturum ID'si
            message_content  : kullanıcının yazdığı metin
            message_timestamp: mesajın geldiği zaman
            channel          : TEXT | IMAGE | CAMERA | VOICE
            camera_signal    : kamera açıksa MediaPipe çıktısı, yoksa None
        """
        # ── Davranışsal sinyaller ─────────────────────────────────────
        idle_time = self._calculate_idle_time(session_id, message_timestamp)
        response_time = self._estimate_response_time(message_content)
        retry_count = self._update_retry_count(session_id, message_content)
        topic = self._detect_topic(message_content)

        # ── Zaman damgasını güncelle ──────────────────────────────────
        self._last_message_times[session_id] = message_timestamp

        # ── Kamera sinyallerini çıkar (varsa) ────────────────────────
        cam = self._extract_camera_features(camera_signal)

        return FeatureVector(
            session_id=session_id,
            timestamp=message_timestamp,
            # Davranışsal
            idle_time_seconds=idle_time,
            retry_count=retry_count,
            response_time_seconds=response_time,
            message_length=len(message_content),
            topic=topic,
            # Kamera (None ise kamera kapalı — State Model bunu bilir)
            ear_score=cam.get("ear_score"),
            gaze_on_screen=cam.get("gaze_on_screen"),
            hand_on_chin=cam.get("hand_on_chin"),
            head_tilt_angle=cam.get("head_tilt_angle"),
        )

    def reset_session(self, session_id: str) -> None:
        """Oturum kapanınca sayaçları temizle."""
        self._retry_counts.pop(session_id, None)
        self._last_message_times.pop(session_id, None)
        self._topic_history.pop(session_id, None)

    # ── Private metodlar ─────────────────────────────────────────────

    def _calculate_idle_time(
        self, session_id: str, now: datetime
    ) -> float:
        """Son mesajdan bu yana geçen süreyi saniye cinsinden döner."""
        last = self._last_message_times.get(session_id)
        if last is None:
            return 0.0
        return (now - last).total_seconds()

    def _estimate_response_time(self, content: str) -> float:
        """
        Yazma hızını tahmin et: karakter sayısı / ortalama yazma hızı.
        Ortalama yazma hızı: dakikada 200 karakter (≈ 3.3 karakter/sn).
        Bu tahminî bir değer — gerçek yazma süresi client'tan alınamıyor.
        """
        chars_per_second = 3.3
        estimated = len(content) / chars_per_second
        return round(estimated, 2)

    def _update_retry_count(self, session_id: str, content: str) -> int:
        """
        Aynı konuda kaçıncı soru olduğunu takip eder.
        Basit heuristik: kısa mesajlar (< 50 karakter) retry sayılır.
        Gerçek topic detection Hafta 2'de LLM ile geliştirilecek.
        """
        if session_id not in self._retry_counts:
            self._retry_counts[session_id] = 0

        # Kısa ve soru içeren mesaj → retry olabilir
        is_short = len(content) < 50
        is_question = "?" in content or content.lower().startswith(
            ("ne", "neden", "nasıl", "niye", "kim", "hangi", "kaç")
        )

        if is_short and is_question:
            self._retry_counts[session_id] += 1
        else:
            # Uzun açıklamalı mesaj gelince sayacı sıfırla
            self._retry_counts[session_id] = 0

        return self._retry_counts[session_id]

    def _detect_topic(self, content: str) -> Optional[str]:
        """
        Mesajdan konu etiketi çıkar.
        Hafta 1: basit keyword matching.
        Hafta 2: LLM tabanlı konu tespiti buraya gelecek.
        """
        topic_keywords = {
            "matematik": ["türev", "integral", "limit", "fonksiyon", "denklem", "matris"],
            "fizik": ["kuvvet", "hız", "ivme", "enerji", "momentum", "termodinamik"],
            "kimya": ["mol", "reaksiyon", "asit", "baz", "element", "bileşik"],
            "biyoloji": ["hücre", "dna", "protein", "evrim", "enzim", "mitoz"],
            "programlama": ["kod", "fonksiyon", "algoritma", "döngü", "class", "api"],
        }

        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return topic

        return None

    def _extract_camera_features(
        self, signal: Optional[CameraSignal]
    ) -> dict:
        """CameraSignal'dan dict üretir. Signal yoksa boş dict döner."""
        if signal is None:
            return {}
        return {
            "ear_score": signal.ear_score,
            "gaze_on_screen": signal.gaze_on_screen,
            "hand_on_chin": signal.hand_on_chin,
            "head_tilt_angle": signal.head_tilt_angle,
        }