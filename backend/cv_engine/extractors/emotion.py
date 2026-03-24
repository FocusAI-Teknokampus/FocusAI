"""
CV Engine / Extractors — Duygu Tanıma
DeepFace kullanarak yüz ifadesinden duygu tespit eder.
Ağır bir model — her frame'de değil, her N saniyede çalıştır.
"""

import time
import numpy as np
from typing import Optional

# Kaç saniyede bir analiz yapılsın (DeepFace yavaş)
EMOTION_INTERVAL_SEC = 3.0

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Dikkat dağıtan duygular
DISTRACTED_EMOTIONS = {"angry", "sad", "fear", "disgust"}


class EmotionExtractor:
    """
    pipeline.py tarafından çağrılır.
    DeepFace her frame'de çalıştırılmaz — interval kontrolü içeride yapılır.
    """

    def __init__(self, interval_sec: float = EMOTION_INTERVAL_SEC):
        self._interval = interval_sec
        self._last_run: float = 0.0
        self._last_result: Optional[dict] = None
        self._available = False

    def initialize(self) -> bool:
        """DeepFace import kontrolü — yüklü değilse sessizce False döner."""
        try:
            import deepface  # noqa: F401
            self._available = True
            return True
        except ImportError:
            print("[EmotionExtractor] DeepFace bulunamadı, duygu analizi devre dışı.")
            return False

    def extract(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        frame_bgr: BGR formatında numpy array (cv2 çıktısı, RGB'ye çevirme!)
        Interval dolmadıysa son sonucu döndürür.
        DeepFace yoksa None döner.
        """
        if not self._available:
            return None

        now = time.time()
        if now - self._last_run < self._interval:
            return self._last_result  # Son sonucu tekrar gönder

        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # result liste veya dict olabilir
            if isinstance(result, list):
                result = result[0]

            dominant = result.get("dominant_emotion", "neutral")
            emotions = result.get("emotion", {})

            self._last_result = {
                "dominant_emotion":   dominant,
                "emotion_scores":     {k: round(v, 2) for k, v in emotions.items()},
                "is_distracted_emotion": dominant in DISTRACTED_EMOTIONS,
            }
            self._last_run = now
            return self._last_result

        except Exception as e:
            print(f"[EmotionExtractor] Analiz hatası: {e}")
            return self._last_result  # Hata olsa bile son sonucu döndür

    def release(self):
        self._available = False
        self._last_result = None
