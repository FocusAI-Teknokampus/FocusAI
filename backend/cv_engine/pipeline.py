"""
CV Engine — Ana Pipeline
Kamerayı açar, her frame'i extractors'a gönderir,
CameraSignal üretir ve FrameBuffer'a yazar.
Kamera yoksa FallbackSignal döner — sistem çökmez.
"""

import time
import threading
import numpy as np
from typing import Optional, Callable

from .buffer import FrameBuffer
from .extractors import GazeExtractor, GestureExtractor, EmotionExtractor


# --------------------------------------------------------------------------
# CameraSignal — Feature Extractor'a giden ana veri yapısı
# --------------------------------------------------------------------------

class CameraSignal:
    """
    pipeline.py'ın ürettiği sinyal.
    Feature Extractor (state/feature_extractor.py) bu nesneyi alır.
    """

    def __init__(
        self,
        active: bool,
        gaze: Optional[dict],
        gesture: Optional[dict],
        emotion: Optional[dict],
        timestamp: float,
        frame_id: int,
        processing_ms: float,
    ):
        self.active = active          # Kamera çalışıyor mu?
        self.gaze = gaze              # GazeExtractor çıktısı
        self.gesture = gesture        # GestureExtractor çıktısı
        self.emotion = emotion        # EmotionExtractor çıktısı (3 sn'de bir)
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.processing_ms = processing_ms

    @property
    def attention_score(self) -> Optional[float]:
        """
        0.0–1.0 arası dikkat skoru.
        Kamera kapalıysa None — Feature Extractor bunu handle eder.
        """
        if not self.active or self.gaze is None or self.gesture is None:
            return None

        score = 1.0

        # Göz sinyalleri
        if self.gaze.get("is_drowsy"):
            score -= 0.4
        elif self.gaze.get("ear_avg", 0.3) < 0.25:
            score -= 0.2
        if self.gaze.get("gaze_direction") in ("left", "right", "down"):
            score -= 0.2

        # Baş pozu
        if self.gesture.get("is_head_down"):
            score -= 0.3
        if self.gesture.get("is_head_turned"):
            score -= 0.2

        # El sinyalleri
        if self.gesture.get("hand_on_chin") or self.gesture.get("hand_on_face"):
            score -= 0.1

        # Duygu (opsiyonel)
        if self.emotion and self.emotion.get("is_distracted_emotion"):
            score -= 0.1

        return round(max(0.0, min(1.0, score)), 3)

    def to_dict(self) -> dict:
        return {
            "active":          self.active,
            "gaze":            self.gaze,
            "gesture":         self.gesture,
            "emotion":         self.emotion,
            "timestamp":       self.timestamp,
            "frame_id":        self.frame_id,
            "processing_ms":   self.processing_ms,
            "attention_score": self.attention_score,
        }


def _fallback(reason: str = "unavailable") -> CameraSignal:
    return CameraSignal(
        active=False,
        gaze=None,
        gesture=None,
        emotion=None,
        timestamp=time.time(),
        frame_id=0,
        processing_ms=0.0,
    )


# --------------------------------------------------------------------------
# CVPipeline
# --------------------------------------------------------------------------

class CVPipeline:
    """
    Tek bir nesne — uygulama başlarken bir kez oluşturulur.

    Kullanım (main.py veya feature_extractor.py):
        pipeline = CVPipeline()
        pipeline.start()
        signal = pipeline.latest()   # Feature Extractor bunu çağırır
        pipeline.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 15,
        use_emotion: bool = False,      # DeepFace ağır, varsayılan kapalı
        on_signal: Optional[Callable[[CameraSignal], None]] = None,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.use_emotion = use_emotion
        self.on_signal = on_signal

        self._gaze = GazeExtractor()
        self._gesture = GestureExtractor()
        self._emotion = EmotionExtractor() if use_emotion else None

        self._buffer = FrameBuffer()
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_id = 0
        self._error_count = 0

        self._latest: CameraSignal = _fallback("disabled")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Pipeline'ı başlat.
        Kamera yoksa False döner ama uygulama çalışmaya devam eder.
        """
        if self._running:
            return True

        ok_gaze    = self._gaze.initialize()
        ok_gesture = self._gesture.initialize()
        if not ok_gaze or not ok_gesture:
            print("[CVPipeline] Extractor başlatılamadı. Kamerasız modda devam.")
            return False

        if self.use_emotion and self._emotion:
            self._emotion.initialize()

        if not self._open_camera():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[CVPipeline] Başlatıldı — kamera {self.camera_index} @ {self.target_fps} FPS")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release_camera()
        self._gaze.release()
        self._gesture.release()
        if self._emotion:
            self._emotion.release()
        print("[CVPipeline] Durduruldu.")

    def latest(self) -> CameraSignal:
        """Feature Extractor bu metodu çağırır — thread-safe."""
        with self._lock:
            return self._latest

    def buffer(self) -> FrameBuffer:
        """Son N frame'e erişmek için."""
        return self._buffer

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._latest.active

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                self._cap = None
                print(f"[CVPipeline] Kamera {self.camera_index} açılamadı. Kamerasız modda devam.")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            return True
        except Exception as e:
            print(f"[CVPipeline] Kamera hatası: {e}")
            return False

    def _release_camera(self):
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _loop(self):
        import cv2
        interval = 1.0 / self.target_fps

        while self._running:
            t0 = time.time()

            if not self._cap or not self._cap.isOpened():
                self._error_count += 1
                if self._error_count > 10:
                    print("[CVPipeline] Kamera bağlantısı kesildi.")
                    break
                time.sleep(0.5)
                continue

            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                self._error_count += 1
                continue
            self._error_count = 0

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            tp = time.time()
            gaze    = self._gaze.extract(frame_rgb)
            gesture = self._gesture.extract(frame_rgb)
            # Emotion BGR ister, diğerleri RGB
            emotion = self._emotion.extract(frame_bgr) if self._emotion else None
            proc_ms = (time.time() - tp) * 1000

            self._frame_id += 1
            signal = CameraSignal(
                active=True,
                gaze=gaze,
                gesture=gesture,
                emotion=emotion,
                timestamp=time.time(),
                frame_id=self._frame_id,
                processing_ms=round(proc_ms, 1),
            )

            with self._lock:
                self._latest = signal

            self._buffer.push(signal.to_dict())

            if self.on_signal:
                try:
                    self.on_signal(signal)
                except Exception as e:
                    print(f"[CVPipeline] Callback hatası: {e}")

            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        self._release_camera()
