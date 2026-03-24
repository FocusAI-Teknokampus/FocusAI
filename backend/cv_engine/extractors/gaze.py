"""
CV Engine / Extractors — Göz Takibi
MediaPipe FaceMesh ile EAR hesaplar, uyuklama ve bakış yönü tespit eder.
"""

import time
import numpy as np
from typing import Optional

# MediaPipe FaceMesh landmark indeksleri
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Eşik değerleri
EAR_BLINK_THRESHOLD  = 0.20   # Altı = göz kırpma
EAR_DROWSY_THRESHOLD = 0.23   # Altı (uzun süre) = uyuklama
DROWSY_DURATION_SEC  = 1.5    # Kaç saniye altında kalırsa uyuklama sayılır
GAZE_SIDE_THRESHOLD  = 0.35   # İris offset oranı


def _ear(landmarks, eye_indices: list, w: int, h: int) -> float:
    """Eye Aspect Ratio — göz açıkken ~0.3, kapalıyken ~0.0"""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0


def _gaze_direction(landmarks, w: int, h: int) -> str:
    """İris konumuna göre bakış yönü: center / left / right / up / down"""
    try:
        lc = landmarks[362]  # Sol göz sol köşe
        rc = landmarks[263]  # Sol göz sağ köşe
        iris = landmarks[474]
        eye_w = abs(rc.x - lc.x)
        if eye_w < 1e-6:
            return "center"
        offset = (iris.x - lc.x) / eye_w

        top = landmarks[386]
        bot = landmarks[374]
        eye_h = abs(bot.y - top.y)
        if eye_h > 1e-6:
            v = (iris.y - top.y) / eye_h
            if v < 0.3:
                return "up"
            if v > 0.7:
                return "down"

        if offset < GAZE_SIDE_THRESHOLD:
            return "right"
        if offset > (1.0 - GAZE_SIDE_THRESHOLD):
            return "left"
        return "center"
    except Exception:
        return "center"


class GazeExtractor:
    """
    pipeline.py tarafından her frame'de çağrılır.
    Döndürdüğü dict doğrudan CameraSignal'e aktarılır.
    """

    def __init__(self):
        self._face_mesh = None
        self._initialized = False
        self._drowsy_start: Optional[float] = None
        self._eyes_closed_since: Optional[float] = None

    def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"[GazeExtractor] Başlatma hatası: {e}")
            return False

    def extract(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        frame_rgb: RGB formatında numpy array
        Döndürür: dict (EAR, uyuklama, bakış yönü) veya None
        """
        if not self._initialized or self._face_mesh is None:
            return None

        h, w = frame_rgb.shape[:2]
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            self._drowsy_start = None
            self._eyes_closed_since = None
            return None

        lms = results.multi_face_landmarks[0].landmark
        ear_l = _ear(lms, LEFT_EYE, w, h)
        ear_r = _ear(lms, RIGHT_EYE, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        now = time.time()

        # Gözler kapalı süre
        if ear_avg < EAR_BLINK_THRESHOLD:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            closed_dur = now - self._eyes_closed_since
        else:
            self._eyes_closed_since = None
            closed_dur = 0.0

        # Uyuklama
        if ear_avg < EAR_DROWSY_THRESHOLD:
            if self._drowsy_start is None:
                self._drowsy_start = now
            is_drowsy = (now - self._drowsy_start) >= DROWSY_DURATION_SEC
        else:
            self._drowsy_start = None
            is_drowsy = False

        return {
            "ear_left":            round(ear_l, 4),
            "ear_right":           round(ear_r, 4),
            "ear_avg":             round(ear_avg, 4),
            "is_blinking":         ear_avg < EAR_BLINK_THRESHOLD,
            "is_drowsy":           is_drowsy,
            "gaze_direction":      _gaze_direction(lms, w, h),
            "eyes_closed_duration": round(closed_dur, 2),
        }

    def release(self):
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        self._initialized = False
