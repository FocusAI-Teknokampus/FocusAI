"""
CV Engine / Extractors — Göz Takibi + Esneme
MediaPipe FaceLandmarker ile EAR, MAR hesaplar;
uyuklama, bakış yönü ve esneme tespiti yapar.
"""

import time
import numpy as np
from typing import Optional

# MediaPipe FaceMesh landmark indeksleri
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# EAR eşik değerleri
EAR_BLINK_THRESHOLD  = 0.20
EAR_DROWSY_THRESHOLD = 0.23
DROWSY_DURATION_SEC  = 1.5

# MAR (esneme) eşik değerleri
MAR_YAWN_THRESHOLD   = 0.45
YAWN_DURATION_SEC    = 0.8   # Bu süre boyunca MAR > eşik → esneme sayılır

# Bakış eşiği
GAZE_SIDE_THRESHOLD  = 0.35


def _ear(landmarks, eye_indices: list, w: int, h: int) -> float:
    """Eye Aspect Ratio — göz açıkken ~0.3, kapalıyken ~0.0"""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0


def _mar(landmarks, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio — ağız açıkken büyük, kapalıyken küçük.
    Landmark 13/14 = üst/alt dudak ortası, 78/308 = sol/sağ ağız köşesi.
    """
    top    = np.array([landmarks[13].x * w, landmarks[13].y * h])
    bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left   = np.array([landmarks[78].x * w, landmarks[78].y * h])
    right  = np.array([landmarks[308].x * w, landmarks[308].y * h])
    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 1e-6 else 0.0


def _gaze_direction(landmarks, w: int, h: int) -> str:
    """İris konumuna göre bakış yönü: center / left / right / up / down"""
    try:
        lc   = landmarks[362]
        rc   = landmarks[263]
        iris = landmarks[474]
        eye_w = abs(rc.x - lc.x)
        if eye_w < 1e-6:
            return "center"
        offset = (iris.x - lc.x) / eye_w

        top   = landmarks[386]
        bot   = landmarks[374]
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
    EAR, MAR, uyuklama, esneme ve bakış yönü bilgilerini
    dict olarak döndürür. CameraSignal'e doğrudan aktarılır.
    """

    def __init__(self):
        self._face_mesh       = None
        self._initialized     = False

        # Uyuklama state
        self._drowsy_start:         Optional[float] = None
        self._eyes_closed_since:    Optional[float] = None

        # Esneme state
        self._yawn_start:   Optional[float] = None
        self._yawn_total:   int             = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,          # İris + dudak detayı için gerekli
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"[GazeExtractor] Başlatma hatası: {e}")
            return False

    def release(self):
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        frame_rgb: RGB formatında numpy array
        Döndürür: dict (EAR, MAR, uyuklama, esneme, bakış yönü) veya None
        """
        if not self._initialized or self._face_mesh is None:
            return None

        h, w = frame_rgb.shape[:2]
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            # Yüz yok — tüm state'leri sıfırla
            self._drowsy_start       = None
            self._eyes_closed_since  = None
            self._yawn_start         = None
            return None

        lms = results.multi_face_landmarks[0].landmark
        now = time.time()

        # ── EAR ───────────────────────────────────────────────────────
        ear_l   = _ear(lms, LEFT_EYE,  w, h)
        ear_r   = _ear(lms, RIGHT_EYE, w, h)
        ear_avg = (ear_l + ear_r) / 2.0

        # Göz kapalı süre
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

        # ── MAR / Esneme ───────────────────────────────────────────────
        ma = _mar(lms, w, h)
        is_yawning = False

        if ma > MAR_YAWN_THRESHOLD:
            if self._yawn_start is None:
                self._yawn_start = now
            elif (now - self._yawn_start) >= YAWN_DURATION_SEC:
                is_yawning = True
        else:
            # Esneme bitti — sayacı artır
            if self._yawn_start is not None and (now - self._yawn_start) >= YAWN_DURATION_SEC:
                self._yawn_total += 1
            self._yawn_start = None

        # ── Bakış yönü ─────────────────────────────────────────────────
        gaze_dir = _gaze_direction(lms, w, h)

        return {
            "ear_left":             round(ear_l, 4),
            "ear_right":            round(ear_r, 4),
            "ear_avg":              round(ear_avg, 4),
            "is_blinking":          ear_avg < EAR_BLINK_THRESHOLD,
            "is_drowsy":            is_drowsy,
            "eyes_closed_duration": round(closed_dur, 2),
            "gaze_direction":       gaze_dir,
            "mar":                  round(ma, 4),
            "is_yawning":           is_yawning,
            "yawn_count":           self._yawn_total,
        }

    # ------------------------------------------------------------------
    # Helpers (pipeline veya dış modüller için)
    # ------------------------------------------------------------------

    def get_landmarks(self, frame_rgb: np.ndarray):
        """
        Ham FaceMesh landmark listesini döndürür (GestureExtractor'ın
        el-yüz bölge tespiti için kullanabileceği ortak veri).
        None: yüz tespit edilemedi.
        """
        if not self._initialized or self._face_mesh is None:
            return None
        results = self._face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
