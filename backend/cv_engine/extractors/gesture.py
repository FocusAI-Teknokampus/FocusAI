"""
CV Engine / Extractors — Baş Pozu ve El Hareketi
MediaPipe FaceMesh + Hands ile baş açıları ve bölge bazlı
el-yüz tespiti yapar. Kaşıma sıklığı (scratch frequency) takibi içerir.
"""

import time
import numpy as np
from typing import Optional, Tuple, List

# Baş pozu eşikleri (derece)
HEAD_DOWN_PITCH = 20.0
HEAD_TURN_YAW   = 25.0

# 3D referans noktaları (solvePnP için)
_3D_POINTS = np.array([
    [0.0,    0.0,    0.0  ],   # Burun ucu
    [0.0,  -330.0,  -65.0],   # Çene
    [-225.0, 170.0, -135.0],  # Sol göz köşesi
    [225.0,  170.0, -135.0],  # Sağ göz köşesi
    [-150.0,-150.0, -125.0],  # Sol ağız köşesi
    [150.0, -150.0, -125.0],  # Sağ ağız köşesi
], dtype=np.float64)

_POSE_IDS = [1, 152, 263, 33, 287, 57]

# El-yüz bölge sınırları (yüz yüksekliğinin oranı olarak)
_REGION_FOREHEAD = 0.18   # Alın üstü (bu altı kaş/gözlük bölgesi)
_REGION_BROW     = 0.32   # Kaş/göz başlangıcı
_REGION_NOSE     = 0.58   # Burun altı
_REGION_MOUTH    = 0.74   # Ağız
_CHIN_EXTRA      = 0.08   # Yüzün altına taşma toleransı
_X_MARGIN        = 0.15   # Yatay genişletme oranı (yüz genişliğine göre)

# Kaşıma sıklığı eşikleri
SCRATCH_WINDOW   = 120    # Saniye — kaç saniyelik pencerede say
SCRATCH_THRESH   = 5      # Bu pencerede kaç kez olursa ceza ver
SCRATCH_PENALTY_PER = 0.04  # Her fazla dokunuş için ceza


def _head_pose(landmarks, w: int, h: int) -> Tuple[float, float, float]:
    """solvePnP ile baş pitch/yaw/roll açılarını hesaplar (derece)."""
    try:
        import cv2
        img_pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in _POSE_IDS],
            dtype=np.float64
        )
        focal = float(w)
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(_3D_POINTS, img_pts, cam, np.zeros((4, 1)))
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0.0
        return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
    except Exception:
        return 0.0, 0.0, 0.0


def _classify_hand_regions(
    hand_lms_list,
    face_lms,
) -> Tuple[bool, bool, bool, bool]:
    """
    Her el landmark'ının yüz üzerindeki bölgesini sınıflandırır.

    Döndürür: (hon_chin, hon_mouth, hon_face, hon_scratch)
      - hon_chin     : çene bölgesinde el var
      - hon_mouth    : ağız bölgesinde el var
      - hon_face     : yüzde herhangi bir yerde el var
      - hon_scratch  : alın/kaş/yanak/burun bölgesinde el var (kaşıma)
    """
    if not hand_lms_list or face_lms is None:
        return False, False, False, False

    xs = [lm.x for lm in face_lms]
    ys = [lm.y for lm in face_lms]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    face_h = y1 - y0
    face_w = x1 - x0

    forehead_y = y0 + face_h * _REGION_FOREHEAD
    brow_y     = y0 + face_h * _REGION_BROW
    nose_y     = y0 + face_h * _REGION_NOSE
    mouth_y    = y0 + face_h * _REGION_MOUTH
    chin_bot_y = y1 + face_h * _CHIN_EXTRA
    x_margin   = face_w * _X_MARGIN

    hon_chin = hon_mouth = hon_face = hon_scratch = False

    for hand in hand_lms_list:
        lms_iter = hand.landmark if hasattr(hand, "landmark") else hand
        cx = lms_iter[9].x  # Avuç ortası (landmark 9)

        for lm in lms_iter:
            lx, ly = lm.x, lm.y
            in_lx = (x0 - x_margin) < lx < (x1 + x_margin)

            if not in_lx:
                continue
            if ly > chin_bot_y:
                continue   # Yüzün çok altı → göğüs, yoksay

            if ly < forehead_y:
                pass       # Kafa/saç üstü → yoksay
            elif ly < brow_y:
                hon_scratch = True
                hon_face    = True
            elif ly < nose_y:
                hon_scratch = True
                hon_face    = True
            elif ly < mouth_y:
                hon_mouth = True
                hon_face  = True
            elif ly < chin_bot_y:
                in_x = (x0 - x_margin) < cx < (x1 + x_margin)
                if in_x:
                    hon_chin = True
                    hon_face = True

    return hon_chin, hon_mouth, hon_face, hon_scratch


class GestureExtractor:
    """
    pipeline.py tarafından her frame'de çağrılır.
    Baş açıları, bölge bazlı el sınıflandırması ve kaşıma
    sıklığı bilgilerini dict olarak döndürür.
    """

    def __init__(self):
        self._face_mesh   = None
        self._hands       = None
        self._initialized = False

        # Kaşıma sıklığı state
        self._scratch_times:      List[float] = []
        self._face_touch_active:  bool        = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"[GestureExtractor] Başlatma hatası: {e}")
            return False

    def release(self):
        if self._face_mesh:
            self._face_mesh.close()
        if self._hands:
            self._hands.close()
        self._initialized = False

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        frame_rgb: RGB formatında numpy array
        Döndürür: dict (baş pozu + el bölgesi + kaşıma sıklığı) veya None
        """
        if not self._initialized:
            return None

        h, w = frame_rgb.shape[:2]
        now  = time.time()

        # ── Yüz landmark ──────────────────────────────────────────────
        face_lms = None
        pitch = yaw = roll = 0.0
        if self._face_mesh:
            fr = self._face_mesh.process(frame_rgb)
            if fr.multi_face_landmarks:
                face_lms = fr.multi_face_landmarks[0].landmark
                pitch, yaw, roll = _head_pose(face_lms, w, h)

        # ── El landmark ───────────────────────────────────────────────
        hand_lms_list = None
        if self._hands:
            hr = self._hands.process(frame_rgb)
            hand_lms_list = hr.multi_hand_landmarks if hr else None

        # ── Bölge sınıflandırması ─────────────────────────────────────
        hon_chin, hon_mouth, hon_face, hon_scratch = _classify_hand_regions(
            hand_lms_list, face_lms
        )

        # ── Kaşıma sıklığı (debounce + pencere sayacı) ────────────────
        if hon_scratch and not self._face_touch_active:
            self._scratch_times.append(now)
            self._face_touch_active = True
        elif not hon_scratch:
            self._face_touch_active = False

        # Eski kayıtları temizle
        self._scratch_times = [t for t in self._scratch_times if now - t < SCRATCH_WINDOW]
        scratch_count = len(self._scratch_times)

        if scratch_count >= SCRATCH_THRESH:
            scratch_penalty = min(0.20, (scratch_count - SCRATCH_THRESH + 1) * SCRATCH_PENALTY_PER)
        else:
            scratch_penalty = 0.0

        return {
            "head_pitch":      round(pitch, 1),
            "head_yaw":        round(yaw, 1),
            "head_roll":       round(roll, 1),
            "is_head_down":    pitch > HEAD_DOWN_PITCH,
            "is_head_turned":  abs(yaw) > HEAD_TURN_YAW,
            "hand_on_chin":    hon_chin,
            "hand_on_mouth":   hon_mouth,
            "hand_on_face":    hon_face,
            "hand_scratching": hon_scratch,
            "hand_detected":   bool(hand_lms_list),
            "scratch_count":   scratch_count,
            "scratch_window":  SCRATCH_WINDOW,
            "scratch_penalty": round(scratch_penalty, 3),
        }
