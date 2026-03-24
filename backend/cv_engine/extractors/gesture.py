"""
CV Engine / Extractors — Baş Pozu ve El Hareketi
MediaPipe FaceMesh + Hands ile baş açıları ve el-yüz tespiti yapar.
"""

import numpy as np
from typing import Optional, Tuple

# Baş pozu eşikleri (derece)
HEAD_DOWN_PITCH = 20.0
HEAD_TURN_YAW   = 25.0

# Yüz bounding box genişletme
FACE_BOX_EXPAND = 0.15

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


def _head_pose(landmarks, w: int, h: int) -> Tuple[float, float, float]:
    try:
        import cv2
        img_pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in _POSE_IDS],
            dtype=np.float64
        )
        focal = w
        cam = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(_3D_POINTS, img_pts, cam, np.zeros((4,1)))
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        if sy > 1e-6:
            x = np.arctan2(rmat[2,1], rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0.0
        return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
    except Exception:
        return 0.0, 0.0, 0.0


def _hand_near_face(hand_lms_list, face_lms) -> Tuple[bool, bool]:
    if not hand_lms_list or face_lms is None:
        return False, False
    xs = [lm.x for lm in face_lms]
    ys = [lm.y for lm in face_lms]
    fx_min, fx_max = min(xs), max(xs)
    fy_min, fy_max = min(ys), max(ys)
    fh = fy_max - fy_min
    chin_y = fy_max - fh * 0.35
    e = FACE_BOX_EXPAND
    on_face = False
    on_chin = False
    for hand in hand_lms_list:
        for lm in hand.landmark:
            if (fx_min - e) < lm.x < (fx_max + e) and (fy_min - e) < lm.y < (fy_max + e):
                on_face = True
                if lm.y > chin_y:
                    on_chin = True
                    break
        if on_chin:
            break
    return on_chin, on_face


class GestureExtractor:
    """
    pipeline.py tarafından her frame'de çağrılır.
    Baş açıları ve el bilgilerini dict olarak döndürür.
    """

    def __init__(self):
        self._face_mesh = None
        self._hands = None
        self._initialized = False

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

    def extract(self, frame_rgb: np.ndarray) -> Optional[dict]:
        if not self._initialized:
            return None

        h, w = frame_rgb.shape[:2]

        face_lms = None
        pitch = yaw = roll = 0.0
        if self._face_mesh:
            fr = self._face_mesh.process(frame_rgb)
            if fr.multi_face_landmarks:
                face_lms = fr.multi_face_landmarks[0].landmark
                pitch, yaw, roll = _head_pose(face_lms, w, h)

        hand_lms_list = None
        if self._hands:
            hr = self._hands.process(frame_rgb)
            hand_lms_list = hr.multi_hand_landmarks if hr else None

        on_chin, on_face = _hand_near_face(hand_lms_list, face_lms)

        return {
            "head_pitch":    round(pitch, 1),
            "head_yaw":      round(yaw, 1),
            "head_roll":     round(roll, 1),
            "is_head_down":  pitch > HEAD_DOWN_PITCH,
            "is_head_turned": abs(yaw) > HEAD_TURN_YAW,
            "hand_on_chin":  on_chin,
            "hand_on_face":  on_face,
            "hand_detected": bool(hand_lms_list),
        }

    def release(self):
        if self._face_mesh:
            self._face_mesh.close()
        if self._hands:
            self._hands.close()
        self._initialized = False
