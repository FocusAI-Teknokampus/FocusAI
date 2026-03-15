# backend/cv_engine/extractors/gaze.py
import math
import mediapipe as mp
import cv2

# MediaPipe Yüz Ağı (Face Mesh) modelini başlatıyoruz
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe literatürüne göre göz çevresindeki referans noktaları
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def check_gaze_on_screen(frame) -> bool:
    """Şimdilik öğrenci hep ekrana bakıyor varsayıyoruz."""
    return True

def _euclidean_distance(p1: tuple, p2: tuple) -> float:
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.dist(p1, p2)

def calculate_ear(frame) -> float:
    """Görüntüyü alır, yüzü bulur ve Göz Kapağı Açıklık Oranını (EAR) hesaplar."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return 0.0 # Yüz kamerada yok!

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    def get_eye_ear(eye_indices):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        
        v1 = _euclidean_distance(pts[1], pts[5])
        v2 = _euclidean_distance(pts[2], pts[4])
        h1 = _euclidean_distance(pts[0], pts[3])
        
        if h1 == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h1)

    left_ear = get_eye_ear(LEFT_EYE_INDICES)
    right_ear = get_eye_ear(RIGHT_EYE_INDICES)
    
    return (left_ear + right_ear) / 2.0