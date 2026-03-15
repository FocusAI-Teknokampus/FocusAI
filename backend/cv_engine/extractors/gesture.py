# backend/cv_engine/extractors/gesture.py
import mediapipe as mp
import math
import cv2

# MediaPipe Pose (İskelet ve Vücut Duruşu) modelini başlatıyoruz
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def _euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def check_hand_on_chin(frame) -> bool:
    """
    Öğrencinin elinin çenesinde/yüzünde olup olmadığını tespit eder.
    El bilekleri ile ağız/burun arasındaki mesafeyi ölçer.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if not results.pose_landmarks:
         return False # Kamerada insan iskeleti yok

    landmarks = results.pose_landmarks.landmark
    h, w, _ = frame.shape

    # Önemli noktaların koordinatlarını alıyoruz
    # 0: Burun, 9: Sol Ağız Kenarı, 10: Sağ Ağız Kenarı
    # 15: Sol El Bileği, 16: Sağ El Bileği
    
    # Boyun/Çene merkezini tahmin etmek için ağız ve burun çevresini referans alıyoruz
    nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x * w, 
            landmarks[mp_pose.PoseLandmark.NOSE.value].y * h)
            
    left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, 
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h)
                  
    right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)

    # El bileği ile yüz (burun/çene hattı) arasındaki mesafeyi ölçüyoruz
    dist_left = _euclidean_distance(nose, left_wrist)
    dist_right = _euclidean_distance(nose, right_wrist)

    # Yüzün kendi büyüklüğünü referans alarak dinamik bir eşik değeri (threshold) belirliyoruz.
    # Omuzlar arası mesafe, kişinin kameraya ne kadar yakın olduğunu anlamak için harika bir referanstır.
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
    
    shoulder_width = _euclidean_distance(left_shoulder, right_shoulder)
    
    # Eşik değeri: Eğer el, omuz genişliğinin %30'undan daha fazla yüze yaklaşmışsa, "el çenededir".
    threshold = shoulder_width * 0.30

    # Eğer sağ veya sol el yüze temas edecek kadar yakınsa True döndür
    if dist_left < threshold or dist_right < threshold:
        return True

    return False