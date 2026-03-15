# backend/scorer/rules.py
from backend.core.schemas import FrameData

def evaluate_fatigue(window_data: list[FrameData]) -> float:
    """Göz kapağı (EAR) verisine bakarak yorgunluk cezası keser."""
    if not window_data: return 0.0
    
    avg_ear = sum(f.ear_score for f in window_data) / len(window_data)
    
    # Ortalama EAR 0.20'nin altındaysa öğrenci uyukluyordur.
    if avg_ear < 0.20:
        return 30.0  # 30 Puan Ceza
    elif avg_ear < 0.25:
        return 10.0  # Hafif dalgınlık
    return 0.0

def evaluate_posture(window_data: list[FrameData], fps: int) -> float:
    """El-Çene temas süresine bakarak sıkılma/duruş cezası keser."""
    hand_on_chin_frames = sum(1 for f in window_data if f.hand_on_chin)
    seconds_on_chin = hand_on_chin_frames / fps
    
    # 2 saniyeden (Kayan pencerenin %40'ı) az ise refleks/kaşınmadır.
    if seconds_on_chin > 2.0:
        return 20.0  # 20 Puan Ceza
    return 0.0

def evaluate_attention(window_data: list[FrameData], fps: int) -> float:
    """Gözlerin ekranda olup olmadığına bakar."""
    looking_away_frames = sum(1 for f in window_data if not f.gaze_on_screen)
    seconds_away = looking_away_frames / fps
    
    # 1 saniyelik sağa sola bakma normaldir (not alma vb.)
    if seconds_away > 1.5:
        return 25.0  # 25 Puan Ceza
    return 0.0