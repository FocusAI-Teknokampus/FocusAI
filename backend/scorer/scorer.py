# backend/scorer/scorer.py
from backend.core.schemas import FrameData
from backend.scorer import rules

class AttentionScorer:
    def __init__(self, fps: int = 5):
        self.fps = fps
        self.base_score = 100.0

    def compute_score(self, window_data: list[FrameData]) -> dict:
        """5 saniyelik veriyi alıp final dikkat skorunu ve nedenlerini döndürür."""
        if not window_data:
            return {"score": self.base_score, "reasons": []}

        score = self.base_score
        reasons = []

        # Kuralları işlet ve cezaları topla
        fatigue_penalty = rules.evaluate_fatigue(window_data)
        if fatigue_penalty > 0:
            reasons.append(f"Yorgunluk/Uyuklama (-{fatigue_penalty})")
            score -= fatigue_penalty

        posture_penalty = rules.evaluate_posture(window_data, self.fps)
        if posture_penalty > 0:
            reasons.append(f"Uzun süreli el-yüz teması (-{posture_penalty})")
            score -= posture_penalty

        attention_penalty = rules.evaluate_attention(window_data, self.fps)
        if attention_penalty > 0:
            reasons.append(f"Ekrana bakmama (-{attention_penalty})")
            score -= attention_penalty

        # Skoru 0 ile 100 arasında sınırla (Eksiye düşmesini engelle)
        final_score = max(0.0, min(100.0, score))

        return {
            "score": final_score,
            "reasons": reasons
        }