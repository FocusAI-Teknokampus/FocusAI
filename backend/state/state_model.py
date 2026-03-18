# backend/state/state_model.py
#
# FeatureVector → UserState + confidence tahmini.
# Hafta 1: kural tabanlı (sabit threshold'lar).
# Hafta 2: adaptive threshold + Uncertainty Engine entegrasyonu.
#
# Sahip: K1
# Bağımlılıklar: schemas.py, feature_extractor.py (K3)

from backend.core.schemas import (
    FeatureVector,
    LearningPattern,
    StateEstimate,
    UserState,
)
from backend.core.config import settings


class StateModel:
    """
    FeatureVector'dan kullanıcı durumu tahmin eder.

    Kurallar (Hafta 1 — sabit eşikler):
        STUCK       : retry_count >= 3  VE  idle_time > 120sn
        FATIGUED    : ear_score < 0.20  VEYA  idle_time > 300sn
        DISTRACTED  : gaze_on_screen == False  VEYA  hand_on_chin == True
        FOCUSED     : yukarıdakilerin hiçbiri

    Confidence hesabı:
        Kaç sinyal aynı duruma işaret ediyorsa o kadar yüksek.
        Örnek: retry_count yüksek ama idle_time düşükse → stuck, confidence 0.6
               İkisi de yüksekse → stuck, confidence 0.9

    Kullanım:
        model = StateModel()
        estimate = model.predict(feature_vector, user_profile)
    """

    def predict(
        self,
        features: FeatureVector,
        adaptive_threshold: float = None,
    ) -> StateEstimate:
        """
        Ana tahmin metodu.

        Parametreler:
            features           : FeatureExtractor'dan gelen vektör
            adaptive_threshold : kullanıcıya özel eşik (None → default 0.75)
        """
        threshold = adaptive_threshold or settings.default_uncertainty_threshold

        state, confidence = self._evaluate_rules(features)
        pattern = self._detect_learning_pattern(features)

        return StateEstimate(
            session_id=features.session_id,
            state=state,
            confidence=round(confidence, 2),
            learning_pattern=pattern,
            threshold=threshold,
        )

    # ── Kural motoru ─────────────────────────────────────────────────

    def _evaluate_rules(
        self, f: FeatureVector
    ) -> tuple[UserState, float]:
        """
        Her kural bir "oy" verir. En çok oy alan durum kazanır.
        Confidence = kazanan durum oyları / toplam sinyal sayısı.
        """
        votes: dict[UserState, float] = {
            UserState.STUCK: 0.0,
            UserState.FATIGUED: 0.0,
            UserState.DISTRACTED: 0.0,
            UserState.FOCUSED: 0.0,
        }
        total_signals = 0

        # ── STUCK sinyalleri ─────────────────────────────────────────
        if f.retry_count >= 3:
            votes[UserState.STUCK] += 1.0
            total_signals += 1

        if f.idle_time_seconds > 120:
            votes[UserState.STUCK] += 0.8
            total_signals += 1

        if f.retry_count >= 5:                        # Güçlü sinyal
            votes[UserState.STUCK] += 0.5             # Ek ağırlık
            total_signals += 1

        # ── FATIGUED sinyalleri ──────────────────────────────────────
        if f.ear_score is not None and f.ear_score < 0.20:
            votes[UserState.FATIGUED] += 1.0
            total_signals += 1

        if f.ear_score is not None and f.ear_score < 0.25:
            votes[UserState.FATIGUED] += 0.5
            total_signals += 1

        if f.idle_time_seconds > 300:                 # 5 dk sessizlik
            votes[UserState.FATIGUED] += 0.8
            total_signals += 1

        if f.head_tilt_angle is not None and abs(f.head_tilt_angle) > 25:
            votes[UserState.FATIGUED] += 0.6          # Başı eğik
            total_signals += 1

        # ── DISTRACTED sinyalleri ────────────────────────────────────
        if f.gaze_on_screen is not None and not f.gaze_on_screen:
            votes[UserState.DISTRACTED] += 1.0
            total_signals += 1

        if f.hand_on_chin is True:
            votes[UserState.DISTRACTED] += 0.7
            total_signals += 1

        if f.message_length < 5 and f.idle_time_seconds > 30:
            votes[UserState.DISTRACTED] += 0.5        # Çok kısa mesaj, uzun bekleme
            total_signals += 1

        # ── FOCUSED sinyalleri ───────────────────────────────────────
        if f.retry_count == 0 and f.idle_time_seconds < 60:
            votes[UserState.FOCUSED] += 1.0
            total_signals += 1

        if f.ear_score is not None and f.ear_score >= 0.28:
            votes[UserState.FOCUSED] += 0.8
            total_signals += 1

        if f.gaze_on_screen is True and f.hand_on_chin is False:
            votes[UserState.FOCUSED] += 0.7
            total_signals += 1

        # ── Karar ────────────────────────────────────────────────────
        if total_signals == 0:
            return UserState.UNKNOWN, 0.0

        winner = max(votes, key=lambda s: votes[s])
        winner_votes = votes[winner]

        if winner_votes == 0.0:
            return UserState.FOCUSED, 0.5

        confidence = min(1.0, winner_votes / total_signals)
        return winner, confidence

    def _detect_learning_pattern(
        self, f: FeatureVector
    ) -> LearningPattern:
        """
        Öğrenme stratejisini tespit et.
        State'den bağımsız — sadece davranış sinyallerine bakar.
        """
        # Kısa sorular hızlı geliyorsa → shallow
        if f.retry_count >= 3 and f.response_time_seconds < 5:
            return LearningPattern.SHALLOW_LEARNING

        # Uzun süre uğraşıyorsa → deep attempt
        if f.idle_time_seconds > 180 and f.retry_count >= 1:
            return LearningPattern.DEEP_ATTEMPT

        # Misconception tespiti Hafta 2'de long-term memory ile gelecek
        # Şimdilik: çok yüksek retry ve mesajlar çok kısa
        if f.retry_count >= 5 and f.message_length < 30:
            return LearningPattern.MISCONCEPTION

        return LearningPattern.NORMAL