# backend/state/uncertainty_engine.py
#
# StateEstimate → müdahale kararı.
# Confidence eşiği aşılmadıysa soru sorar, müdahale etmez.
# Adaptive threshold Hafta 2'de UserProfile'dan okunacak.
#
# Sahip: K1
# Bağımlılıklar: schemas.py, state_model.py

from datetime import datetime, timedelta
from typing import Optional

from backend.core.schemas import (
    InterventionType,
    LearningPattern,
    MentorIntervention,
    StateEstimate,
    UserProfile,
    UserState,
)
from backend.core.config import settings


class UncertaintyEngine:
    """
    İki soruyu yanıtlar:
      1. Müdahale etmeli miyim? (confidence + cooldown kontrolü)
      2. Hangi müdahaleyi yapmalıyım? (state + pattern'a göre)

    Hafta 1: threshold sabit (0.75), cooldown sabit (120sn).
    Hafta 2: threshold UserProfile.adaptive_threshold'dan okunur.
             Her müdahale sonucuna göre threshold güncellenir.

    Kullanım:
        engine = UncertaintyEngine()
        intervention = engine.decide(state_estimate, profile)
        if intervention:
            # Mentor Agent bu intervention'ı LLM'e gönderecek
    """

    def __init__(self):
        # session_id → son müdahale zamanı
        self._last_intervention: dict[str, datetime] = {}

    def decide(
        self,
        estimate: StateEstimate,
        profile: Optional[UserProfile] = None,
        session_id: str = "",
    ) -> Optional[MentorIntervention]:
        """
        Ana karar metodu. None döner → müdahale yok.

        Parametreler:
            estimate   : StateModel'den gelen tahmin
            profile    : long-term memory'den gelen kullanıcı profili
            session_id : cooldown takibi için
        """
        # ── Adaptive threshold ────────────────────────────────────────
        # Hafta 1: default. Hafta 2: profile'dan oku.
        threshold = settings.default_uncertainty_threshold
        if profile and profile.adaptive_threshold:
            threshold = profile.adaptive_threshold

        # ── Müdahale gerekiyor mu? ────────────────────────────────────
        if not self._should_intervene(estimate, threshold, session_id):
            return None

        # ── Müdahale türünü seç ───────────────────────────────────────
        intervention_type = self._select_intervention_type(estimate)

        # ── Mesajı oluştur ────────────────────────────────────────────
        message = self._generate_message(
            intervention_type, estimate.state, estimate.learning_pattern
        )

        # ── Cooldown güncelle ─────────────────────────────────────────
        if session_id:
            self._last_intervention[session_id] = datetime.now()

        return MentorIntervention(
            intervention_type=intervention_type,
            message=message,
            triggered_by=estimate.state,
            learning_pattern=estimate.learning_pattern,
            confidence=estimate.confidence,
        )

    def update_threshold(
        self,
        user_id: str,
        profile: UserProfile,
        intervention_was_useful: bool,
    ) -> float:
        """
        Hafta 2 — adaptive threshold güncelleme.
        Müdahale işe yaradıysa eşiği biraz düşür (daha kolay tetikle).
        İşe yaramadıysa eşiği yükselt (daha zor tetikle).

        Bu metod şimdilik iskelet — Hafta 2'de long-term memory ile dolacak.
        """
        current = profile.adaptive_threshold
        if intervention_was_useful:
            # Eşiği düşür: daha sık müdahale et
            new_threshold = max(0.55, current - 0.05)
        else:
            # Eşiği yükselt: daha az müdahale et
            new_threshold = min(0.95, current + 0.05)
        return round(new_threshold, 2)

    # ── Private metodlar ─────────────────────────────────────────────

    def _should_intervene(
        self,
        estimate: StateEstimate,
        threshold: float,
        session_id: str,
    ) -> bool:
        """
        Müdahale koşulları:
          1. Durum sorunlu (STUCK / FATIGUED / DISTRACTED)
          2. Confidence eşiği aşıldı
          3. Cooldown süresi doldu
        """
        # FOCUSED veya UNKNOWN → müdahale yok
        if estimate.state in [UserState.FOCUSED, UserState.UNKNOWN]:
            return False

        # Confidence düşükse → soru sor (aşağıda), müdahale etme
        if estimate.confidence < threshold:
            return False

        # Cooldown kontrolü
        if session_id and session_id in self._last_intervention:
            elapsed = (
                datetime.now() - self._last_intervention[session_id]
            ).total_seconds()
            if elapsed < settings.intervention_cooldown_seconds:
                return False

        return True

    def _select_intervention_type(
        self, estimate: StateEstimate
    ) -> InterventionType:
        """
        State + learning pattern kombinasyonuna göre müdahale türü seç.

        Karar tablosu:
          STUCK + SHALLOW  → STRATEGY
          STUCK + DEEP     → HINT
          STUCK + MISC     → QUESTION (önce anla)
          FATIGUED         → BREAK
          DISTRACTED       → QUESTION
          Confidence düşük → QUESTION
        """
        state = estimate.state
        pattern = estimate.learning_pattern

        # Güven düşükse önce sor
        if estimate.confidence < estimate.threshold:
            return InterventionType.QUESTION

        if state == UserState.FATIGUED:
            return InterventionType.BREAK

        if state == UserState.STUCK:
            if pattern == LearningPattern.SHALLOW_LEARNING:
                return InterventionType.STRATEGY
            elif pattern == LearningPattern.DEEP_ATTEMPT:
                return InterventionType.HINT
            elif pattern == LearningPattern.MISCONCEPTION:
                return InterventionType.QUESTION
            return InterventionType.HINT

        if state == UserState.DISTRACTED:
            return InterventionType.QUESTION

        return InterventionType.NONE

    def _generate_message(
        self,
        intervention_type: InterventionType,
        state: UserState,
        pattern: LearningPattern,
    ) -> str:
        """
        Hafta 1: sabit şablon mesajlar.
        Hafta 2: Mentor Agent bu metodu çağırmaz —
                 LLM Dynamic Persona Prompt ile mesajı üretir.
                 Bu metod fallback olarak kalır.
        """
        messages = {
            InterventionType.HINT: (
                "Bu konuda biraz takıldın gibi görünüyor. "
                "Problemi farklı bir açıdan ele almayı dener misin?"
            ),
            InterventionType.STRATEGY: (
                "Çok sayıda kısa soru soruyorsun. "
                "Bir adım geri çekip konunun genel yapısına bakmak ister misin?"
            ),
            InterventionType.BREAK: (
                "Bir süredir yoğun çalışıyorsun. "
                "5 dakika mola vermeni öneririm — verim tekrar yükselecek."
            ),
            InterventionType.QUESTION: (
                "Şu an nasıl hissediyorsun? "
                "Takıldığın bir nokta varsa birlikte ele alabiliriz."
            ),
            InterventionType.MODE_SWITCH: (
                "Bu konuyu farklı bir yöntemle denesek nasıl olur? "
                "Düşüncelerini bana anlat, birlikte bakalım."
            ),
        }
        return messages.get(
            intervention_type,
            "Nasıl gidiyor? Yardımcı olabileceğim bir şey var mı?"
        )