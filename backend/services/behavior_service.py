# backend/services/behavior_service.py

import json
from typing import Optional

from sqlalchemy.orm import Session

from backend.core.schemas import FeatureVector, StateEstimate, UserState
from backend.services.session_service import SessionService


class BehaviorService:
    """
    FeatureVector ve StateEstimate sonuçlarını kalıcı event kayıtlarına dönüştürür.

    Sorumluluklar:
        - behavior event üretmek
        - focus event üretmek
        - session retry/state bilgisini güncellemek

    Not:
        Bu sınıf yeni state tahmini yapmaz.
        Sadece mevcut graph çıktısını DB kayıtlarına çevirir.
    """

    def __init__(self, db: Session):
        """
        Parametreler:
            db: Request'e özel SQLAlchemy session
        """
        self.db = db
        self.session_service = SessionService(db)

    def persist_analysis(
        self,
        session_id: str,
        user_id: str,
        feature_vector: Optional[FeatureVector],
        state_estimate: Optional[StateEstimate],
    ) -> None:
        """
        Graph analiz çıktısını DB'ye yazar.

        Parametreler:
            session_id: Oturum kimliği
            user_id: Kullanıcı kimliği
            feature_vector: FeatureExtractor çıktısı
            state_estimate: StateModel çıktısı
        """
        if feature_vector is None or state_estimate is None:
            return

        # 1) Session state ve retry bilgisini güncelle
        self.session_service.update_session_state(
            session_id=session_id,
            state=state_estimate.state.value,
            retry_count=feature_vector.retry_count,
        )

        # 2) Focus event kaydet
        focus_score = self._estimate_focus_score(state_estimate)
        self.session_service.log_focus_event(
            session_id=session_id,
            user_id=user_id,
            focus_score=focus_score,
            source="text",
            state_label=state_estimate.state.value,
        )

        # 3) Behavior event(ler)i üret
        events = self._infer_behavior_events(feature_vector, state_estimate)
        for event in events:
            self.session_service.log_behavior_event(
                session_id=session_id,
                user_id=user_id,
                event_type=event["event_type"],
                state_before=event.get("state_before"),
                state_after=event.get("state_after"),
                topic=event.get("topic"),
                severity=event.get("severity"),
                metadata_json=json.dumps(event.get("metadata", {}), ensure_ascii=False),
            )

    def _estimate_focus_score(self, estimate: StateEstimate) -> float:
        """
        StateEstimate içinden 0.0–1.0 arası basit focus score üretir.

        Mantık:
            - focused yüksek
            - recovering orta-üst
            - stuck / distracted / fatigued daha düşük
            - confidence ile hafifçe ağırlıklandırılır
        """
        base_map = {
            UserState.FOCUSED: 0.90,
            UserState.UNKNOWN: 0.50,
            UserState.STUCK: 0.40,
            UserState.DISTRACTED: 0.25,
            UserState.FATIGUED: 0.20,
            UserState.RECOVERING: 0.65,
        }

        base = base_map.get(estimate.state, 0.50)
        conf = estimate.confidence or 0.0

        # confidence yükseldikçe base değere biraz daha yaklaş
        score = base * 0.8 + conf * 0.2
        return round(max(0.0, min(1.0, score)), 3)

    def _infer_behavior_events(
        self,
        f: FeatureVector,
        estimate: StateEstimate,
    ) -> list[dict]:
        """
        FeatureVector + StateEstimate'ten behavior event listesi üretir.

        Event mantığı:
            - tekrar sayısı
            - çok kısa ve hızlı mesajlar
            - uzun bekleme
            - topic değişimi sinyali
            - learning pattern sonucu
        """
        events: list[dict] = []

        # Tekrar soru / aynı konuda dönme
        if f.retry_count >= 2:
            events.append({
                "event_type": "question_repeat",
                "state_before": None,
                "state_after": estimate.state.value,
                "topic": f.topic,
                "severity": min(1.0, 0.3 + 0.1 * f.retry_count),
                "metadata": {
                    "retry_count": f.retry_count,
                    "message_length": f.message_length,
                },
            })

        # Çok kısa ve hızlı soru → yüzeysel pattern sinyali
        if f.retry_count >= 3 and f.response_time_seconds < 5:
            events.append({
                "event_type": "rapid_short_questions",
                "state_before": None,
                "state_after": estimate.state.value,
                "topic": f.topic,
                "severity": 0.75,
                "metadata": {
                    "retry_count": f.retry_count,
                    "response_time_seconds": f.response_time_seconds,
                },
            })

        # Uzun duraksama
        if f.idle_time_seconds > 180:
            events.append({
                "event_type": "long_pause",
                "state_before": None,
                "state_after": estimate.state.value,
                "topic": f.topic,
                "severity": 0.70,
                "metadata": {
                    "idle_time_seconds": f.idle_time_seconds,
                },
            })

        # Çok yüksek tekrar + kısa mesaj → misconception benzeri sinyal
        if f.retry_count >= 5 and f.message_length < 30:
            events.append({
                "event_type": "same_misconception_again",
                "state_before": None,
                "state_after": estimate.state.value,
                "topic": f.topic,
                "severity": 0.85,
                "metadata": {
                    "retry_count": f.retry_count,
                    "message_length": f.message_length,
                },
            })

        # Topic algısı varsa basit signal kaydı
        if f.topic:
            events.append({
                "event_type": "topic_signal",
                "state_before": None,
                "state_after": estimate.state.value,
                "topic": f.topic,
                "severity": 0.20,
                "metadata": {},
            })

        return events