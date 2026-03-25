# backend/services/session_service.py

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    UserRecord,
    UserProfileRecord,
    SessionRecord,
    MessageRecord,
    InterventionRecord,
    BehaviorEventRecord,
    FocusEventRecord,
)


class SessionService:
    """
    Veritabanındaki kullanıcı, oturum, mesaj ve event kayıtlarını yöneten servis.

    Not:
        Bu servis HTTP katmanını bilmez.
        Router veya Agent katmanı bu servisi çağırır.
    """

    def __init__(self, db: Session):
        """
        Parametreler:
            db: SQLAlchemy veritabanı oturumu
        """
        self.db = db

    # ============================================================
    # USER
    # ============================================================

    def get_or_create_user(self, user_id: str) -> UserRecord:
        """
        Kullanıcıyı getirir; yoksa oluşturur.

        Parametreler:
            user_id: Kullanıcı kimliği

        Döner:
            UserRecord
        """
        user = self.db.query(UserRecord).filter(UserRecord.user_id == user_id).first()

        if user:
            user.last_active_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(user)
            return user

        user = UserRecord(
            user_id=user_id,
            total_sessions=0,
            last_active_at=datetime.utcnow(),
        )
        self.db.add(user)
        self.db.flush()

        profile = UserProfileRecord(
        user_id=user.user_id,
        preferred_explanation_style="detailed",
        weak_topics="[]",
        strong_topics="[]",
        recurring_misconceptions="[]",
        avg_session_duration_minutes=0.0,
        adaptive_threshold=0.75,
        )
        self.db.add(profile)

        self.db.commit()
        self.db.refresh(user)
        return user

    # ============================================================
    # SESSION
    # ============================================================

    def create_session(
        self,
        session_id: str,
        user_id: str,
        topic: Optional[str] = None,
        subtopic: Optional[str] = None,
        study_mode: Optional[str] = None,
        camera_used: bool = False,
    ) -> SessionRecord:
        """
        Dışarıdan üretilmiş session_id ile yeni session kaydı oluşturur.

        Neden ayrı metod?
            Repo'da session_id üretimi SessionAgent içinde yapılıyor.
            Bu yüzden DB katmanı agent'ın ürettiği kimliği kullanmalı.

        Parametreler:
            session_id: Agent tarafından üretilen benzersiz oturum kimliği
            user_id: Kullanıcı kimliği
            topic: Ana konu
            subtopic: Alt konu
            study_mode: Çalışma modu
            camera_used: Kamera açık mı?

        Döner:
            SessionRecord
        """
        user = self.get_or_create_user(user_id)

        session = SessionRecord(
            session_id=session_id,
            user_id=user.user_id,
            topic=topic,
            subtopic=subtopic,
            study_mode=study_mode,
            camera_used=camera_used,
            started_at=datetime.utcnow(),
            current_state="unknown",
            average_focus_score=None,
            retry_count=0,
            intervention_count=0,
        )

        self.db.add(session)

        user.total_sessions += 1
        user.last_active_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(session)
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Session kaydını getirir.

        Parametreler:
            session_id: Oturum kimliği

        Döner:
            SessionRecord | None
        """
        return (
            self.db.query(SessionRecord)
            .filter(SessionRecord.session_id == session_id)
            .first()
        )

    def end_session(
        self,
        session_id: str,
        final_state: Optional[str] = None,
    ) -> Optional[SessionRecord]:
        """
        Session'ı veritabanında kapatır.

        Parametreler:
            session_id: Kapatılacak oturum
            final_state: Son kullanıcı durumu

        Döner:
            SessionRecord | None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.ended_at = datetime.utcnow()

        if final_state:
            session.current_state = final_state

        avg_focus = self._compute_average_focus_score(session_id)
        session.average_focus_score = avg_focus

        self.db.commit()
        self.db.refresh(session)
        return session

    def update_session_state(
        self,
        session_id: str,
        state: str,
        retry_count: Optional[int] = None,
    ) -> Optional[SessionRecord]:
        """
        Session state bilgisini günceller.

        Parametreler:
            session_id: Oturum kimliği
            state: Yeni durum etiketi
            retry_count: İstenirse güncel retry değeri

        Döner:
            SessionRecord | None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.current_state = state

        if retry_count is not None:
            session.retry_count = retry_count

        avg_focus = self._compute_average_focus_score(session_id)
        session.average_focus_score = avg_focus

        self.db.commit()
        self.db.refresh(session)
        return session

    # ============================================================
    # MESSAGE
    # ============================================================

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_state: Optional[str] = None,
        detected_topic: Optional[str] = None,
        message_type: Optional[str] = None,
        llm_confidence: Optional[float] = None,
    ) -> MessageRecord:
        """
        Session mesajını kalıcı olarak kaydeder.

        Parametreler:
            session_id: Mesajın ait olduğu oturum
            role: user | assistant | system
            content: Mesaj metni
            user_state: O andaki kullanıcı state bilgisi
            detected_topic: Mesajdan çıkarılan konu
            message_type: question, answer, reflection vb.
            llm_confidence: Varsa model güven skoru

        Döner:
            MessageRecord
        """
        message = MessageRecord(
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            user_state=user_state,
            detected_topic=detected_topic,
            message_type=message_type,
            llm_confidence=llm_confidence,
        )

        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    # ============================================================
    # INTERVENTION
    # ============================================================

    def save_intervention(
        self,
        session_id: str,
        user_id: str,
        intervention_type: str,
        message: str,
        triggered_by: Optional[str] = None,
        reason: Optional[str] = None,
        confidence: Optional[float] = None,
        was_successful: Optional[bool] = None,
    ) -> InterventionRecord:
        """
        Mentor müdahalesini DB'ye kaydeder.
        """
        intervention = InterventionRecord(
            session_id=session_id,
            user_id=user_id,
            intervention_type=intervention_type,
            message=message,
            triggered_by=triggered_by,
            reason=reason,
            confidence=confidence,
            was_successful=was_successful,
            timestamp=datetime.utcnow(),
        )

        self.db.add(intervention)

        session = self.get_session(session_id)
        if session:
            session.intervention_count += 1

        self.db.commit()
        self.db.refresh(intervention)
        return intervention

    # ============================================================
    # BEHAVIOR EVENT
    # ============================================================

    def log_behavior_event(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        state_before: Optional[str] = None,
        state_after: Optional[str] = None,
        topic: Optional[str] = None,
        severity: Optional[float] = None,
        metadata_json: str = "{}",
    ) -> BehaviorEventRecord:
        """
        Davranış analizi için event kaydeder.
        """
        event = BehaviorEventRecord(
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            state_before=state_before,
            state_after=state_after,
            topic=topic,
            severity=severity,
            metadata_json=metadata_json,
            created_at=datetime.utcnow(),
        )

        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    # ============================================================
    # FOCUS EVENT
    # ============================================================

    def log_focus_event(
        self,
        session_id: str,
        user_id: str,
        focus_score: float,
        source: str = "text",
        state_label: Optional[str] = None,
    ) -> FocusEventRecord:
        """
        Focus zaman serisine yeni skor ekler.
        """
        event = FocusEventRecord(
            session_id=session_id,
            user_id=user_id,
            focus_score=focus_score,
            source=source,
            state_label=state_label,
            created_at=datetime.utcnow(),
        )

        self.db.add(event)

        session = self.get_session(session_id)
        if session:
            session.average_focus_score = self._compute_average_focus_score(
                session_id=session_id,
                pending_score=focus_score,
            )

        self.db.commit()
        self.db.refresh(event)
        return event

    # ============================================================
    # INTERNAL
    # ============================================================

    def _compute_average_focus_score(
        self,
        session_id: str,
        pending_score: Optional[float] = None,
    ) -> Optional[float]:
        """
        Session için ortalama focus score hesaplar.
        """
        rows = (
            self.db.query(FocusEventRecord)
            .filter(FocusEventRecord.session_id == session_id)
            .all()
        )

        scores = [row.focus_score for row in rows]
        if pending_score is not None:
            scores.append(pending_score)

        if not scores:
            return None

        return sum(scores) / len(scores)