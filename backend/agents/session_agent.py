# backend/agents/session_agent.py

from uuid import uuid4
from typing import Optional
import json

from sqlalchemy.orm import Session

from backend.core.schemas import (
    MemoryEntry,
    SessionEndRequest,
    SessionStartRequest,
    SessionStartResponse,
    ShortTermContext,
    UserProfile,
    UserState,
)
from backend.memory.short_term import ShortTermMemory, _sessions as _active_sessions
from backend.memory.long_term import LongTermMemory
from backend.services.session_service import SessionService
from backend.services.analytics_service import AnalyticsService
from backend.core.database import SessionLocal, UserProfileRecord


class SessionAgent:
    """
    Oturum yaşam döngüsünü yöneten ana agent.

    Sorumlulukları:
        - yeni session başlatmak
        - aktif session context'ini RAM'de tutmak
        - long-term profile okumak
        - chat sırasında context güncellemek
        - session bitince memory + DB işlemlerini tamamlamak

    Bu sınıf mevcut repo mimarisine uygundur:
        router -> SessionAgent -> ShortTermMemory / LongTermMemory / DB services
    """

    def __init__(self):
        """
        Kısa ve uzun dönem hafıza katmanlarını başlatır.
        """
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    # ============================================================
    # SESSION START
    # ============================================================

    def start_session(
        self,
        request: SessionStartRequest,
        db: Session,
    ) -> SessionStartResponse:
        """
        Yeni bir öğrenme oturumu başlatır.

        Akış:
            1. session_id üret
            2. RAM'de ShortTermContext oluştur
            3. DB'de session kaydı oluştur
            4. API response dön

        Parametreler:
            request: Session start isteği
            db: Request'e özel SQLAlchemy session

        Döner:
            SessionStartResponse
        """
        session_id = str(uuid4())

        # RAM'deki aktif oturum bağlamı
        context = ShortTermContext(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            messages=[],
            current_state=UserState.UNKNOWN,
            retry_count=0,
            topics_covered=[request.topic] if request.topic else [],
        )
        self.short_term.save(context)

        # DB kaydı
        service = SessionService(db)
        service.create_session(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_used=request.camera_enabled,
        )

        return SessionStartResponse(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_enabled=request.camera_enabled,
        )

    # ============================================================
    # LOADERS
    # ============================================================

    def load_context(self, session_id: str) -> Optional[ShortTermContext]:
        """
        Aktif session context'ini RAM'den getirir.

        Parametreler:
            session_id: Oturum kimliği

        Döner:
            ShortTermContext | None
        """
        return self.short_term.get(session_id)

    def load_profile(self, user_id: str, db: Optional[Session] = None) -> UserProfile:
        """
        Kullanıcı profilini yükler.

        Öncelik sırası:
            1. LongTermMemory / Mem0
            2. DB'deki user_profiles tablosu
            3. varsayılan boş profil

        Parametreler:
            user_id: Kullanıcı kimliği
            db: Opsiyonel DB session

        Döner:
            UserProfile
        """
        # 1) Önce long-term memory dene
        owns_session = db is None
        active_db = db or SessionLocal()

        try:
            profile = self.long_term.get_profile(user_id)
            if profile:
                return profile
        except Exception:
            pass

        # 2) DB fallback
        # Graph iÃ§inden gelen Ã§aÄŸrÄ±lar her zaman request DB session'i taÅŸÄ±mÄ±yor.
        # Bu nedenle burada kontrollÃ¼ bir local session aÃ§Ä±p profilin gerÃ§ekten okunmasÄ±nÄ± saÄŸlÄ±yoruz.
        try:
            row = (
                active_db.query(UserProfileRecord)
                .filter(UserProfileRecord.user_id == user_id)
                .first()
            )

            if row:
                return UserProfile(
                    user_id=user_id,
                    preferred_explanation_style=row.preferred_explanation_style or "detailed",
                    weak_topics=json.loads(row.weak_topics or "[]"),
                    strong_topics=json.loads(row.strong_topics or "[]"),
                    recurring_misconceptions=json.loads(row.recurring_misconceptions or "[]"),
                    avg_session_duration_minutes=row.avg_session_duration_minutes or 0.0,
                    adaptive_threshold=row.adaptive_threshold or 0.75,
                    total_sessions=0,
                    last_session_at=None,
                )
        finally:
            if owns_session:
                active_db.close()

        # 3) Varsayılan fallback
        return UserProfile(user_id=user_id)

    # ============================================================
    # CONTEXT UPDATE
    # ============================================================

    def update_context(
        self,
        session_id: str,
        role: str,
        content: str,
        db: Optional[Session] = None,
        new_state: Optional[UserState] = None,
        topic: Optional[str] = None,
        message_type: Optional[str] = None,
        llm_confidence: Optional[float] = None,
    ) -> None:
        """
        Aktif session context'ini günceller.

        Yapılanlar:
            - RAM'de mesaja ekleme yapar
            - state değiştiyse RAM context'ini günceller
            - yeni topic geldiyse ekler
            - DB verilmişse message kaydını kalıcı yazar
            - DB verilmişse session state bilgisini de günceller

        Parametreler:
            session_id: Oturum kimliği
            role: user | assistant | system
            content: Mesaj içeriği
            db: Opsiyonel veritabanı session'ı
            new_state: Yeni kullanıcı durumu
            topic: Yeni tespit edilen konu
            message_type: question / answer / reflection vb.
            llm_confidence: LLM güven skoru
        """
        # RAM
        self.short_term.add_message(session_id, role, content)

        if new_state is not None:
            self.short_term.update_state(session_id, new_state)

        if topic:
            self.short_term.add_topic(session_id, topic)

        # DB
        if db is not None:
            context = self.short_term.get(session_id)
            current_state_value = context.current_state.value if context else None

            service = SessionService(db)
            service.save_message(
                session_id=session_id,
                role=role,
                content=content,
                user_state=current_state_value,
                detected_topic=topic,
                message_type=message_type,
                llm_confidence=llm_confidence,
            )

            if context is not None and new_state is not None:
                service.update_session_state(
                    session_id=session_id,
                    state=new_state.value,
                    retry_count=context.retry_count,
                )

    # ============================================================
    # SESSION END
    # ============================================================

    def end_session(
        self,
        request: SessionEndRequest,
        db: Session,
    ) -> dict:
        """
        Aktif oturumu kapatır.

        Akış:
            1. RAM'den context al
            2. önemli olayları MemoryEntry listesine çevir
            3. long-term memory'ye yaz
            4. DB'de session'ı kapat
            5. session report üret
            6. RAM'den temizle

        Parametreler:
            request: Session end isteği
            db: Request'e özel DB session

        Döner:
            dict:
                status
                memory_entries_written
                topics_covered
                summary_text
        """
        context = self.short_term.get(request.session_id)

        if not context:
            return {"status": "not_found"}

        session_entry = MemoryEntry(
            user_id=request.user_id,
            session_id=request.session_id,
            entry_type="session_completed",
            content=(
                f"Oturum tamamlandı. "
                f"Konu: {context.topic or 'genel'}. "
                f"Toplam mesaj: {len(context.messages)}."
            ),
            topic=context.topic,
        )

        entries = self._extract_memory_entries(context)
        entries.append(session_entry)

        written = self._write_to_memory(
            user_id=request.user_id,
            session_id=request.session_id,
            entries=entries,
        )

        final_state = context.current_state.value if context.current_state else None

        session_service = SessionService(db)
        analytics_service = AnalyticsService(db)

        session_service.end_session(
            session_id=request.session_id,
            final_state=final_state,
        )

        report = analytics_service.finalize_session_report(request.session_id)

        # FeatureExtractor'ın session bazlı sayaçlarını temizle
        try:
            from backend.agents.graph import _feature_extractor
            _feature_extractor.reset_session(request.session_id)
        except Exception:
            pass

        self.short_term.delete(request.session_id)

        return {
            "status": "ended",
            "memory_entries_written": written,
            "topics_covered": context.topics_covered,
            "summary_text": report.summary_text if report else None,
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def get_recent_messages_as_text(self, session_id: str, last_n: int = 6) -> str:
        """
        Son N mesajı prompt için düz metne dönüştürür.
        """
        messages = self.short_term.get_recent_messages(session_id, last_n)

        lines = []
        for msg in messages:
            prefix = "Kullanıcı" if msg["role"] == "user" else "Mentor"
            lines.append(f"{prefix}: {msg['content']}")

        return "\n".join(lines)

    def _extract_memory_entries(self, context: ShortTermContext) -> list[MemoryEntry]:
        """
        Session context içinden önemli hafıza olaylarını çıkarır.
        """
        entries: list[MemoryEntry] = []

        if context.retry_count >= 3:
            entries.append(
                MemoryEntry(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    entry_type="high_retry",
                    content=(
                        f"{context.retry_count} kez aynı konuda soru soruldu. "
                        f"Konu: {context.topic or 'bilinmiyor'}"
                    ),
                    topic=context.topic,
                )
            )

        for topic in context.topics_covered:
            entries.append(
                MemoryEntry(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    entry_type="topic_studied",
                    content=f"{topic} konusu çalışıldı.",
                    topic=topic,
                )
            )

        return entries

    def _write_to_memory(
        self,
        user_id: str,
        session_id: str,
        entries: list[MemoryEntry],
    ) -> int:
        """
        MemoryEntry listesini long-term memory katmanına yazar.
        """
        try:
            return self.long_term.write_batch(entries)
        except Exception:
            return 0


# Geriye dönük uyumluluk
_active_sessions = _active_sessions
