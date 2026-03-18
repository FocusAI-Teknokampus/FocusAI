# backend/agents/session_agent.py
#
# Oturum bağlamını yönetir.
# - Oturum başlayınca ShortTermContext oluşturur
# - Long-term memory'den UserProfile yükler
# - Her mesajda context'i günceller
# - Oturum kapanınca önemli olayları Mem0'a yazar
#
# Sahip: K1
# Bağımlılıklar: schemas.py, memory/short_term.py (K2), memory/long_term.py (K2)

from datetime import datetime
from typing import Optional

from backend.core.schemas import (
    LearningPattern,
    MemoryEntry,
    SessionEndRequest,
    SessionStartRequest,
    SessionStartResponse,
    ShortTermContext,
    UserProfile,
    UserState,
)
from backend.core.config import settings


# Aktif oturumları RAM'de tutan basit store
# Gerçek uygulamada Redis veya uygulama state'i kullanılır
_active_sessions: dict[str, ShortTermContext] = {}


class SessionAgent:
    """
    Oturum yaşam döngüsünü yönetir.

    start_session() → oturum açar, context oluşturur
    load_context()  → aktif context'i döner
    load_profile()  → long-term memory'den profil okur
    update_context() → her mesajda context günceller
    end_session()   → özetler, memory'ye yazar, temizler

    Kullanım (graph.py'daki session_node içinde):
        agent = SessionAgent()
        context = agent.load_context(session_id)
        profile = agent.load_profile(user_id)
    """

    def start_session(
        self, request: SessionStartRequest
    ) -> SessionStartResponse:
        """
        POST /session/start endpoint'i bu metodu çağırır.
        Yeni bir ShortTermContext oluşturur ve RAM'e yazar.
        """
        from uuid import uuid4

        session_id = str(uuid4())

        context = ShortTermContext(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            messages=[],
            current_state=UserState.UNKNOWN,
            retry_count=0,
            topics_covered=[request.topic] if request.topic else [],
        )

        _active_sessions[session_id] = context

        return SessionStartResponse(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_enabled=request.camera_enabled,
        )

    def load_context(self, session_id: str) -> Optional[ShortTermContext]:
        """
        Graph'ın session_node'u her mesajda bunu çağırır.
        Session yoksa None döner — API katmanı bunu handle etmeli.
        """
        return _active_sessions.get(session_id)

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Long-term memory'den kullanıcı profilini yükler.
        Hafta 2'de K2'nin LongTermMemory sınıfına bağlanacak.
        Şimdilik: yeni kullanıcı için varsayılan profil döner.
        """
        # Hafta 2'de burası dolacak:
        # from backend.memory.long_term import LongTermMemory
        # memory = LongTermMemory()
        # return memory.get_profile(user_id)

        # Şimdilik varsayılan profil
        return UserProfile(
            user_id=user_id,
            preferred_explanation_style="detailed",
            weak_topics=[],
            strong_topics=[],
            recurring_misconceptions=[],
            adaptive_threshold=settings.default_uncertainty_threshold,
            total_sessions=0,
        )

    def update_context(
        self,
        session_id: str,
        role: str,                      # "user" veya "assistant"
        content: str,
        new_state: Optional[UserState] = None,
        topic: Optional[str] = None,
    ) -> None:
        """
        Her mesaj alışverişinde çağrılır.
        Context'e mesaj ekler, durumu günceller.
        Mesaj sayısı max_messages'ı geçerse eskiyi atar.
        """
        context = _active_sessions.get(session_id)
        if not context:
            return

        # Mesajı ekle
        context.messages.append({"role": role, "content": content})

        # Max mesaj sınırını uygula — eski mesajları at
        if len(context.messages) > settings.short_term_max_messages:
            # İlk mesajı koru (oturum başlangıcı), ortadan kırp
            context.messages = (
                context.messages[:1]
                + context.messages[-(settings.short_term_max_messages - 1):]
            )

        # State güncelle
        if new_state:
            context.current_state = new_state

        # Yeni konu ekle
        if topic and topic not in context.topics_covered:
            context.topics_covered.append(topic)

    def end_session(self, request: SessionEndRequest) -> dict:
        """
        POST /session/end endpoint'i bunu çağırır.
        1. Önemli olayları MemoryEntry olarak Mem0'a yazar
        2. SQLite'a oturum özeti kaydeder
        3. RAM'den siler
        """
        context = _active_sessions.get(request.session_id)
        if not context:
            return {"status": "not_found"}

        # Önemli olayları çıkar ve memory'ye yaz
        entries = self._extract_memory_entries(context)
        self._write_to_memory(request.user_id, request.session_id, entries)

        # Temizle
        del _active_sessions[request.session_id]

        return {
            "status": "ended",
            "memory_entries_written": len(entries),
            "topics_covered": context.topics_covered,
        }

    def get_recent_messages_as_text(
        self, session_id: str, last_n: int = 6
    ) -> str:
        """
        Son N mesajı LLM prompt'u için düz metin formatına çevirir.
        MentorAgent'ın prompt oluştururken kullanır.
        """
        context = _active_sessions.get(session_id)
        if not context:
            return ""

        recent = context.messages[-last_n:]
        lines = []
        for msg in recent:
            prefix = "Kullanıcı" if msg["role"] == "user" else "Mentor"
            lines.append(f"{prefix}: {msg['content']}")

        return "\n".join(lines)

    # ── Private metodlar ─────────────────────────────────────────────

    def _extract_memory_entries(
        self, context: ShortTermContext
    ) -> list[MemoryEntry]:
        """
        Context'teki önemli olayları MemoryEntry listesine dönüştür.
        Şimdilik basit kurallar — Hafta 2'de LLM destekli analiz eklenecek.
        """
        entries = []

        # Yüksek retry → muhtemelen bir misconception vardı
        if context.retry_count >= 3:
            entries.append(MemoryEntry(
                user_id=context.user_id,
                session_id=context.session_id,
                entry_type="high_retry",
                content=(
                    f"{context.retry_count} kez aynı konuda soru soruldu. "
                    f"Konu: {context.topic or 'bilinmiyor'}"
                ),
                topic=context.topic,
            ))

        # İşlenen konuları kaydet
        for topic in context.topics_covered:
            entries.append(MemoryEntry(
                user_id=context.user_id,
                session_id=context.session_id,
                entry_type="topic_studied",
                content=f"{topic} konusu çalışıldı.",
                topic=topic,
            ))

        return entries

    def _write_to_memory(
        self,
        user_id: str,
        session_id: str,
        entries: list[MemoryEntry],
    ) -> None:
        """
        MemoryEntry listesini Mem0'a yazar.
        Hafta 2'de K2'nin LongTermMemory sınıfına bağlanacak.
        """
        # Hafta 2'de burası dolacak:
        # from backend.memory.long_term import LongTermMemory
        # memory = LongTermMemory()
        # for entry in entries:
        #     memory.write(entry)
        pass